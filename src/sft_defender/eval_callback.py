"""
Defense evaluation callback for SFT training.

Handles:
- GPU placeholder management (occupy eval GPU at start, free for eval, re-occupy after)
- Periodic defense evaluation via vLLM subprocess
- W&B logging of defense metrics
- Checkpoint saving at epoch end

DDP-safe: registered on ALL ranks. All ranks participate in checkpoint saving
and barrier synchronization; only rank 0 runs the vLLM subprocess.
"""

import os
import sys
import json
import math
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from transformers import TrainerCallback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.evaluator import compute_fci_metrics


def _log(msg: str):
    if int(os.environ.get("LOCAL_RANK", -1)) in (-1, 0):
        print(msg)


def _resolve_physical_gpu_id(cuda_device_index: int) -> str:
    """Map a CUDA device index to a physical GPU ID.

    If CUDA_VISIBLE_DEVICES=4,5,6, then cuda:2 -> physical GPU "6".
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        gpu_ids = [g.strip() for g in cuda_visible.split(",")]
        if cuda_device_index < len(gpu_ids):
            return gpu_ids[cuda_device_index]
    return str(cuda_device_index)


def _flatten_metrics(organized: dict, prefix: str) -> dict:
    """Flatten nested Evaluator metrics into W&B-compatible flat keys."""
    flat = {}
    for level_key in ("CONVERSATION_LEVEL", "TURN_LEVEL"):
        level_data = organized.get(level_key, {})
        for category, metrics in level_data.items():
            for metric_name, value in metrics.items():
                if value is not None:
                    flat[f"{prefix}/{category}_{metric_name}"] = value
    # OVERALL metrics (F1, precision, recall, etc.)
    for metric_name, value in organized.get("OVERALL", {}).items():
        if value is not None:
            flat[f"{prefix}/overall_{metric_name}"] = value
    return flat


class GPUPlaceholder:
    """Manages a placeholder tensor on a GPU to prevent others from using it."""

    def __init__(self, device_index: int, size_gb: float = 72.0):
        self.device_index = device_index
        self.size_gb = size_gb
        self._tensor = None

    def occupy(self):
        """Allocate a large tensor to occupy the GPU."""
        if self._tensor is not None:
            return
        num_elements = int(self.size_gb * 1e9 / 4)  # float32 = 4 bytes
        try:
            self._tensor = torch.zeros(
                num_elements,
                dtype=torch.float32,
                device=f"cuda:{self.device_index}",
            )
            _log(
                f"  GPU placeholder: occupied cuda:{self.device_index} (~{self.size_gb:.0f}GB)"
            )
        except RuntimeError as e:
            _log(f"  Warning: Could not occupy cuda:{self.device_index}: {e}")
            _log(f"  Trying smaller placeholder (10GB)...")
            try:
                num_elements = int(10 * 1e9 / 4)
                self._tensor = torch.zeros(
                    num_elements,
                    dtype=torch.float32,
                    device=f"cuda:{self.device_index}",
                )
                _log(f"  GPU placeholder: occupied cuda:{self.device_index} (~10GB)")
            except RuntimeError:
                _log(
                    f"  Warning: Could not occupy GPU at all. Proceeding without placeholder."
                )

    def release(self):
        """Free the placeholder tensor."""
        if self._tensor is not None:
            del self._tensor
            self._tensor = None
            torch.cuda.empty_cache()
            _log(f"  GPU placeholder: released cuda:{self.device_index}")

    def __del__(self):
        self.release()


class DefenseEvalCallback(TrainerCallback):
    """
    HF Trainer callback for periodic defense evaluation during SFT training.

    DDP-safe: This callback is registered on ALL ranks. All ranks participate
    in checkpoint saving and barrier synchronization. Only rank 0 runs the
    actual vLLM evaluation subprocess, GPU placeholder management, and W&B logging.

    At every eval_steps:
        - Save temp checkpoint -> launch vLLM subprocess -> eval on val -> delete checkpoint
    At every epoch end:
        - Save checkpoint (kept) -> launch vLLM subprocess -> eval on val + test
    When epoch end coincides with eval_steps:
        - Only one checkpoint save, one vLLM load, run both val + test
    """

    def __init__(
        self,
        trainer,
        config,
        eval_steps: int,
        steps_per_epoch: int,
        eval_gpu_index: int,
        eval_data_dir: str,
        output_dir: str,
    ):
        super().__init__()
        self.trainer = trainer
        self.config = config
        self.eval_steps = eval_steps
        self.steps_per_epoch = steps_per_epoch
        self.eval_gpu_index = eval_gpu_index
        self.eval_physical_gpu = _resolve_physical_gpu_id(eval_gpu_index)
        self.eval_data_dir = eval_data_dir
        self.output_dir = Path(output_dir)

        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.is_main_process = local_rank in (-1, 0)
        self.is_distributed = dist.is_initialized()

        # GPU placeholder (only main process manages it)
        self.gpu_placeholder = None
        if self.is_main_process:
            self.gpu_placeholder = GPUPlaceholder(eval_gpu_index)
            self.gpu_placeholder.occupy()

        # Track state
        self._last_eval_step = -1
        self._current_epoch = 0
        self.fci_tipping_point = float(os.environ.get("FCI_TIPPING_POINT", "3.0"))
        self.fci_eps = float(os.environ.get("FCI_EPS", "1e-5"))
        self.best_val_fci = None
        self.best_test_fci = None
        self.best_checkpoint_dir = self.output_dir / "best_by_val_fci"
        self.best_test_metrics = {}

        # Verify eval data exists (only main process needs to check)
        self.has_train_eval = False
        if self.is_main_process:
            for split in ["val", "test"]:
                pkl_path = Path(eval_data_dir) / f"{split}_conversations.pkl"
                if not pkl_path.exists():
                    raise FileNotFoundError(
                        f"Eval data not found: {pkl_path}\n"
                        f"Run: python src/sft_defender/prepare_eval_data.py "
                        f"--dataset-dir <dataset_dir> --output-dir {eval_data_dir}"
                    )
            # Train subset eval is optional
            train_pkl = Path(eval_data_dir) / "train_conversations.pkl"
            self.has_train_eval = train_pkl.exists()
            if self.has_train_eval:
                _log(f"Train eval data found: {train_pkl}")

    def _is_epoch_end(self, global_step: int) -> bool:
        """Check if this step is the last step of an epoch."""
        return global_step > 0 and global_step % self.steps_per_epoch == 0

    def _save_checkpoint(self, path: str, save_training_state: bool = False):
        """Save model checkpoint using trainer (DDP/DeepSpeed safe).

        trainer.save_model() also saves the tokenizer when it's set on the trainer.
        When save_training_state=True, also saves optimizer, scheduler, and
        trainer state for full resume support (LR schedule continuity).
        """
        _log(f"  Saving checkpoint to {path}...")
        self.trainer.save_model(path)

        if save_training_state and self.is_main_process:
            _log(f"  Saving training state (optimizer, scheduler)...")
            if self.trainer.optimizer is not None:
                torch.save(
                    self.trainer.optimizer.state_dict(),
                    os.path.join(path, "optimizer.pt"),
                )
            if self.trainer.lr_scheduler is not None:
                torch.save(
                    self.trainer.lr_scheduler.state_dict(),
                    os.path.join(path, "scheduler.pt"),
                )
            self.trainer.state.save_to_json(os.path.join(path, "trainer_state.json"))

    def _run_eval_subprocess(
        self, checkpoint_path: str, conversations_pkl: str
    ) -> dict:
        """Launch eval_defense.py as a subprocess on the eval GPU."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir="/tmp"
        ) as f:
            result_path = f.name

        eval_script = os.path.join(os.path.dirname(__file__), "eval_defense.py")

        cmd = [
            sys.executable,
            eval_script,
            "--checkpoint",
            checkpoint_path,
            "--eval-conversations",
            conversations_pkl,
            "--defense-batch-size",
            str(self.config.defense_batch_size),
            "--output",
            result_path,
            "--training-type",
            self.config.training_type,
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
        ]
        if self.config.training_type == "lora":
            cmd.extend(["--base-model", self.config.base_model])
        if self.config.hf_token:
            cmd.extend(["--hf-token", self.config.hf_token])
        if getattr(self.config, "mda_mode", False):
            cmd.append("--mda-mode")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.eval_physical_gpu
        # Force unbuffered output so we see subprocess logs in real-time
        env["PYTHONUNBUFFERED"] = "1"
        # Strip ALL distributed/torchrun env vars to prevent interference with vLLM
        for key in list(env.keys()):
            if key in (
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "LOCAL_WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "GROUP_RANK",
            ):
                del env[key]
            elif key.startswith("TORCHELASTIC_"):
                del env[key]

        _log(f"  Launching eval subprocess on GPU {self.eval_physical_gpu}...")
        _log(f"  Command: {' '.join(cmd)}")
        proc = None
        try:
            # Merge stderr into stdout to avoid pipe deadlock (vLLM logs to stderr
            # via Python logging; if stderr buffer fills while we read stdout, both block).
            # start_new_session=True detaches from torchrun's process group/signals.
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )

            # Stream all output (stdout + stderr) in real-time
            for line in proc.stdout:
                print(f"  [eval] {line}", end="", flush=True)

            proc.wait(timeout=1800)

            if proc.returncode != 0:
                _log(f"  Eval subprocess failed (rc={proc.returncode})")
                return {}

            with open(result_path, "r") as f:
                results = json.load(f)
            return results
        except subprocess.TimeoutExpired:
            _log("  Eval subprocess timed out (30 min)!")
            if proc is not None:
                proc.kill()
                proc.wait()
            return {}
        except Exception as e:
            _log(f"  Eval subprocess error: {e}")
            return {}
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
            if os.path.exists(result_path):
                os.unlink(result_path)

    def _evaluate_defense(self, global_step: int, splits: list):
        """Run defense evaluation on specified splits.

        DDP-safe: ALL ranks enter this method.
        - All ranks participate in checkpoint saving.
        - All ranks wait at barriers during evaluation.
        - Only rank 0 runs the vLLM subprocess, W&B logging, and cleanup.
        """
        # Prepend train subset if available
        if self.has_train_eval and "train" not in splits:
            splits = ["train"] + splits
        is_epoch_end = "test" in splits
        epoch_num = self._current_epoch

        # Determine checkpoint path
        if is_epoch_end:
            checkpoint_path = str(self.output_dir / f"epoch_{epoch_num}")
        else:
            checkpoint_path = str(self.output_dir / "tmp_eval_checkpoint")

        # ALL ranks save checkpoint (trainer.save_model handles DDP internally)
        # Save full training state (optimizer/scheduler) at epoch end for resume support
        self._save_checkpoint(checkpoint_path, save_training_state=is_epoch_end)

        # Barrier: ensure checkpoint is fully saved before eval starts
        if self.is_distributed:
            dist.barrier()

        # Only rank 0 runs the actual evaluation
        if self.is_main_process:
            self.gpu_placeholder.release()

            try:
                all_metrics = {}
                split_fci = {}
                split_metrics = {}
                for split in splits:
                    pkl_path = str(
                        Path(self.eval_data_dir) / f"{split}_conversations.pkl"
                    )
                    _log(f"\n  Running defense evaluation on {split} set...")

                    results = self._run_eval_subprocess(checkpoint_path, pkl_path)
                    if results:
                        flat = _flatten_metrics(results, prefix=split)
                        all_metrics.update(flat)
                        fci_metrics = compute_fci_metrics(
                            results,
                            prefix=split,
                            tipping_point=self.fci_tipping_point,
                            eps=self.fci_eps,
                        )
                        all_metrics.update(fci_metrics)
                        split_metrics[split] = {**flat, **fci_metrics}
                        if f"{split}/fci" in fci_metrics:
                            split_fci[split] = fci_metrics[f"{split}/fci"]
                        _log(f"  {split} results: {json.dumps(flat, indent=2)}")
                        if f"{split}/fci" in fci_metrics:
                            _log(f"  {split} FCI: {fci_metrics[f'{split}/fci']:.6f}")
                    else:
                        _log(f"  {split} evaluation returned no results.")

                # Update best checkpoint by val FCI (test never participates in selection).
                if "val" in split_fci:
                    self._update_best_by_val_fci(
                        checkpoint_path=checkpoint_path,
                        global_step=global_step,
                        val_fci=split_fci["val"],
                        test_fci=split_fci.get("test"),
                        test_metrics=split_metrics.get("test"),
                    )

                # Persist best summary to W&B.
                if self.best_val_fci is not None:
                    all_metrics["best/val_fci"] = self.best_val_fci
                if self.best_test_fci is not None:
                    all_metrics["best/test_fci"] = self.best_test_fci
                if self.best_test_metrics:
                    for key, value in self.best_test_metrics.items():
                        if key.startswith("test/"):
                            all_metrics[f"best_{key}"] = value

                # Log to W&B
                if all_metrics:
                    try:
                        import wandb

                        if wandb.run is not None:
                            wandb.log({"defense_step": global_step, **all_metrics})
                            _log(
                                f"  Logged {len(all_metrics)} defense metrics to W&B at step {global_step}"
                            )
                    except ImportError:
                        pass

            finally:
                self.gpu_placeholder.occupy()

                # Delete temp checkpoint (not epoch-end checkpoint)
                if not is_epoch_end and os.path.exists(checkpoint_path):
                    shutil.rmtree(checkpoint_path, ignore_errors=True)
                    _log(f"  Deleted temp checkpoint: {checkpoint_path}")

        # Barrier: ALL ranks wait for rank 0 to finish eval before resuming training
        if self.is_distributed:
            dist.barrier()

    def run_init_eval(self, checkpoint_path: str, global_step: int = 0):
        """Run val+test eval on an existing checkpoint before training starts.

        Unlike _evaluate_defense(), this does NOT save a new checkpoint — it uses
        the checkpoint already on disk at checkpoint_path. Useful for evaluating
        a resumed model without redundant saving.

        DDP-safe: all ranks enter, only rank 0 runs eval.
        """
        if self.is_distributed:
            dist.barrier()

        if self.is_main_process:
            self.gpu_placeholder.release()

            try:
                all_metrics = {}
                init_splits = ["val", "test"]
                if self.has_train_eval:
                    init_splits = ["train"] + init_splits
                for split in init_splits:
                    pkl_path = str(
                        Path(self.eval_data_dir) / f"{split}_conversations.pkl"
                    )
                    _log(
                        f"\n  [Init Eval] Running defense evaluation on {split} set..."
                    )

                    results = self._run_eval_subprocess(checkpoint_path, pkl_path)
                    if results:
                        flat = _flatten_metrics(results, prefix=f"init_{split}")
                        all_metrics.update(flat)
                        _log(f"  {split} results: {json.dumps(flat, indent=2)}")
                    else:
                        _log(f"  {split} evaluation returned no results.")

                # Log to W&B
                if all_metrics:
                    try:
                        import wandb

                        if wandb.run is not None:
                            wandb.log({"defense_step": global_step, **all_metrics})
                            _log(
                                f"  Logged {len(all_metrics)} init-eval metrics to W&B at step {global_step}"
                            )
                    except ImportError:
                        pass

            finally:
                self.gpu_placeholder.occupy()

        if self.is_distributed:
            dist.barrier()

    def on_step_end(self, args, state, control, **kwargs):
        """Check if we should run eval at this step (ALL ranks enter)."""
        if state.global_step <= 0:
            return

        # Check if this is an eval_steps boundary
        if state.global_step % self.eval_steps != 0:
            return

        # If this is also an epoch end, let on_epoch_end handle it
        if self._is_epoch_end(state.global_step):
            return

        _log(f"\n{'='*60}")
        _log(f"[Step {state.global_step}] Defense evaluation (val)")
        _log(f"{'='*60}")

        self._evaluate_defense(state.global_step, splits=["val"])
        self._last_eval_step = state.global_step

    def on_epoch_end(self, args, state, control, **kwargs):
        """Run val + test evaluation and save checkpoint at epoch end (ALL ranks enter)."""
        self._current_epoch += 1

        _log(f"\n{'='*60}")
        _log(
            f"[Epoch {self._current_epoch}] Defense evaluation (val + test) + checkpoint"
        )
        _log(f"{'='*60}")

        self._evaluate_defense(state.global_step, splits=["val", "test"])
        self._last_eval_step = state.global_step

    def on_train_end(self, args, state, control, **kwargs):
        """Clean up GPU placeholder."""
        if self.gpu_placeholder is not None:
            self.gpu_placeholder.release()
        _log("DefenseEvalCallback: cleaned up GPU placeholder.")

    def _update_best_by_val_fci(
        self,
        checkpoint_path: str,
        global_step: int,
        val_fci: float,
        test_fci: Optional[float] = None,
        test_metrics: Optional[dict] = None,
    ):
        # Strictly select best model by val FCI only.
        if self.best_val_fci is not None and val_fci < self.best_val_fci:
            return

        # Tie case: keep existing best checkpoint; only fill best_test_fci when available.
        if self.best_val_fci is not None and val_fci == self.best_val_fci:
            if self.best_test_fci is None and test_fci is not None:
                self.best_test_fci = test_fci
                if test_metrics:
                    self.best_test_metrics = dict(test_metrics)
                _log(
                    f"  [Best-By-Val-FCI] Backfilled best_test_fci={self.best_test_fci:.6f} "
                    f"at step={global_step}"
                )
            return

        _log(
            f"  [Best-By-Val-FCI] Update at step={global_step}: "
            f"{self.best_val_fci} -> {val_fci}"
        )

        tmp_best = str(self.best_checkpoint_dir) + "_tmp"
        if os.path.exists(tmp_best):
            shutil.rmtree(tmp_best, ignore_errors=True)
        shutil.copytree(checkpoint_path, tmp_best)

        if self.best_checkpoint_dir.exists():
            shutil.rmtree(self.best_checkpoint_dir, ignore_errors=True)
        os.rename(tmp_best, self.best_checkpoint_dir)

        self.best_val_fci = val_fci
        if test_fci is not None:
            self.best_test_fci = test_fci
            if test_metrics:
                self.best_test_metrics = dict(test_metrics)

        _log(
            f"  [Best-By-Val-FCI] Saved best checkpoint to: {self.best_checkpoint_dir}"
        )
