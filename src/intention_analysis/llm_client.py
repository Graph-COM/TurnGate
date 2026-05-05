"""
LLM client abstraction for Intention Analysis.
Implementations can use OpenAI API, vLLM (in-process), etc.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import concurrent.futures

# Debug mode: set SFT_DEBUG=1 to enable verbose logging
_SFT_DEBUG = os.environ.get("SFT_DEBUG", "0") == "1"


class LLMClient(ABC):
    """Abstract interface for calling an LLM with a single prompt or batch."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate text from the given prompt.

        Args:
            prompt: The input prompt string.
            max_tokens: Optional maximum tokens to generate.

        Returns:
            The generated text (e.g., intention or response).
        """
        pass

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """
        Generate text for a batch of prompts.
        Default implementation loops over generate.
        """
        return [self.generate(p, max_tokens) for p in prompts]

    def generate_with_messages(
        self, messages: List[Dict], max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from a messages list (role/content dicts).
        Default fallback: concatenate and call generate.
        """
        prompt = "\n".join(
            f"{m['role'].upper()}: {m.get('content', '')}" for m in messages
        )
        return self.generate(prompt, max_tokens)

    def generate_batch_with_messages(
        self,
        messages_list: List[List[Dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """
        Generate text for a batch of message lists.
        Subclasses should override for efficient batched inference.
        """
        return [self.generate_with_messages(m, max_tokens) for m in messages_list]


class OpenAILLMClient(LLMClient):
    """
    OpenAI API-based LLM client.
    Set OPENAI_API_KEY in environment or pass api_key to constructor.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens_default: int = 1024,
    ):
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_tokens_default = max_tokens_default

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        try:
            import os
            from openai import OpenAI

            client = OpenAI(
                api_key=self._api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=self._base_url,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens
                or self._max_tokens_default,  # may change to max_tokens if model are different
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

    def generate_batch(
        self, prompts: List[str], max_tokens: Optional[int] = None
    ) -> List[str]:
        """Use ThreadPoolExecutor to call OpenAI API in parallel."""
        # Cap max_workers to a reasonable limit (e.g., 64) to avoid OS overhead
        max_workers = min(len(prompts), 64)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.generate, prompt, max_tokens) for prompt in prompts
            ]
            return [f.result() for f in futures]

    def generate_with_messages(
        self, messages: List[Dict], max_tokens: Optional[int] = None
    ) -> str:
        """Call OpenAI chat completions with a full messages list."""
        try:
            import os
            from openai import OpenAI

            client = OpenAI(
                api_key=self._api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=self._base_url,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens or self._max_tokens_default,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

    def generate_batch_with_messages(
        self,
        messages_list: List[List[Dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """Use ThreadPoolExecutor to call OpenAI API in parallel with full message lists."""
        max_workers = min(len(messages_list), 64)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.generate_with_messages, messages, max_tokens)
                for messages in messages_list
            ]
            return [f.result() for f in futures]


class VLLMClient(LLMClient):
    """
    vLLM-backed client for open-source models (in-process).
    Loads model directly with vLLM for accelerated inference, no separate server needed.

    Supports:
    - Base models
    - Full fine-tuned models
    - LoRA adapters (via enable_lora + lora_modules)
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-4B-Instruct-2507",
        max_tokens_default: int = 1024,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 12288,
        enforce_eager: bool = False,
        enable_prefix_caching: bool = True,
        hf_token: Optional[str] = None,
        # LoRA-specific parameters
        enable_lora: bool = False,
        lora_modules: Optional[
            Dict[str, str]
        ] = None,  # {"adapter_name": "path/to/lora"}
        max_lora_rank: int = 64,
        **vllm_kwargs: Any,
    ):
        """
        Args:
            model: Base model name or path to full fine-tuned model
            enable_lora: Enable LoRA support in vLLM
            lora_modules: Dict mapping adapter names to LoRA checkpoint paths
                         e.g., {"sft": "checkpoints/sft_defender_lora/final_model"}
            max_lora_rank: Maximum LoRA rank (default 64)
            ... (other vLLM parameters)
        """
        self.model = model
        self._max_tokens_default = max_tokens_default
        self._hf_token = hf_token
        self._enable_lora = enable_lora
        self._lora_modules = lora_modules

        self._vllm_config = {
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "enable_prefix_caching": enable_prefix_caching,
            "disable_custom_all_reduce": vllm_kwargs.get(
                "disable_custom_all_reduce", True
            ),
            "disable_log_stats": vllm_kwargs.get("disable_log_stats", True),
        }

        # Add LoRA configuration if enabled
        if enable_lora:
            self._vllm_config["enable_lora"] = True
            self._vllm_config["max_lora_rank"] = max_lora_rank
            if lora_modules:
                self._vllm_config["lora_modules"] = lora_modules

        self._vllm_config.update(
            {
                k: v
                for k, v in vllm_kwargs.items()
                if k not in ("disable_custom_all_reduce", "disable_log_stats")
            }
        )
        self._vllm_model = None
        self._tokenizer = None  # Lazy load on first generate

    def _ensure_loaded(self):
        """Lazy-load vLLM model on first use."""
        if self._vllm_model is not None:
            return
        try:
            from vllm import LLM

            # Build vLLM initialization kwargs
            vllm_init_kwargs = {
                "model": self.model,
                "trust_remote_code": self._vllm_config.get("trust_remote_code", True),
                "tensor_parallel_size": self._vllm_config.get(
                    "tensor_parallel_size", 1
                ),
                "max_model_len": self._vllm_config.get("max_model_len", 12288),
                "gpu_memory_utilization": self._vllm_config.get(
                    "gpu_memory_utilization", 0.8
                ),
                "enforce_eager": self._vllm_config.get("enforce_eager", False),
                "enable_prefix_caching": self._vllm_config.get(
                    "enable_prefix_caching", True
                ),
                "disable_custom_all_reduce": self._vllm_config.get(
                    "disable_custom_all_reduce", True
                ),
                "disable_log_stats": self._vllm_config.get("disable_log_stats", True),
                "hf_token": self._hf_token,
            }

            # Add LoRA configuration if enabled
            if self._enable_lora:
                vllm_init_kwargs["enable_lora"] = True
                vllm_init_kwargs["max_lora_rank"] = self._vllm_config.get(
                    "max_lora_rank", 64
                )
                # Note: lora_modules should NOT be passed to LLM constructor
                # LoRA adapters are applied per-request via LoRARequest
                print(f"LoRA enabled with max_rank={vllm_init_kwargs['max_lora_rank']}")
                if self._lora_modules:
                    print(
                        f"LoRA adapters will be loaded: {list(self._lora_modules.keys())}"
                    )

            self._vllm_model = LLM(**vllm_init_kwargs)

            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=True, token=self._hf_token
            )
        except ImportError as e:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm"
            ) from e

    _format_prompt_logged = False  # Class-level flag for one-time debug logging

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt with chat template for instruct models."""
        thinking_disabled = False
        used_fallback = False
        try:
            messages = [{"role": "user", "content": prompt}]
            try:
                result = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                thinking_disabled = True
            except TypeError:
                # Fallback for tokenizers that don't support enable_thinking
                result = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                used_fallback = True
        except Exception:
            result = prompt

        # Debug: log the first formatted prompt to verify chat template
        if _SFT_DEBUG and not VLLMClient._format_prompt_logged:
            VLLMClient._format_prompt_logged = True
            print(f"\n[SFT_DEBUG] === VLLMClient._format_prompt (first call) ===")
            print(f"  enable_thinking=False applied: {thinking_disabled}")
            print(f"  Used fallback (no enable_thinking support): {used_fallback}")
            print(f"  Has <think> in output: {'<think>' in result}")
            print(f"  Prompt tail (last 150 chars): {repr(result[-150:])}")
            print(f"  Full formatted prompt length: {len(result)} chars")

        return result

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        return self.generate_batch([prompt], max_tokens)[0]

    def _vllm_generate(
        self,
        formatted_prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """Core vLLM generation from pre-formatted prompt strings."""
        from vllm import SamplingParams

        temp = temperature if temperature is not None else 0.7
        sampling_params = SamplingParams(
            max_tokens=max_tokens or self._max_tokens_default,
            temperature=temp,
            top_p=0.9 if temp > 0 else 1.0,  # top_p must be 1.0 when temperature=0
        )

        lora_request = None
        if self._enable_lora and self._lora_modules:
            from vllm.lora.request import LoRARequest

            adapter_name, adapter_path = list(self._lora_modules.items())[0]
            lora_request = LoRARequest(
                lora_name=adapter_name,
                lora_int_id=1,
                lora_path=adapter_path,
            )

        outputs = self._vllm_model.generate(
            formatted_prompts, sampling_params, lora_request=lora_request
        )
        return [(out.outputs[0].text or "").strip() for out in outputs]

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        try:
            self._ensure_loaded()
            formatted_prompts = [self._format_prompt(p) for p in prompts]
            return self._vllm_generate(formatted_prompts, max_tokens, temperature)
        except Exception as e:
            raise RuntimeError(f"vLLM generation failed: {e}") from e

    def generate_batch_with_messages(
        self,
        messages_list: List[List[Dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """
        Generate text for a batch of message lists by applying the model's own
        chat template directly to each messages list, bypassing the single-user-turn
        wrapper in _format_prompt. This is required for guard/moderation models
        (Llama Guard 3, Qwen3Guard) whose chat templates encode the safety policy.
        """
        try:
            self._ensure_loaded()
            formatted_prompts = []
            for messages in messages_list:
                try:
                    result = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    # Fallback for tokenizers that don't support enable_thinking
                    result = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                formatted_prompts.append(result)
            return self._vllm_generate(formatted_prompts, max_tokens, temperature)
        except Exception as e:
            raise RuntimeError(f"vLLM generation (messages) failed: {e}") from e

    def unload(self):
        """Unload the model from VRAM to free memory. Model will be re-loaded on next generate call."""
        if self._vllm_model is not None:
            try:
                # Delete the vLLM instance to free VRAM
                del self._vllm_model
                self._vllm_model = None
                import gc

                gc.collect()  # Force garbage collection
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass
                print(f"Unloaded model {self.model} from VRAM")
            except Exception as e:
                print(f"Warning: Failed to unload model: {e}")
        if self._tokenizer is not None:
            try:
                del self._tokenizer
                self._tokenizer = None
            except Exception as e:
                print(f"Warning: Failed to unload tokenizer: {e}")
