# Scripts

Shell scripts for running baselines, training, and batch experiments.

## Main Scripts

- `run_no_defense.sh` — baseline evaluation
- `run_rl_defense.sh` — TurnGate evaluation
- `train_naive_sft.sh` — naive SFT training
- `train_reweighted_sft.sh` — reweighted SFT training
- `train_turngate.sh` — TurnGate training
- `evaluate_all_baselines.sh` — baseline comparison

## Typical Use

```bash
cd online-battle
bash run_no_defense.sh
bash run_rl_defense.sh
```

From the repo root:

```bash
bash scripts/train_turngate.sh
bash scripts/evaluate_all_baselines.sh
```

## Notes

- Most scripts accept environment overrides such as `OUTPUT_DIR`.
- Set `GEMINI_API_KEY` before Gemini-based runs.
