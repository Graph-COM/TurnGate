# Online-Battle

Evaluation framework for TurnGate experiments. Use it to run jailbreak attacks, apply defenses, and score results. This codebase is adapted from the original [CKA-Agent codebase](https://github.com/Graph-COM/CKA-Agent)

## Common Commands

```bash
python main.py --config config/config_no_defense.yml
python main.py --config config/config_rl_defense.yml
python main.py --config config/config_benchmark_shift_harmbench.yml
export GEMINI_API_KEY="your-key"
python main.py --config config/config_target_shift_gemini.yml
python main.py --config config/config_attacker_shift_maj.yml
```

## Main Areas

- `config/` — experiment, method, model, and dataset configs
- `methods/` — attack implementations
- `defense/` — defense wrappers
- `evaluation/` — scoring and metrics
- `data/` — datasets used by the experiments

## Docs

- [config/README.md](config/README.md)
- [methods/README.md](methods/README.md)
- [scripts/README.md](scripts/README.md)
