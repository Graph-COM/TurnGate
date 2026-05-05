# Configuration Guide

This directory holds the main experiment configs plus the method, model, and dataset sub-configs they reference.

## Main Configs

- `config_no_defense.yml` — baseline
- `config_rl_defense.yml` — TurnGate defense
- `config_benchmark_shift_harmbench.yml` — HarmBench shift
- `config_benchmark_shift_strongreject.yml` — StrongReject shift
- `config_benchmark_shift_jbb.yml` — JBB shift
- `config_target_shift_gemini.yml` — Gemini target shift
- `config_attacker_shift_maj.yml` — MAJ attacker shift

## Sub-Configs

- `method/cka-agent.yml` — CKA-Agent attack settings
- `method/maj-agent.yml` — MAJ attack settings
- `model/target_gemini.yml` — Gemini target model
- `dataset/*.yml` — benchmark dataset mappings

## Common Changes

- Reduce GPU use: lower `dataset.batch_size` in the main config.
- Change the judge: update `evaluation.judge_model`.
- Save intermediates: set `output.save_intermediate: true`.

## Quick Checks

- Run from `online-battle/`.
- Verify YAML with `python -c "import yaml; yaml.safe_load(open('config/config_rl_defense.yml'))"`.
- Set `GEMINI_API_KEY` for Gemini-based configs.
