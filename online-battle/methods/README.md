# Attack Methods

Supported methods:
- `cka-agent` — tree-search jailbreak baseline
- `multi-agent-jailbreak` — MAJ decomposition attack

## Usage

```bash
python main.py --config config/config_rl_defense.yml --method cka-agent
python main.py --config config/config_attacker_shift_maj.yml --method multi-agent-jailbreak
```

## Config Files

- `config/method/cka-agent.yml`
- `config/method/maj-agent.yml`

## Add a New Method

1. Implement `AbstractJailbreakMethod` in `methods/proposed/`.
2. Register it in `methods/method_factory.py`.
3. Add a matching config under `config/method/`.
