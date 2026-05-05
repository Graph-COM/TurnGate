#!/bin/bash
set -euo pipefail

echo "=== Experiment: No Defense ==="

export CUDA_LAUNCH_BLOCKING=1
export CUDA_CACHE_PATH=/tmp/cuda_cache
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

cd "$(dirname "${BASH_SOURCE[0]}")"
# source .venv/bin/activate

python main.py --config config/config_no_defense.yml --phase full --verbose

echo "=== Experiment finished ==="
