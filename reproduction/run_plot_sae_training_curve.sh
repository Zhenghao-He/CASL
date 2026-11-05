#!/bin/bash

set -euo pipefail

HISTORY_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_sae"
OUTPUT_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_sae_curves"
MAX_EPOCH=100

python /p/realai/zhenghao/LoRA_SAE/plot_sae_training_curve.py \
  --history-dir "$HISTORY_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --max-epoch "$MAX_EPOCH"
