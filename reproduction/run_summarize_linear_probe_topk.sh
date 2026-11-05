#!/bin/bash

set -euo pipefail

INPUT_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_linear_probe_sweep"
OUTPUT_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_linear_probe_topk_summary"

python /p/realai/zhenghao/LoRA_SAE/summarize_linear_probe_topk.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR"
