#!/bin/bash

set -euo pipefail

TOP100_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_concept_latents"
PROBE_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_linear_probe"
OUTPUT_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_overlap_summary"
PROBE_TOP_K=10

python /p/realai/zhenghao/LoRA_SAE/summarize_latent_overlap.py \
  --top100-dir "$TOP100_DIR" \
  --probe-dir "$PROBE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --probe-top-k "$PROBE_TOP_K"
