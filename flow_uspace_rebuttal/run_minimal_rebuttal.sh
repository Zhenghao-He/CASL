#!/usr/bin/env bash
set -euo pipefail

# Run from Asyrp_SAE. Set GPU before invoking, for example:
# CUDA_VISIBLE_DEVICES=0 bash flow_uspace_rebuttal/run_minimal_rebuttal.sh smiling 1 -1 2
concept=${1:?concept}
topk=${2:?topk}
casl_alpha=${3:?casl alpha}
dense_alpha=${4:?dense alpha}

python flow_uspace_rebuttal/run.py edit \
  --config flow_uspace_rebuttal/config_large.yaml \
  --concepts "$concept" \
  --test-images 32 \
  --test-ids-file "flow_uspace_rebuttal/manifests/${concept}_negative32.txt" \
  --manifest-tag minimal32_e100 \
  --euler-steps 100 \
  --topks "$topk" \
  --casl-strengths "$casl_alpha" \
  --dense-strengths "$dense_alpha" \
  --methods dense casl

python flow_uspace_rebuttal/evaluate_minimal.py \
  --config flow_uspace_rebuttal/config_large.yaml \
  --manifests "manifest_${concept}_minimal32_e100.csv" \
  --concepts "$concept" \
  --output-prefix "minimal32_e100_${concept}" \
  --device cuda:0 \
  --batch-size 64 \
  --with-preservation
