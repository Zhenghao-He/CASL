#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/tmp/casl_hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/tmp/casl_hf/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/casl_hf/transformers}"
PYTHON_BIN="${PYTHON_BIN:-/u/wff7ad/program/anaconda3/bin/python}"

"${PYTHON_BIN}" -m sdxl_rebuttal.experiment all \
  --config sdxl_rebuttal/configs/smoke.yaml \
  --device "${DEVICE:-cuda:0}"
"${PYTHON_BIN}" -m sdxl_rebuttal.evaluate \
  --config sdxl_rebuttal/configs/smoke.yaml \
  --device "${DEVICE:-cuda:0}"
"${PYTHON_BIN}" -m sdxl_rebuttal.analyze \
  --config sdxl_rebuttal/configs/smoke.yaml
