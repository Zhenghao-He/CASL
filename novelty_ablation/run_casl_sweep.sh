#!/usr/bin/env bash
set -euo pipefail

concept="${1:?concept required}"
gpu="${2:-0}"
n_images="${3:-8}"
strengths="${4:-0,32,64,80,96,112,128,144,160,176,192}"
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"

export CUDA_VISIBLE_DEVICES="$gpu"
export XDG_CACHE_HOME=/p/realai/zhenghao/cache
python main.py \
  --sae_edit_only \
  --config celeba_p2.yml \
  --exp "./runs/${concept}" \
  --edit_attr "$concept" \
  --do_train 0 \
  --do_test 1 \
  --n_train_img 100 \
  --n_test_img "$n_images" \
  --n_iter 5 \
  --bs_train 1 \
  --t_0 999 \
  --n_inv_step 50 \
  --n_train_step 50 \
  --n_test_step 100 \
  --get_h_num 1 \
  --user_defined_t_edit 500 \
  --user_defined_t_addnoise 200 \
  --train_delta_block \
  --delta_h \
  --activation ReLU \
  --k 4096 \
  --scale 128 \
  --alpha 32 \
  --top_k_indice 1024 \
  --topk_to_analyze 1 \
  --sae_strengths="$strengths" \
  --sh_file_name script_inference.sh
