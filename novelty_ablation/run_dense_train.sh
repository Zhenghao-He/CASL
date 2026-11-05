#!/usr/bin/env bash
set -euo pipefail

concept="${1:?concept required}"
gpu="${2:-0}"
n_train="${3:-100}"
n_iter="${4:-5}"
run_tag="${5:-novelty_dense}"
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"

export CUDA_VISIBLE_DEVICES="$gpu"
export XDG_CACHE_HOME=/p/realai/zhenghao/cache
python main.py \
  --run_train \
  --config celeba_p2.yml \
  --exp "./runs/${run_tag}_${concept}" \
  --edit_attr "$concept" \
  --do_train 1 \
  --do_test 0 \
  --n_train_img "$n_train" \
  --n_test_img 32 \
  --n_iter "$n_iter" \
  --bs_train 1 \
  --t_0 999 \
  --n_inv_step 50 \
  --n_train_step 50 \
  --n_test_step 100 \
  --get_h_num 1 \
  --user_defined_t_edit 500 \
  --user_defined_t_addnoise 200 \
  --train_delta_h \
  --additive_dense_h \
  --save_x0 \
  --use_x0_tensor \
  --lr_training 0.5 \
  --clip_loss_w 1.0 \
  --l1_loss_w 3.0 \
  --activation ReLU \
  --k 4096 \
  --scale 128 \
  --alpha 32 \
  --retrain 1 \
  --sh_file_name script_train.sh
