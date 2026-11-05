#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
official_root="$(cd "${repo_root}/../Asyrp_official" && pwd)"
device="${CUDA_DEVICE:-0}"

cd "${official_root}"
CUDA_VISIBLE_DEVICES="${device}" python -u main.py \
  --run_train \
  --config afhq.yml \
  --exp ./runs/rebuttal_dog_puppy_train \
  --edit_attr dog_puppy \
  --do_train 1 \
  --do_test 0 \
  --n_train_img 100 \
  --n_test_img 32 \
  --n_iter 1 \
  --bs_train 1 \
  --t_0 999 \
  --n_inv_step 40 \
  --n_train_step 40 \
  --n_test_step 40 \
  --get_h_num 1 \
  --user_defined_t_edit 500 \
  --user_defined_t_addnoise 200 \
  --train_delta_block \
  --use_x0_tensor \
  --lr_training 0.5 \
  --clip_loss_w 1.0 \
  --l1_loss_w 3.0 \
  --retrain 0 \
  --sh_file_name "${repo_root}/rebuttal_metrics/train_asyrp_dog_puppy.sh"
