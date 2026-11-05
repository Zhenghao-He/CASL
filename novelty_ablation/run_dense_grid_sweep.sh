#!/usr/bin/env bash
set -euo pipefail

concept="${1:?concept required}"
gpu="${2:-0}"
n_images="${3:-8}"
min_strength="${4:--2}"
max_strength="${5:-2}"
num_strengths="${6:-9}"
checkpoint_iter="${7:-4}"
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"

export CUDA_VISIBLE_DEVICES="$gpu"
export XDG_CACHE_HOME=/p/realai/zhenghao/cache
checkpoint_name="sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/novelty_dense_${concept}_LC_CelebA_HQ_P2_t999_ninv50_ngen50_${checkpoint_iter}.pth"
python main.py \
  --run_test \
  --config celeba_p2.yml \
  --exp "./runs/novelty_dense_grid_${concept}" \
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
  --train_delta_h \
  --additive_dense_h \
  --manual_checkpoint_name "$checkpoint_name" \
  --hs_coeff_delta_h 2 \
  --delta_interpolation \
  --min_delta "$min_strength" \
  --max_delta "$max_strength" \
  --num_delta "$num_strengths" \
  --activation ReLU \
  --k 4096 \
  --scale 128 \
  --alpha 32 \
  --sh_file_name script_inference.sh

python novelty_ablation/split_dense_grids.py \
  --concept "$concept" \
  --min-strength "$min_strength" \
  --max-strength "$max_strength" \
  --num-strengths "$num_strengths"
