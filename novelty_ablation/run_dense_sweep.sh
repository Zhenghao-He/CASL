#!/usr/bin/env bash
set -euo pipefail

concept="${1:?concept required}"
gpu="${2:-0}"
n_images="${3:-8}"
strengths="${4:--2,-1.5,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2}"
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"

export CUDA_VISIBLE_DEVICES="$gpu"
export XDG_CACHE_HOME=/p/realai/zhenghao/cache
checkpoint_name="sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/novelty_dense_${concept}_LC_CelebA_HQ_P2_t999_ninv50_ngen50_4.pth"

IFS=',' read -ra values <<< "$strengths"
for strength in "${values[@]}"; do
  # run_test multiplies by n_train_step/n_test_step = 0.5.
  coefficient="$(python -c "print(2.0 * float('$strength'))")"
  tag="${strength//-/_neg_}"
  tag="${tag//./p}"
  python main.py \
    --run_test \
    --config celeba_p2.yml \
    --exp "./runs/novelty_dense_sweep_${concept}_${tag}" \
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
    --hs_coeff_delta_h "$coefficient" \
    --activation ReLU \
    --k 4096 \
    --scale 128 \
    --alpha 32 \
    --sh_file_name script_inference.sh
done
