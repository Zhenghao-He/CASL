#!/usr/bin/env bash
set -euo pipefail

export XDG_CACHE_HOME=/p/realai/zhenghao/cache
store_root=/u/wff7ad/casl_mapping_ablation
checkpoint_dir=checkpoint/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse
mkdir -p "$store_root/runs" "$store_root/checkpoints" "$store_root/logs"
mkdir -p "$checkpoint_dir"

concepts=(smiling young big_nose beards blond_hair)
mappings=(linear mlp)
job_index=0

for mapping in "${mappings[@]}"; do
  for concept in "${concepts[@]}"; do
    exp_id="mapping_${mapping}_${concept}"
    run_target="$store_root/runs/$exp_id"
    run_link="runs/$exp_id"
    checkpoint_target="$store_root/checkpoints/${exp_id}_LC_CelebA_HQ_P2_t999_ninv50_ngen50_0.pth"
    checkpoint_link="$checkpoint_dir/${exp_id}_LC_CelebA_HQ_P2_t999_ninv50_ngen50_0.pth"
    mkdir -p "$run_target"
    if [[ ! -e "$run_link" ]]; then
      ln -s "$run_target" "$run_link"
    fi
    if [[ ! -e "$checkpoint_link" ]]; then
      ln -s "$checkpoint_target" "$checkpoint_link"
    fi
    gpu=$((job_index % 4))
    (
      CUDA_VISIBLE_DEVICES="$gpu" python main.py \
        --train_xblock \
        --delta_h \
        --alignment_mapping "$mapping" \
        --alignment_hidden_dim 512 \
        --alpha 32 \
        --activation ReLU \
        --k 4096 \
        --scale 128 \
        --config celeba_p2.yml \
        --exp "./runs/$exp_id" \
        --edit_attr "$concept" \
        --do_train 1 \
        --do_test 1 \
        --n_train_img 8 \
        --n_test_img 8 \
        --n_iter 1 \
        --bs_train 1 \
        --t_0 999 \
        --n_inv_step 50 \
        --n_train_step 50 \
        --n_test_step 20 \
        --get_h_num 1 \
        --user_defined_t_edit 500 \
        --user_defined_t_addnoise 200 \
        --train_delta_block \
        --save_x0 \
        --save_x_origin \
        --use_x0_tensor \
        --hs_coeff_delta_h 1.0 \
        --lr_training 0.5 \
        --clip_loss_w 1.0 \
        --l1_loss_w 3.0 \
        --retrain 1 \
        --seed 1234 \
        --sh_file_name mapping_ablation/run_mapping_ablation.sh \
        >"$store_root/logs/${exp_id}.log" 2>&1
    ) &
    job_index=$((job_index + 1))
  done
done

wait
