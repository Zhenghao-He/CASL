#!/usr/bin/env bash
set -euo pipefail

export XDG_CACHE_HOME=/p/realai/zhenghao/cache
store_root=/u/wff7ad/casl_mapping_ablation
sae_name=delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse
concepts=(smiling young big_nose beards blond_hair)
mappings=(linear mlp)
job_index=0

for mapping in "${mappings[@]}"; do
  for concept in "${concepts[@]}"; do
    exp_id="mapping_${mapping}_${concept}"
    output_root="runs/${exp_id}_LC_CelebA_HQ_P2_t999_ninv50_ngen50/${sae_name}/xblock_test_images/20"
    if [[ -d "$output_root/iter0_ngen50" && ! -e "$output_root/iter0_ngen50_stochastic" ]]; then
      mv "$output_root/iter0_ngen50" "$output_root/iter0_ngen50_stochastic"
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
        --user_defined_t_addnoise 0 \
        --train_delta_block \
        --save_x0 \
        --save_x_origin \
        --use_x0_tensor \
        --hs_coeff_delta_h 1.0 \
        --lr_training 0.5 \
        --clip_loss_w 1.0 \
        --l1_loss_w 3.0 \
        --retrain 0 \
        --seed 1234 \
        --sh_file_name mapping_ablation/run_deterministic_test.sh \
        >"$store_root/logs/${exp_id}_deterministic_test.log" 2>&1
    ) &
    job_index=$((job_index + 1))
    if ((job_index % 4 == 0)); then
      wait
    fi
  done
done

wait
