#!/bin/bash

# Example settings, please adjust these variables before running
BEFORE_DIR=./runs/smiling_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0
# AFTER_DIR=../Asyrp_official/runs/smiling_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter0/delta_h
# AFTER_DIR=./runs/smiling_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin
AFTER_DIR=./runs/smiling_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/test_xblock_editted_delta_images_topk1/100/test_0_scale96/delta_h
MODEL_PATH=./celeba_classifier/resnet_model/resnet18_multiplicative_pretrain_
LABEL_FILE=../CelebA-HQ/list_attr_celeba_shift.txt
TARGET_CONCEPT=Smiling
BATCH_SIZE=32
DEVICE=cuda:0
SAVE_PATH=./nls_results_smiling_sae_top1.txt
# SAVE_PATH=./nls_results_official0.txt

python evaluate.py \
  --before_dir "$BEFORE_DIR" \
  --after_dir "$AFTER_DIR" \
  --model_path "$MODEL_PATH" \
  --label_file "$LABEL_FILE" \
  --target_concept "$TARGET_CONCEPT" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --save_path "$SAVE_PATH"
