# #!/bin/bash

# MODEL_NAME="vit" # ViT, ResNet18
# TopK=1
# BEFORE_DIR=./runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0
# concepts=("Smiling" "Big_Nose" "Young" "Blond_Hair" "Mustache")


# MODEL_PATH=./celeba_classifier/${MODEL_NAME}_model/${MODEL_NAME}_multiplicative_pretrain_
# LABEL_FILE=../CelebA-HQ/list_attr_celeba_shift.txt
# # TARGET_CONCEPT=Young # Big_Nose, Young, Smiling, etc.
# BATCH_SIZE=32
# DEVICE=cuda:0
# for concept in "${concepts[@]}"; do
#   AFTER_DIR=./runs/${concept,,}_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/test_xblock_editted_delta_images_topk${TopK}/100/test_0_scale96/delta_h
#   # AFTER_DIR=../Asyrp_official/runs/${concept,,}_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h
#   # AFTER_DIR=./runs/${concept,,}_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin
#   # AFTER_DIR=./boundary_good/${concept}_500Range_7 #boundary
#   # SAVE_PATH=./nls_results_beards_asyrp.txts
#   SAVE_PATH=./epr/${MODEL_NAME}/nls_results_${concept,,}_sae_top${TopK}.txt
#   # SAVE_PATH=./nls_results_beards_recons.txt
#   # SAVE_PATH=./nls_results_beards_boundary.txt
#   # SAVE_PATH=./nls_results_official0.txt

#   python evaluate.py \
#     --before_dir "$BEFORE_DIR" \
#     --after_dir "$AFTER_DIR" \
#     --model_path "$MODEL_PATH" \
#     --label_file "$LABEL_FILE" \
#     --target_concept "$concept" \
#     --batch_size "$BATCH_SIZE" \
#     --device "$DEVICE" \
#     --save_path "$SAVE_PATH"

#!/bin/bash

MODEL_NAME="vit"  # vit or resnet18
TopK=1
BATCH_SIZE=32
DEVICE=cuda:3

concepts=("Smiling" "Blond_Hair")
# concepts=("Smiling" "Big_Nose" "Young" "Blond_Hair" "No_Beard")
BEFORE_DIR=./runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0
MODEL_PATH=./celeba_classifier/${MODEL_NAME}_model/${MODEL_NAME}_multiplicative_pretrain_
LABEL_FILE=../CelebA-HQ/list_attr_celeba_shift.txt

for concept in "${concepts[@]}"; do
  for method in "sae" ; do
  # for method in "sae" "asyrp" "recons" "boundary"; do

    # lowercase for folder path
    concept_lc="${concept,,}"

    # 设置 AFTER_DIR 和 SAVE_PATH
    case $method in
      "sae")
        AFTER_DIR=./runs/${concept_lc}_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/test_xblock_editted_delta_images_topk${TopK}/100/test_0_scale176/delta_h
        SAVE_PATH=./epr/${MODEL_NAME}/epr_results_${concept_lc}_sae_top${TopK}.txt
        ;;
      # "recons")
      #   AFTER_DIR=./runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin
      #   SAVE_PATH=./epr/${MODEL_NAME}/epr_results_${concept_lc}_recons.txt
      #   ;;
      # "asyrp")
      #   AFTER_DIR=../Asyrp_official/runs/${concept_lc}_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h
      #   SAVE_PATH=./epr/${MODEL_NAME}/epr_results_${concept_lc}_asyrp.txt
      #   ;;
      # "boundary")
      #   AFTER_DIR=./boundary_good/${concept}_500Range_7
      #   SAVE_PATH=./epr/${MODEL_NAME}/epr_results_${concept_lc}_boundary.txt
      #   ;;

#       "sae")
#         AFTER_DIR=./runs/beards_LC_CelebA_HQ_P2_t999_ninv50_ngen50/\
# delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/test_xblock_editted_delta_images_topk${TopK}/100/test_0_scale96/delta_h
#         SAVE_PATH=./epr/${MODEL_NAME}/epr_results_${concept_lc}_sae_top${TopK}.txt
#         ;;
#       "asyrp")
#         AFTER_DIR=../Asyrp_official/runs/beards_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h
#         SAVE_PATH=./epr/${MODEL_NAME}/epr_results_${concept_lc}_asyrp.txt
#         ;;
#       "recons")
#         AFTER_DIR=./runs/beards_LC_CelebA_HQ_P2_t999_ninv50_ngen50/\
# delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin
#         SAVE_PATH=./epr/${MODEL_NAME}/epr_results_${concept_lc}_recons.txt
#         ;;
      # "boundary")
      #   AFTER_DIR=./boundary_good/${concept}_500Range_7
      #   SAVE_PATH=./epr/${MODEL_NAME}/epr_results_${concept_lc}_boundary.txt
      #   ;;
    esac

    echo "Running: $concept ($method)"
    python evaluate_${MODEL_NAME,,}.py \
      --before_dir "$BEFORE_DIR" \
      --after_dir "$AFTER_DIR" \
      --model_path "$MODEL_PATH" \
      --label_file "$LABEL_FILE" \
      --target_concept "$concept" \
      --batch_size "$BATCH_SIZE" \
      --device "$DEVICE" \
      --save_path "$SAVE_PATH"

  done
done
