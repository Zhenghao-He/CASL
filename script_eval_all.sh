
# #!/bin/bash

# MODEL_NAME="resnet"  # vit or resnet18
# TopK=1
# BATCH_SIZE=32
# DEVICE=cuda:0
# SCALE=0
# # concepts=("No_Beard")
# concepts=("Smiling" "Big_Nose" "Young" "Blond_Hair" "No_Beard")
# BEFORE_DIR=./runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0
# MODEL_PATH=./celeba_classifier/${MODEL_NAME}_model/${MODEL_NAME}_multiplicative_pretrain_
# LABEL_FILE=../CelebA-HQ/list_attr_celeba_shift.txt

# for concept in "${concepts[@]}"; do
  
#   if [[ "$concept" == "No_Beard" ]]; then
#     concept_lc="beard"
#   else
#     concept_lc="${concept,,}"
#   fi

#   AFTER_DIR=./runs/${concept_lc}_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/test_xblock_editted_delta_images_topk${TopK}/100/test_0_scale${SCALE}/delta_h
#   SAVE_PATH=./epr/${MODEL_NAME}/epr_results_${concept_lc}_sae_top${TopK}.txt


#   echo "Running: $concept ($method)"
#   python evaluate.py \
#     --before_dir "$BEFORE_DIR" \
#     --after_dir "$AFTER_DIR" \
#     --model_path "$MODEL_PATH" \
#     --label_file "$LABEL_FILE" \
#     --target_concept "$concept" \
#     --batch_size "$BATCH_SIZE" \
#     --device "$DEVICE" \
#     --save_path "$SAVE_PATH"

# done
#!/bin/bash
#!/bin/bash
#!/bin/bash

MODEL_NAME="resnet"  # resnet18 or mobilenetv2
BATCH_SIZE=32
MODEL_PATH=./celeba_classifier/${MODEL_NAME}_model/${MODEL_NAME}18_multiplicative_pretrain_
LABEL_FILE=../CelebA-HQ/list_attr_celeba_shift.txt
BEFORE_DIR=./runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0

# concepts=("Eyeglasses" "Mouth_Slightly_Open" "Arched_Eyebrows" "Narrow_Eyes")
# concepts=("Eyeglasses")
concepts=("Smiling" "Big_Nose" "Young" "Blond_Hair" "No_Beard" "Eyeglasses" "Mouth_Slightly_Open" "Arched_Eyebrows" "Narrow_Eyes")
# concepts=("Bald" "Eyeglasses" "Mouth_Slightly_Open" "Arched_Eyebrows" "Narrow_Eyes")
topks=(1)
scales=(0 32 64 96 128 160 192)

NUM_GPUS=8
MAX_TASKS_PER_GPU=2
MAX_JOBS=$((NUM_GPUS * MAX_TASKS_PER_GPU))

mkdir -p ./epr/${MODEL_NAME}
mkdir -p logs

job_count=0

for concept in "${concepts[@]}"; do
  concept_lc="${concept,,}"
  if [[ "$concept" == "No_Beard" ]]; then
    concept_lc="beards"
  fi

  for topk in "${topks[@]}"; do
    for scale in "${scales[@]}"; do

      while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        sleep 5
      done

      gpu_id=$((job_count % NUM_GPUS))
      ((job_count++))

      AFTER_DIR=./runs/${concept_lc}_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/test_xblock_editted_delta_images_topk${topk}/100/test_0_scale${scale}/delta_h
      SAVE_PATH=./epr/${MODEL_NAME}/epr_results_${concept_lc}_topk${topk}_scale${scale}.txt
      if [[ -f "$SAVE_PATH" ]]; then
        echo "Skip: $SAVE_PATH already exists."
        continue
      fi

      LOG_PATH=logs/${concept_lc}_topk${topk}_scale${scale}.log

      echo "Launching: $concept | TopK=$topk | Scale=$scale on GPU:$gpu_id --> $LOG_PATH"

      CUDA_VISIBLE_DEVICES=$gpu_id python evaluate.py \
        --before_dir "$BEFORE_DIR" \
        --after_dir "$AFTER_DIR" \
        --model_path "$MODEL_PATH" \
        --label_file "$LABEL_FILE" \
        --target_concept "$concept" \
        --batch_size "$BATCH_SIZE" \
        --device "cuda:0" \
        --save_path "$SAVE_PATH" > "$LOG_PATH" 2>&1 &

    done
  done
done

wait
echo "✅ All jobs finished."
