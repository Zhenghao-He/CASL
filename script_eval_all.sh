
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

# ===================== 用户可配置参数 =====================
MODEL_NAME="resnet"  # 可选: resnet / vit
BATCH_SIZE=32
MODEL_PATH=./celeba_classifier/${MODEL_NAME}_model/${MODEL_NAME}_multiplicative_pretrain_
LABEL_FILE=../CelebA-HQ/list_attr_celeba_shift.txt
BEFORE_DIR=./runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0

concepts=("Smiling" "Big_Nose" "Young" "Blond_Hair" "No_Beard")
topks=(1 2 4 8 16 32)
scales=(0 32 64 96 128 160 192)  # 可选: 0, 32, 64, 96, 128, 160, 192, 224, 256

NUM_GPUS=8              # 你有几张卡（比如 8）
MAX_TASKS_PER_GPU=2     # 每张卡允许的最大并发任务数（建议 1~2）

mkdir -p ./epr/${MODEL_NAME}
mkdir -p logs
# ==========================================================

declare -A gpu_task_count
for ((i = 0; i < NUM_GPUS; i++)); do
  gpu_task_count[$i]=0
done

wait_for_free_gpu() {
  while true; do
    for ((i = 0; i < NUM_GPUS; i++)); do
      if [[ ${gpu_task_count[$i]} -lt $MAX_TASKS_PER_GPU ]]; then
        echo $i
        return
      fi
    done
    sleep 5
  done
}

for concept in "${concepts[@]}"; do
  if [[ "$concept" == "No_Beard" ]]; then
    concept_lc="beard"
  else
    concept_lc="${concept,,}"
  fi

  for topk in "${topks[@]}"; do
    for scale in "${scales[@]}"; do

      gpu_id=$(wait_for_free_gpu)
      ((gpu_task_count[$gpu_id]++))

      AFTER_DIR=./runs/${concept_lc}_LC_CelebA_HQ_P2_t999_ninv50_ngen50/\
delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/test_xblock_editted_delta_images_topk${topk}/100/test_0_scale${scale}/delta_h

      SAVE_PATH=./epr/${MODEL_NAME}/sae/epr_results_${concept_lc}_topk${topk}_scale${scale}.txt
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

      pid=$!

      # 用 subshell 来异步监听进程结束并释放 gpu_slot
      {
        wait $pid
        ((gpu_task_count[$gpu_id]--))
      } &
    done
  done
done

wait
echo "All jobs finished."
