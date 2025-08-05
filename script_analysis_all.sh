#!/bin/bash

# 设置缓存目录和线程限制（关键）
export XDG_CACHE_HOME=/p/realai/zhenghao/cache
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 固定参数设置
sh_file_name="script_analysis.sh"
config="church.yml"
# config="celeba_p2.yml"
activation="ReLU"
scale=128
k=4096
alpha=32
bs_train=1
NUM_GPUS=8
gpu_id=0

# concepts=("church_night" "church_red_brick" "church_golden" "church_factory" "church_wooden_house" "church_gothic" "church_ancient_tower" "church_temple" "church_snowy_church")
concepts=("smiling" "big_nose" "young" "beards" "blond_hair" "eyeglasses" "mouth_slightly_open" "arched_eyebrows" "narrow_eyes")
# concepts=("bald" "eyeglasses" "mouth_slightly_open" "arched_eyebrows" "narrow_eyes")
topks=(1)

mkdir -p logs

for concept in "${concepts[@]}"; do
  for topk in "${topks[@]}"; do

    log_name="logs/${concept}_topk${topk}.log"
    gpu=$((gpu_id % NUM_GPUS))

    # 启动任务（注意 CUDA_VISIBLE_DEVICES=$gpu 是逻辑 GPU 0）
    (
      export CUDA_VISIBLE_DEVICES=$gpu
      echo "Running $concept TopK=$topk on GPU $gpu"
      python main.py \
        --sae_analysis \
        --delta_h \
        --batch_size 128 \
        --top_k_indice 1024 \
        --epoch_num 100 \
        --alpha $alpha \
        --topk_to_analyze $topk \
        --activation "$activation" \
        --k $k \
        --scale $scale \
        --config $config \
        --exp ./runs/$concept \
        --edit_attr $concept \
        --do_train 1 \
        --do_test 1 \
        --n_train_img 100 \
        --n_test_img 32 \
        --n_iter 1 \
        --bs_train $bs_train \
        --t_0 999 \
        --n_inv_step 50 \
        --n_train_step 50 \
        --n_test_step 100 \
        --get_h_num 1 \
        --user_defined_t_edit 500 \
        --user_defined_t_addnoise 200 \
        --train_delta_block \
        --sh_file_name $sh_file_name \
        --save_x0 \
        --use_x0_tensor \
        --hs_coeff_delta_h 1.0 \
        --lr_training 0.5 \
        --clip_loss_w 1.0 \
        --l1_loss_w 3.0 \
        --retrain 1 \
        --custom_train_dataset_dir "hdd1/datasets/celeba_hq/train/male" \
        --custom_test_dataset_dir "hdd1/datasets/celeba_hq/val/male" \
        > "$log_name" 2>&1
    ) &

    echo "Started: $concept TopK=$topk on GPU $gpu --> $log_name"

    # 防止瞬间爆发，可适当调大（单位：秒）
    sleep 2

    ((gpu_id++))

  done
done

wait
echo "All jobs launched across CUDA:0 to CUDA:$((NUM_GPUS-1))"
