# #!/bin/bash

# sh_file_name="script_sample_z.sh"
# gpu="2"
# config="celeba_p2.yml"
# export XDG_CACHE_HOME=/p/realai/zhenghao/cache

# concepts=("Big_Nose" "Young" "No_Beard" "Blond_Hair")
# topks=(1 2 4 8)

# for concept in "${concepts[@]}"
# do
# #   guid=$(echo "$concept" | tr '[:upper:]' '[:lower:]') # 转小写
#   for topk in "${topks[@]}"
#   do
#     echo "Running for concept: $concept, topk: $topk"
#     if [ "$concept" == "No_Beard" ]; then
#       guid="beards"
#     else
#       guid=${concept,,} # 转小写
#     fi
#     CUDA_VISIBLE_DEVICES=$gpu python main.py --sample_z \
#       --config $config \
#       --exp ./runs/$guid \
#       --edit_attr $guid \
#       --top_k_indice 1024 \
#       --topk_to_analyze $topk \
#       --sample_attr "$concept" \
#       --sample_test \
#       --do_train 1 \
#       --do_test 1 \
#       --n_train_img 1000 \
#       --alpha 32 \
#       --activation "ReLU" \
#       --scale 128 \
#       --n_test_img 32 \
#       --n_iter 5 \
#       --bs_train 1 \
#       --t_0 999 \
#       --n_inv_step 50 \
#       --n_train_step 50 \
#       --n_test_step 100 \
#       --get_h_num 1 \
#       --user_defined_t_edit 500 \
#       --user_defined_t_addnoise 200 \
#       --train_delta_block \
#       --sh_file_name $sh_file_name \
#       --save_x0 \
#       --use_x0_tensor \
#       --hs_coeff_delta_h 1.0 \
#       --lr_training 0.5 \
#       --clip_loss_w 1.0 \
#       --l1_loss_w 3.0 \
#       --retrain 1 \
#       --custom_train_dataset_dir "hdd1/datasets/celeba_hq/train/male" \
#       --custom_test_dataset_dir "hdd1/datasets/celeba_hq/val/male" \
#       --sh_file_name $sh_file_name

#     # 可加 sleep 等待资源释放，防止爆显存
#     sleep 5
#   done
# done
#!/bin/bash

sh_file_name="script_sample_z.sh"
gpus=(0 1 2 3)   # 你的GPU编号，根据实际情况修改
config="celeba_p2.yml"
export XDG_CACHE_HOME=/p/realai/zhenghao/cache

concepts=("Big_Nose" "Young" "No_Beard" "Blond_Hair")
topks=(1 2 4 8)

for idx in "${!concepts[@]}"
do
  concept="${concepts[$idx]}"
  gpu="${gpus[$idx]}"
  (
    for topk in "${topks[@]}"
    do
      echo "Running for concept: $concept, topk: $topk on GPU $gpu"
      if [ "$concept" == "No_Beard" ]; then
        guid="beards"
      else
        guid=${concept,,}
      fi
      CUDA_VISIBLE_DEVICES=$gpu python main.py --sample_z \
        --config $config \
        --exp ./runs/$guid \
        --edit_attr $guid \
        --top_k_indice 1024 \
        --topk_to_analyze $topk \
        --sample_attr "$concept" \
        --sample_test \
        --do_train 1 \
        --do_test 1 \
        --n_train_img 1000 \
        --alpha 32 \
        --activation "ReLU" \
        --scale 128 \
        --n_test_img 32 \
        --n_iter 5 \
        --bs_train 1 \
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
        --sh_file_name $sh_file_name

      sleep 5
    done
  ) &
done

wait
