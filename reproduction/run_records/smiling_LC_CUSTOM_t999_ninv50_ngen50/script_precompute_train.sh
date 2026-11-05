#!/bin/bash

sh_file_name="script_precompute.sh"
gpu="0"

config="custom.yml" # if you use other dataset, config/path_config.py should be matched
guid="smiling" # guid should be in utils/text_dic.py

export XDG_CACHE_HOME=/p/realai/zhenghao/cache
CUDA_VISIBLE_DEVICES=$gpu python main.py --run_train                        \
                        --config $config                                    \
                        --exp ./runs/$guid                                  \
                        --edit_attr $guid                                   \
                        --do_train 1                                        \
                        --do_test 1                                         \
                        --n_train_img 100                                   \
                        --n_test_img 32                                     \
                        --bs_train 1                                        \
                        --get_h_num 1                                       \
                        --train_delta_block                                 \
                        --t_0 999                                           \
                        --n_inv_step 50                                     \
                        --n_train_step 50                                   \
                        --n_test_step 50                                    \
                        --just_precompute                                   \
                        --custom_train_dataset_dir "hdd1/datasets/celeba_hq/train/male"       \
                        --custom_test_dataset_dir "hdd1/datasets/celeba_hq/val/male"         \
                        --sh_file_name "script_precompute.sh"

