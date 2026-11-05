# 数据准备（确保路径对！）
export CELEBA_ROOT="/p/realai/zhenghao/LoRA_SAE/CelebA"
export IMAGES_DIR="$CELEBA_ROOT/images"
export LABELS_DIR="$CELEBA_ROOT/labels/list_attr_celeba.txt"

# 训练命令（带预训练权重、20 epoch、multiplier 0.8）
python celeba_resnet_train.py \
    -p \
    -e 20 \
    --multiplier .8 \
    --batch-size 512 \
    --device "cuda:1" \


# 训练完成提示
echo "训练完成！"
