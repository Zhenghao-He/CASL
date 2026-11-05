#!/bin/bash

# 简单的CLIP执行命令示例
# 你可以直接复制这些命令到终端运行

echo "=== CLIP Score Calculation Commands ==="
echo ""

echo "1. 进入工作目录:"
echo "cd /p/realai/zhenghao/LoRA_SAE/Asyrp_SAE"
echo ""

echo "2. 单个目录执行命令示例:"
echo ""

echo "# Blond Hair"
echo 'python clip_batch.py --image_dir "boundary_good/Blond_Hair_500Range_7/images" --prompt "people with blond hair" --output "clip_blond_hair.csv"'
echo ""

echo "# Smiling"
echo 'python clip_batch.py --image_dir "boundary_good/Smiling_500Range_7/images" --prompt "a smiling face" --output "clip_smiling.csv"'
echo ""

echo "# Young"
echo 'python clip_batch.py --image_dir "boundary_good/Young_500Range_7/images" --prompt "a young face" --output "clip_young.csv"'
echo ""

echo "# Big Nose"
echo 'python clip_batch.py --image_dir "boundary_good/Big_Nose_500Range_7/images" --prompt "face with a big nose" --output "clip_big_nose.csv"'
echo ""

echo "# Mustache"
echo 'python clip_batch.py --image_dir "boundary_good/Mustache_500Range_7/images" --prompt "a face with beards" --output "clip_mustache.csv"'
echo ""

echo "3. 批量执行:"
echo "./run_clip_batch.sh"
echo ""

echo "4. 单个执行:"
echo "./run_single_clip.sh"