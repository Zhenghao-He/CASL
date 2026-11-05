#!/bin/bash

# 单个CLIP分数计算执行脚本
# Usage: ./run_single_clip.sh

cd /p/realai/zhenghao/LoRA_SAE/Asyrp_SAE

# 设置参数
IMAGE_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/boundary_good/Blond_Hair_500Range_7/images"
PROMPT="people with blond hair"
OUTPUT_FILE="clip_scores_blond_hair.csv"

echo "Running CLIP score calculation..."
echo "Image Directory: $IMAGE_DIR"
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT_FILE"
echo "=================================="

python clip_batch.py \
    --image_dir "$IMAGE_DIR" \
    --prompt "$PROMPT" \
    --output "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ CLIP score calculation completed!"
    echo "Results saved to: $OUTPUT_FILE"
    echo "Statistics saved to: ${OUTPUT_FILE%.*}_statistics.csv"
    
    # 显示结果文件
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo "First 5 results:"
        head -6 "$OUTPUT_FILE"
    fi
else
    echo "✗ Error occurred during calculation"
fi