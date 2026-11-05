#!/bin/bash

# SwiftEdit 批量处理运行脚本
# 简单易用的图片批量编辑

echo "🚀 SwiftEdit Batch Processing"
echo "============================="

# 设置默认参数
INPUT_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0/images"
OUTPUT_DIR="./output_images/beards"
EDIT_PROMPT="a face with beards"

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 输入目录不存在: $INPUT_DIR"
    echo "请创建输入目录并放入要编辑的图片"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行批量处理
echo "📁 输入目录: $INPUT_DIR"
echo "💾 输出目录: $OUTPUT_DIR"
echo "🎯 编辑提示: $EDIT_PROMPT"
echo ""

python batch_edit.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --edit_prompt "$EDIT_PROMPT"

echo ""
echo "✅ 处理完成！结果保存在: $OUTPUT_DIR"