#!/bin/bash
# SwiftEdit Simple Batch Processing Script

echo "🚀 SwiftEdit 批处理 (简化版)"
echo "============================="

# 激活conda环境
source ~/.bashrc
conda activate SwiftEdit

# 运行批处理脚本
python batch_edit_simple.py \
    --input_dir "/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0/images" \
    --output_dir "./output_images_simple" \
    --edit_prompt "a smiling face"

echo "✅ 批处理完成！"