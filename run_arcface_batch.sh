#!/bin/bash

# =============================================================================
# ArcFace Face Identity Preservation Evaluation - Batch Processing Script
# =============================================================================
# 
# EXPERIMENTAL SETUP FOR FACE EDITING EVALUATION
# =============================================
# 
# Purpose: Batch evaluation of face identity preservation across multiple 
#          image editing methods using ArcFace face recognition similarity
# 
# EVALUATION CONFIGURATION:
# ------------------------
# Model: InsightFace buffalo_l (ResNet-50 based)
# Detection Threshold: 0.3 (optimized for edited faces)
# Detection Size: 640×640 pixels  
# Similarity Metric: Cosine similarity on face embeddings
# Range: [0,1] where 1.0 = identical identity, 0.0 = different identity
# 
# DATASET SPECIFICATIONS:
# ----------------------
# Image Resolution: 256×256 pixels
# Format: RGB JPEG images  
# Pairing Logic: 5-digit filename matching (e.g., 29501.jpg ↔ 29501_edited.jpg)
# Original Images: Reconstruction results (x0) from diffusion inversion
# 
# EVALUATED METHODS:
# -----------------
# 1. Boundary: Latent space boundary manipulation
# 2. Asyrp: Asymmetric reverse process editing  
# 3. Ours: SAE-based latent editing (our method)
# 4. Reconstruction: Diffusion reconstruction baseline
# 5. MasaCtrl: Mutual self-attention control editing
# 
# EVALUATION CONCEPTS:
# -------------------
# - Young: Age manipulation towards younger appearance
# - Smiling: Expression editing to add smiling
# - Big_Nose: Nose size manipulation  
# - Mustache/Beards: Facial hair addition
# - Blond_Hair: Hair color change to blonde
# 
# EXPECTED RESULTS:
# ----------------
# Success Rate: ~81% (after threshold optimization)
# Typical Similarity Ranges:
# - Excellent preservation (>0.8): Minimal identity change
# - Good preservation (0.5-0.8): Noticeable but acceptable changes  
# - Poor preservation (<0.5): Significant identity alteration
# 
# OUTPUT STRUCTURE:
# ----------------
# Results saved to: arcface_results/{method}/arcface_{concept}.csv
# Statistics: {method}/arcface_{concept}_statistics.csv
# Error logs: {method}/arcface_{concept}_errors.txt
# 
# HARDWARE REQUIREMENTS:
# ---------------------
# Environment: sliders_clone conda environment
# Dependencies: insightface, opencv-python, torch, pandas
# Execution: CPU-based for reproducibility
# Runtime: ~1-2 minutes per method-concept combination
# 
# =============================================================================

# ArcFace Batch Processing Script
# 批量计算多个方法的ArcFace相似度分数 - 所有方法使用相同的原始图片

cd /p/realai/zhenghao/LoRA_SAE/Asyrp_SAE

echo "Starting ArcFace batch processing..."
echo "========================================="

# 固定的原始图片目录
ORIGINAL_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0/images"

# 定义不同方法的编辑后图片目录和提示词（参考LPIPS脚本的结构）
declare -A ARCFACE_CONFIGS=(
    # Boundary 方法
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/boundary_good/Smiling_500Range_7/images"]="a smiling face|boundary"
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/boundary_good/Big_Nose_500Range_7/images"]="face with a big nose|boundary"
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/boundary_good/Young_500Range_7/images"]="a young face|boundary"
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/boundary_good/Mustache_500Range_7/images"]="a face with beards|boundary"
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/boundary_good/Blond_Hair_500Range_7/images"]="people with blond hair|boundary"
    
    # # Asyrp 方法
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_official/runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images"]="a young face|asyrp"
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_official/runs/smiling_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images"]="a smiling face|asyrp"
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_official/runs/big_nose_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images"]="face with a big nose|asyrp"
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_official/runs/blond_hair_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images"]="people with blond hair|asyrp"
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_official/runs/beards_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images"]="a face with beards|asyrp"
    
    # # Ours 方法
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/delta_h/images"]="a young face|ours"
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/smiling_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/delta_h/images"]="a smiling face|ours"
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/big_nose_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/delta_h/images"]="face with a big nose|ours"
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/blond_hair_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/delta_h/images"]="people with blond hair|ours"
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/beards_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/delta_h/images"]="a face with beards|ours"
    
    # Reconstruction 方法
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin/images"]="a young face|reconstruction"
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/smiling_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin/images"]="a smiling face|reconstruction"
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/big_nose_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin/images"]="face with a big nose|reconstruction"
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/blond_hair_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin/images"]="people with blond hair|reconstruction"
    # ["/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/beards_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin/images"]="a face with beards|reconstruction"
    
    # MasaCtrl 方法
    # ["/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/young1"]="a young face|masactrl"
    # ["/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/smiling1"]="a smiling face|masactrl"
    # ["/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/big_nose1"]="face with a big nose|masactrl"
    # ["/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/beards1"]="a face with beards|masactrl"
    # ["/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/blonde_hair1"]="people with blond hair|masactrl"

    # Concept Slider
    # ["/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/age"]="a young face|concept_slider"
    # ["/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/smiling"]="a smiling face|concept_slider"
    # ["/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/bignose"]="face with a big nose|concept_slider"
    # ["/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/beards"]="a face with beards|concept_slider"
    # ["/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/blonde"]="people with blond hair|concept_slider"

    # ["/p/realai/zhenghao/LoRA_SAE/SwiftEdit/output_images/smiling"]="a smiling face|swiftedit"
    # ["/p/realai/zhenghao/LoRA_SAE/SwiftEdit/output_images/big_nose"]="face with a big nose|swiftedit"
    # ["/p/realai/zhenghao/LoRA_SAE/SwiftEdit/output_images/young"]="a young face|swiftedit"
    ["/p/realai/zhenghao/LoRA_SAE/SwiftEdit/output_images/beards"]="a face with beards|swiftedit"
    # ["/p/realai/zhenghao/LoRA_SAE/SwiftEdit/output_images/blond_hair"]="people with blond hair|swiftedit"
)

# 创建结果目录
mkdir -p /p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/arcface_results

echo "Original images directory: $ORIGINAL_DIR"

# 处理每个配置
for edited_dir in "${!ARCFACE_CONFIGS[@]}"; do
    # 分解值：prompt|method
    value="${ARCFACE_CONFIGS[$edited_dir]}"
    prompt="${value%|*}"
    method="${value#*|}"
    
    # 从prompt生成概念名（与CLIP脚本保持一致）
    case "$prompt" in
        "a young face") concept="Young" ;;
        "a smiling face") concept="Smiling" ;;
        "face with a big nose") concept="Big_Nose" ;;
        "a face with beards") concept="Mustache" ;;
        "people with blond hair") concept="Blond_Hair" ;;
        *) concept="Unknown" ;;
    esac
    
    # 创建方法特定的子目录
    mkdir -p "/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/arcface_results/$method"
    output_file="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/arcface_results/$method/arcface_${concept}.csv"

    echo ""
    echo "Processing: $method - $concept"
    echo "Original: $ORIGINAL_DIR"
    echo "Edited: $edited_dir"
    echo "Output: $output_file"
    echo "----------------------------------------"
    
    # 检查原始目录是否存在
    if [ ! -d "$ORIGINAL_DIR" ]; then
        echo "✗ Original directory not found: $ORIGINAL_DIR"
        continue
    fi
    
    # 检查编辑后目录是否存在
    if [ ! -d "$edited_dir" ]; then
        echo "✗ Edited directory not found: $edited_dir"
        continue
    fi
    
    # 运行 ArcFace 计算
    python /p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/calculate_arcface.py \
        --original_dir "$ORIGINAL_DIR" \
        --edited_dir "$edited_dir" \
        --output "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $method - $concept"
    else
        echo "✗ Error processing $method - $concept"
    fi
done

echo ""
echo "========================================="
echo "ArcFace batch processing completed!"
echo "Results saved in: /p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/arcface_results/"
echo ""
echo "Generated files:"
find /p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/arcface_results/ -name "*.csv" -type f | sort
echo ""
echo "Directory structure:"
tree /p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/arcface_results/ 2>/dev/null || find /p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/arcface_results/ -type d | sort