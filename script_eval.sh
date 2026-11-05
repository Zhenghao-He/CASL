#!/bin/bash

# EPR Evaluation Batch Processing Script

cd /p/realai/zhenghao/LoRA_SAE/Asyrp_SAE

echo "Starting EPR Evaluation batch processing..."
echo "========================================="

# Configuration
MODEL_NAME="vit"  # vit or resnet18
TopK=1
BATCH_SIZE=32
DEVICE=cuda:3
BEFORE_DIR=./runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0/images
MODEL_PATH=./celeba_classifier/${MODEL_NAME}_model/${MODEL_NAME}_multiplicative_pretrain_
LABEL_FILE=../CelebA-HQ/list_attr_celeba_shift.txt

# Define evaluation configurations by method and concept
declare -A EVAL_CONFIGS=(
    # SAE method
    ["ours/Smiling"]="./runs/smiling_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/test_xblock_editted_delta_images_topk${TopK}/100/test_0_scale64/delta_h/images"
    ["ours/Big_Nose"]="./runs/big_nose_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/test_xblock_editted_delta_images_topk${TopK}/100/test_0_scale64/delta_h/images"
    ["ours/Young"]="./runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/test_xblock_editted_delta_images_topk${TopK}/100/test_0_scale64/delta_h/images"
    ["ours/Blond_Hair"]="./runs/blond_hair_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/test_xblock_editted_delta_images_topk${TopK}/100/test_0_scale64/delta_h/images"
    ["ours/No_Beard"]="./runs/beards_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/test_xblock_editted_delta_images_topk${TopK}/100/test_0_scale64/delta_h/images"
   
    # ASYRP method
    ["asyrp/Smiling"]="../Asyrp_official/runs/smiling_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images"
    ["asyrp/Big_Nose"]="../Asyrp_official/runs/big_nose_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images"
    ["asyrp/Young"]="../Asyrp_official/runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images"
    ["asyrp/Blond_Hair"]="../Asyrp_official/runs/blond_hair_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images"
    ["asyrp/No_Beard"]="../Asyrp_official/runs/beards_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images"
     
    # Boundary method
    ["boundary/Smiling"]="./boundary_good/Smiling_500Range_7/images"
    ["boundary/Big_Nose"]="./boundary_good/Big_Nose_500Range_7/images"
    ["boundary/Young"]="./boundary_good/Young_500Range_7/images"
    ["boundary/Blond_Hair"]="./boundary_good/Blond_Hair_500Range_7/images"
    ["boundary/No_Beard"]="./boundary_good/Mustache_500Range_7/images"
    
    # Reconstruction baseline
    ["recons/Smiling"]="./runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin/images"
    ["recons/Big_Nose"]="./runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin/images"
    ["recons/Young"]="./runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin/images"
    ["recons/Blond_Hair"]="./runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin/images"
    ["recons/No_Beard"]="./runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin/images"
   
    # MasaCtrl generated images
    ["masactrl/Smiling"]="/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/smiling1"
    ["masactrl/Big_Nose"]="/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/big_nose1"
    ["masactrl/Young"]="/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/young1"
    ["masactrl/No_Beard"]="/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/beards1"
    ["masactrl/Blond_Hair"]="/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/blonde_hair1"

#     # Sliders edited images
    ["sliders/Smiling"]="/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/smiling"
    ["sliders/Big_Nose"]="/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/bignose"
    ["sliders/Young"]="/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/age"
    ["sliders/No_Beard"]="/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/beards"
    ["sliders/Blond_Hair"]="/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/blonde"

#     # SwiftEdit output images
    ["swiftedit/Smiling"]="/p/realai/zhenghao/LoRA_SAE/SwiftEdit/output_images/smiling"
    ["swiftedit/Big_Nose"]="/p/realai/zhenghao/LoRA_SAE/SwiftEdit/output_images/big_nose"
    ["swiftedit/Young"]="/p/realai/zhenghao/LoRA_SAE/SwiftEdit/output_images/young"
    ["swiftedit/No_Beard"]="/p/realai/zhenghao/LoRA_SAE/SwiftEdit/output_images/beards"
    ["swiftedit/Blond_Hair"]="/p/realai/zhenghao/LoRA_SAE/SwiftEdit/output_images/blond_hair"
)

mkdir -p epr/${MODEL_NAME}

for config in "${!EVAL_CONFIGS[@]}"; do
    IFS='/' read -r method concept <<< "$config"
    after_dir="${EVAL_CONFIGS[$config]}"
    
    # Convert concept to lowercase for save path
    if [[ "$concept" == "No_Beard" ]]; then
        concept_lc="beards"
    else
        concept_lc="${concept,,}"
    fi
    
    # Create method subdirectory
    mkdir -p epr/${MODEL_NAME}/${method}
    
    # Generate save path with method subdirectory
    if [[ "$method" == "ours" ]]; then
        save_path="./epr/${MODEL_NAME}/${method}/epr_results_${concept_lc}_sae_top${TopK}.txt"
    else
        save_path="./epr/${MODEL_NAME}/${method}/epr_results_${concept_lc}_${method}.txt"
    fi
    
    # Check if input directory exists
    if [[ ! -d "$after_dir" ]]; then
        echo "⚠️  Warning: Directory not found: $after_dir"
        echo "   Skipping: $concept ($method)"
        continue
    fi
    
    echo "📊 Processing: $concept ($method)"
    echo "   Input: $after_dir"
    echo "   Output: $save_path"
    
    python evaluate.py \
        --before_dir "$BEFORE_DIR" \
        --after_dir "$after_dir" \
        --model_path "$MODEL_PATH" \
        --label_file "$LABEL_FILE" \
        --target_concept "$concept" \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --save_path "$save_path"
    
    if [[ $? -eq 0 ]]; then
        echo "   ✅ Completed successfully"
    else
        echo "   ❌ Failed to process"
    fi
    echo ""
done

echo ""
echo "========================================="
echo "EPR Evaluation batch processing completed!"
echo "Results saved in: epr/${MODEL_NAME}/"
echo ""
echo "Generated files by method:"
for method_dir in epr/${MODEL_NAME}/*/; do
    if [[ -d "$method_dir" ]]; then
        method_name=$(basename "$method_dir")
        echo "  📁 $method_name:"
        find "$method_dir" -name "*.txt" -type f | sort | while read file; do
            echo "    - $(basename "$file")"
        done
    fi
done
