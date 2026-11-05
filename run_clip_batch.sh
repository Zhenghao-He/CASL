#!/bin/bash

# CLIP Score Batch Processing Script

cd /p/realai/zhenghao/LoRA_SAE/Asyrp_SAE

chmod +x clip_batch.py

echo "Starting CLIP Score batch processing..."
echo "========================================="

declare -A CONFIGS=(
    #boundary
    # ["boundary_good/Smiling_500Range_7/images"]="a smiling face"
    # ["boundary_good/Big_Nose_500Range_7/images"]="face with a big nose"
    # ["boundary_good/Young_500Range_7/images"]="a young face"
    # ["boundary_good/Mustache_500Range_7/images"]="a face with beards"
    # ["boundary_good/Blond_Hair_500Range_7/images"]="people with blond hair"
    # # Asyrp
    # ["../Asyrp_official/runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images"]="a young face"
    # ["../Asyrp_official/runs/smiling_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images"]="a smiling face"
    # ["../Asyrp_official/runs/big_nose_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images"]="face with a big nose"
    # ["../Asyrp_official/runs/blond_hair_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images"]="people with blond hair"
    # ["../Asyrp_official/runs/beards_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images"]="a face with beards"
    # #ours
    # ["runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/delta_h/images"]="a young face"
    # ["runs/smiling_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/delta_h/images"]="a smiling face"
    # ["runs/big_nose_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/delta_h/images"]="face with a big nose"
    # ["runs/blond_hair_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/delta_h/images"]="people with blond hair"
    # ["runs/beards_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/delta_h/images"]="a face with beards"
    #reconstruction
    # ["runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin/images"]="a young face"
    # ["runs/smiling_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin/images"]="a smiling face"
    # ["runs/big_nose_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin/images"]="face with a big nose"
    # ["runs/blond_hair_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin/images"]="people with blond hair"
    # ["runs/beards_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/origin/images"]="a face with beards"
    # #Masactrl
    # ["/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/young1"]="a young face"
    # ["/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/smiling1"]="a smiling face"
    # ["/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/big_nose1"]="face with a big nose"
    # ["/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/beards1"]="a face with beards"
    # ["/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/blonde_hair1"]="people with blond hair"

    # # Concept Slider
    # ["/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/age"]="a young face"
    # ["/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/smiling"]="a smiling face"
    # ["/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/bignose"]="face with a big nose"
    # ["/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/beards"]="a face with beards"
    # ["/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/blonde"]="people with blond hair"
    # ["/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/smiling"]="a smiling face|concept_slider"
    # ["/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/bignose"]="face with a big nose|concept_slider"
    # ["/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/beards"]="a face with beards|concept_slider"
    # ["/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/blonde"]="people with blond hair|concept_slider"

    ["/p/realai/zhenghao/LoRA_SAE/SwiftEdit/output_images/smiling"]="a smiling face"
    ["/p/realai/zhenghao/LoRA_SAE/SwiftEdit/output_images/big_nose"]="face with a big nose"
    ["/p/realai/zhenghao/LoRA_SAE/SwiftEdit/output_images/young"]="a young face"
    ["/p/realai/zhenghao/LoRA_SAE/SwiftEdit/output_images/mustache"]="a face with beards"
    ["/p/realai/zhenghao/LoRA_SAE/SwiftEdit/output_images/blond_hair"]="people with blond hair"
)

mkdir -p clip_results/ours

for dir_path in "${!CONFIGS[@]}"; do
    prompt="${CONFIGS[$dir_path]}"
    
    case "$prompt" in
        "a young face") feature_name="Young" ;;
        "a smiling face") feature_name="Smiling" ;;
        "face with a big nose") feature_name="Big_Nose" ;;
        "a face with beards") feature_name="Mustache" ;;
        "people with blond hair") feature_name="Blond_Hair" ;;
        *) feature_name="Unknown" ;;
    esac
    
    output_file="clip_results/swift/clip_scores_${feature_name}.csv"
    
    echo ""
    echo "Processing: $feature_name"
    echo "Directory: $dir_path" 
    echo "Prompt: $prompt"
    echo "Output: $output_file"
    echo "----------------------------------------"
    
    if [ -d "$dir_path" ]; then
        python clip_batch.py \
            --image_dir "$dir_path" \
            --prompt "$prompt" \
            --output "$output_file"
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully processed $feature_name"
        else
            echo "✗ Error processing $feature_name"
        fi
    else
        echo "✗ Directory not found: $dir_path"
    fi
done

echo ""
echo "========================================="
echo "Batch processing completed!"
echo "Results saved in: clip_results/"
echo ""
echo "Generated files:"
find clip_results/ -name "*.csv" -type f | sort
