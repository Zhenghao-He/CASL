#!/bin/bash

# LPIPS Calculation Script

# Define input directories
ORIGINAL_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0/images"
  # Replace with your original images directory
EDITED_DIR="/p/realai/zhenghao/LoRA_SAE/sliders/edited_imgs/beards"     # Replace with your edited images directory
OUTPUT_CSV="./lpips_beards_results.csv"        # Output CSV file

# LPIPS calculation parameters
IMAGE_SIZE=512                          # Image resize dimension
NETWORK="alex"                          # LPIPS network: alex or vgg

echo "Calculating LPIPS scores..."
echo "Original directory: $ORIGINAL_DIR"
echo "Edited directory: $EDITED_DIR"
echo "Output file: $OUTPUT_CSV"
echo "----------------------------------------"

# Run LPIPS calculation
python calculate_lpips.py \
    --original_dir "$ORIGINAL_DIR" \
    --edited_dir "$EDITED_DIR" \
    --output "$OUTPUT_CSV" \
    --imsize $IMAGE_SIZE \
    --net $NETWORK

echo "LPIPS calculation completed!"
echo "Results saved to: $OUTPUT_CSV"