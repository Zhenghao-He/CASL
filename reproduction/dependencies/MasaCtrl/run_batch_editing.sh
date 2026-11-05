#!/bin/bash

# Batch MasaCtrl Image Editing Script

# Define variables
INPUT_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0/images"
  # Replace with your input images directory
OUTPUT_DIR="./workdir/celeba/beards"  # Replace with your output directory
TARGET_PROMPT="a photo of a person with beards"  # Replace with your editing prompt
SOURCE_PROMPT=""  # Usually empty for real image editing

# MasaCtrl parameters
STEP=4
LAYER=10
GUIDANCE_SCALE=7.5
NUM_INFERENCE_STEPS=50
SEED=42

# Model and device settings
MODEL_PATH="CompVis/stable-diffusion-v1-4"
DEVICE="cuda"

# Run the batch editor
python batch_masactrl_editor.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --target_prompt "$TARGET_PROMPT" \
    --source_prompt "$SOURCE_PROMPT" \
    --step $STEP \
    --layer $LAYER \
    --guidance_scale $GUIDANCE_SCALE \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --seed $SEED \
    --model_path "$MODEL_PATH" \
    --device "$DEVICE"

echo "Batch processing completed! Check results in $OUTPUT_DIR"