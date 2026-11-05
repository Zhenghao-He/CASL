#!/bin/bash

# Script to run the batch image editor

# Define variables
INPUT_DIR="datasets/celeba-hq"  # Replace with the path to your input images
OUTPUT_DIR="edited_imgs/smiling"  # Replace with the path to save edited images
LORA_PATH="models/smilinglider_alpha1.0_rank4_noxattn/smilinglider_alpha1.0_rank4_noxattn_last.safetensors"  # Replace with the path to your LoRA weights
SCALES="128"  # Define the scales for editing
START_NOISE=500  # Define the start noise level
DEVICE="cuda"  # Define the device to use (e.g., cuda or cpu)

# Run the batch image editor
python batch_image_editor.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --lora_path "$LORA_PATH" \
    --scales $SCALES \
    --start_noise $START_NOISE \
    --device $DEVICE \
    --start_noise $START_NOISE \
    --device $DEVICE