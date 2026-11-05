#!/bin/bash

set -euo pipefail

# Fill these paths before running.
DIT_REPO="/p/realai/zhenghao/LoRA_SAE/DiT"
DIT_CHECKPOINT="/p/realai/zhenghao/LoRA_SAE/DiT-XL-2-256x256.pt"
VAE_SOURCE="stabilityai/sd-vae-ft-ema"

# Local workspace defaults.
IMAGE_DIR="/p/realai/zhenghao/LoRA_SAE/CelebA-HQ/data1024"
OUTPUT_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_activation_dump"

# Extraction settings.
LIMIT=1000
SEED=42
IMAGE_SIZE=256
BATCH_SIZE=4
BLOCK_INDEX=20
TIMESTEP=250
CLASS_LABEL=0
DEVICE="cuda"

python /p/realai/zhenghao/LoRA_SAE/dit_preliminary_validation.py \
  --dit-repo "$DIT_REPO" \
  --dit-checkpoint "$DIT_CHECKPOINT" \
  --vae-source "$VAE_SOURCE" \
  --image-dir "$IMAGE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --limit "$LIMIT" \
  --seed "$SEED" \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --block-index "$BLOCK_INDEX" \
  --timestep "$TIMESTEP" \
  --class-label "$CLASS_LABEL" \
  --device "$DEVICE"
