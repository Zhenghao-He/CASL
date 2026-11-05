#!/bin/bash

set -euo pipefail

CONCEPT="Mustache"  # Change this to the desired concept (e.g., "Smiling", "Eyeglasses", etc.)
DIT_REPO="/p/realai/zhenghao/LoRA_SAE/DiT"
DIT_CHECKPOINT="/p/realai/zhenghao/LoRA_SAE/DiT-XL-2-256x256.pt"
VAE_SOURCE="stabilityai/sd-vae-ft-ema"
SAE_CHECKPOINT="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_sae/dit_sae_relu_lm32_l10.01.pt"

IMAGE_DIR="/p/realai/zhenghao/LoRA_SAE/CelebA-HQ/data1024"
ATTR_FILE="/p/realai/zhenghao/LoRA_SAE/CelebA-HQ/list_attr_celeba_shift.txt"
INDEX_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_concept_latents"
OUTPUT_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_linear_probe"

POSITIVE_LIMIT=1000
NEGATIVE_LIMIT=1000
SEED=42
TEST_SIZE=0.2
IMAGE_SIZE=256
BATCH_SIZE=4
BLOCK_INDEX=20
TIMESTEP=250
CLASS_LABEL=0
DEVICE="cuda"

python /p/realai/zhenghao/LoRA_SAE/train_concept_linear_probe.py \
  --concept "$CONCEPT" \
  --dit-repo "$DIT_REPO" \
  --dit-checkpoint "$DIT_CHECKPOINT" \
  --vae-source "$VAE_SOURCE" \
  --sae-checkpoint "$SAE_CHECKPOINT" \
  --attr-file "$ATTR_FILE" \
  --image-dir "$IMAGE_DIR" \
  --index-dir "$INDEX_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --positive-limit "$POSITIVE_LIMIT" \
  --negative-limit "$NEGATIVE_LIMIT" \
  --seed "$SEED" \
  --test-size "$TEST_SIZE" \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --block-index "$BLOCK_INDEX" \
  --timestep "$TIMESTEP" \
  --class-label "$CLASS_LABEL" \
  --device "$DEVICE"
