#!/bin/bash

set -euo pipefail

CONCEPT="Eyeglasses"
DIT_REPO="/p/realai/zhenghao/LoRA_SAE/DiT"
DIT_CHECKPOINT="/p/realai/zhenghao/LoRA_SAE/DiT-XL-2-256x256.pt"
VAE_SOURCE="stabilityai/sd-vae-ft-ema"
SAE_CHECKPOINT="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_sae/dit_sae_relu_lm32_l10.01.pt"

IMAGE_DIR="/p/realai/zhenghao/LoRA_SAE/CelebA-HQ/data1024"
ATTR_FILE="/p/realai/zhenghao/LoRA_SAE/CelebA-HQ/list_attr_celeba_shift.txt"
OUTPUT_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_concept_latents"

LIMIT=1000
SEED=42
TOP_K=100
IMAGE_SIZE=256
BATCH_SIZE=4
BLOCK_INDEX=20
TIMESTEP=250
CLASS_LABEL=0
DEVICE="cuda"

python /p/realai/zhenghao/LoRA_SAE/extract_concept_latent_indices.py \
  --concept "$CONCEPT" \
  --dit-repo "$DIT_REPO" \
  --dit-checkpoint "$DIT_CHECKPOINT" \
  --vae-source "$VAE_SOURCE" \
  --sae-checkpoint "$SAE_CHECKPOINT" \
  --attr-file "$ATTR_FILE" \
  --image-dir "$IMAGE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --limit "$LIMIT" \
  --seed "$SEED" \
  --top-k "$TOP_K" \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --block-index "$BLOCK_INDEX" \
  --timestep "$TIMESTEP" \
  --class-label "$CLASS_LABEL" \
  --device "$DEVICE"
