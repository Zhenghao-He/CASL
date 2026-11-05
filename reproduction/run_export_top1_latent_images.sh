#!/bin/bash

set -euo pipefail

CONCEPT="Young"
DIT_REPO="/p/realai/zhenghao/LoRA_SAE/DiT"
DIT_CHECKPOINT="/p/realai/zhenghao/LoRA_SAE/DiT-XL-2-256x256.pt"
VAE_SOURCE="stabilityai/sd-vae-ft-ema"
SAE_CHECKPOINT="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_sae/dit_sae_relu_lm32_l10.01.pt"

IMAGE_DIR="/p/realai/zhenghao/LoRA_SAE/CelebA-HQ/data1024"
PROBE_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_linear_probe"
OUTPUT_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_top1_images"

SAMPLE_SIZE=256
TOP_IMAGES=32
TOP_K_FEATURES=1
SEED=42
IMAGE_SIZE=256
BATCH_SIZE=4
BLOCK_INDEX=20
TIMESTEP=250
CLASS_LABEL=0
DEVICE="cuda"

python /p/realai/zhenghao/LoRA_SAE/export_top1_latent_images.py \
  --concept "$CONCEPT" \
  --dit-repo "$DIT_REPO" \
  --dit-checkpoint "$DIT_CHECKPOINT" \
  --vae-source "$VAE_SOURCE" \
  --sae-checkpoint "$SAE_CHECKPOINT" \
  --probe-dir "$PROBE_DIR" \
  --image-dir "$IMAGE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --sample-size "$SAMPLE_SIZE" \
  --top-images "$TOP_IMAGES" \
  --top-k-features "$TOP_K_FEATURES" \
  --seed "$SEED" \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --block-index "$BLOCK_INDEX" \
  --timestep "$TIMESTEP" \
  --class-label "$CLASS_LABEL" \
  --device "$DEVICE"
