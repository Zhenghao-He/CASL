#!/bin/bash

set -euo pipefail

ACTIVATION_FILE="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_activation_dump/block_20_pooled_activations.pt"
OUTPUT_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_sae"

LATENT_MULTIPLIER=32
ACTIVATION="ReLU"
TOPK=64
BATCH_SIZE=64
EPOCHS=1000
LR=5e-4
L1_WEIGHT=1e-2
DEVICE="cuda"

python /p/realai/zhenghao/LoRA_SAE/train_dit_sae.py \
  --activation-file "$ACTIVATION_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --latent-multiplier "$LATENT_MULTIPLIER" \
  --activation "$ACTIVATION" \
  --topk "$TOPK" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --l1-weight "$L1_WEIGHT" \
  --device "$DEVICE"
