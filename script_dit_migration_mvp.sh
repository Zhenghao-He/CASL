#!/bin/bash

set -euo pipefail

# Minimal transferability proof on DiT:
# hidden extraction -> SAE -> top-k probing

export XDG_CACHE_HOME=/p/realai/zhenghao/cache

DIT_REPO="/path/to/DiT"
DIT_CKPT="/path/to/DiT-XL-2-256x256.pt"
IMAGENET_VAL="/path/to/imagenet/val"
CLASS_ID_MAP="/path/to/imagenet_class_id_map.json"
VAE_SOURCE="/path/to/sd-vae-ft-ema"
OUT_DIR="/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_migration_mvp"

python /p/realai/zhenghao/XToken/src/dit_preliminary_validation.py full \
    --dit-repo "$DIT_REPO" \
    --dit-checkpoint "$DIT_CKPT" \
    --imagenet-val "$IMAGENET_VAL" \
    --class-id-map "$CLASS_ID_MAP" \
    --vae-source "$VAE_SOURCE" \
    --output-dir "$OUT_DIR" \
    --device cuda \
    --block-index 20 \
    --samples-per-class 300 \
    --input-dim 1152 \
    --gamma-sae 128 \
    --lambda-sparse 32
