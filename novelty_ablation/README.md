# Novelty ablation: Official ASYRP vs. SAE-only vs. CASL

This experiment uses the original CelebA-HQ P2 diffusion backbone and its
`512 x 8 x 8` U-Net middle `h` activation. All methods use the same 50-step
DDIM inversion, edit interval (`t >= 500`), fixed 32-image test set
(`29501.jpg`--`29532.jpg`), and frozen diffusion backbone.

## Methods

- **Official ASYRP:** uses the authors' CNN-based dense H-space intervention
  and the complete `iter4` outputs under `../Asyrp_official`. The failed
  local Dense-CLIP diagnostic is archived but excluded from the rebuttal.
- **SAE-only:** freezes the shared `512 -> 65536 -> 512` SAE, ranks units by
  standardized positive/negative activation separation (an annotation-assisted
  version of inspecting max-activating images), and decodes only top-1 with
  the frozen SAE decoder. It does not train or use the Stage-2 CLIP mapping.
- **CASL:** freezes the same SAE and uses the paper's concept-specific
  Stage-2 CLIP mapping and top-1 aligned unit.

## Commands

Prepare the SAE-only rankings:

```bash
python novelty_ablation/prepare_sae_only.py
```

Run the SAE-only validation sweep:

```bash
bash novelty_ablation/run_sae_only_sweep.sh smiling 0 8
```

Repeat the two sweep commands for the remaining concepts, then evaluate the
first eight images and select strengths matched to CASL's Table 4 target
strength:

```bash
python novelty_ablation/evaluate_attribute_metrics.py \
  --split validation --device cuda:0
```

Generate strength `0` plus the selected SAE-only strength on all 32 images,
then evaluate official ASYRP, SAE-only, and CASL on held-out IDs
`29509`--`29532`:

```bash
# Example for the validation-selected Smiling strengths.
bash novelty_ablation/run_sae_only_sweep.sh smiling 0 32 -16
python novelty_ablation/evaluate_attribute_metrics.py \
  --split test --include-official-asyrp --device cuda:0

python novelty_ablation/summarize_ablation.py \
  --per-image novelty_ablation/results/test_with_asyrp_per_image.csv \
  --output novelty_ablation/results/asyrp_ablation_table.md \
  --bootstrap 5000
```

The completed selection is stored in `results/selected_strengths.json`, raw
held-out scores in `results/test_with_asyrp_per_image.csv`, and the formatted
table in `results/asyrp_ablation_table.md`.

## Metrics

All edit-induced metrics use the zero-intervention inversion reconstruction,
not the original image, as reference.

- `target_delta`: signed target-attribute probability change, averaged across
  ResNet18, MobileNetV2, VGG16, and ViT-B/16.
- `nontarget_delta`: mean absolute probability change over the other 39
  attributes, averaged across the same four classifiers.
- `epr`: for each classifier, aggregate mean absolute target-logit change
  divided by aggregate mean absolute non-target-logit change; the reported EPR
  is the mean of the four classifier-specific ratios.

For the `beards` edit, the signed target is `-delta No_Beard`.
