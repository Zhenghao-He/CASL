# Novelty ablation: Official ASYRP vs. SAE-only vs. CASL

## Question

This ablation separates the contribution of CASL's two stages:

- **Official ASYRP:** the authors' CNN-based dense H-space intervention.
- **SAE-only:** the frozen Stage-1 SAE with a top-1 target-positive
  activation-separation heuristic and no learned Stage-2 CLIP mapping.
- **CASL:** the frozen SAE plus the learned Stage-2 CLIP concept alignment,
  using top-1 inference.

The previously attempted matched Dense-CLIP control is not used in the
rebuttal: despite fixing the legacy zero-gradient path and training with an
explicit additive H-space update, it did not learn the requested direction.
Its checkpoints and scores remain archived as a failed diagnostic.

## Controlled evaluation

All outputs use the CelebA-HQ P2 diffusion backbone, IDs `29501`--`29532`,
50-step DDIM inversion, the U-Net middle activation (`512 x 8 x 8`), and a
frozen diffusion model. ASYRP images are the complete official `iter4`
outputs in `Asyrp_official`; no local approximation is relabeled as ASYRP.

SAE-only units are selected from 1,000 pooled activations using the largest
positive standardized target-vs.-non-target mean difference. `Beards` is
defined as `1 - No_Beard`. This label-assisted rule is a reproducible,
favorable proxy for manually inspecting max-activating images; it does not
train a mapping or use the Stage-2 CLIP loss. Both intervention signs are
swept because observational activation association does not determine the
causal sign after injection into the diffusion trajectory.

The first eight IDs were used for SAE-only strength selection and
the remaining 24 are held out for reporting. Edit changes use the shared
zero-intervention inversion reconstruction as reference. Signed target
change, mean absolute change over the other 39 CelebA attributes, and EPR are
averaged over ResNet18, MobileNetV2, VGG16, and ViT-B/16. For Beards, the
signed target is the negative change in `No_Beard`.

## New four-classifier results

| Method | Macro $\Delta$target $\uparrow$ | Macro $\Delta$non-target $\downarrow$ | Macro EPR $\uparrow$ |
|---|---:|---:|---:|
| Official ASYRP | 0.167 | **0.126** | 1.702 |
| SAE-only | 0.002 | **0.069** | 0.666 |
| CASL | **0.302** | 0.155 | **1.832** |

CASL has higher signed target change than ASYRP on four concepts and higher
EPR on Big Nose, Beards, and Blond Hair; ASYRP is higher on Smiling and
Young. Macro target change and EPR favor CASL, whereas macro absolute
off-target change favors the weak SAE-only edit. SAE-only chooses zero edit
for Smiling and Young on validation and has near-zero macro target change on
held-out images, so its preservation is not evidence of specificity.
Therefore the defensible conclusion is not that CASL dominates every cell:
CASL produces the stronger target edit and the best macro EPR, while Stage 1
alone is insufficient. The corrected per-concept table, including 95%
image-bootstrap intervals, is in
`results/sae_only_v2/asyrp_sae_only_casl_table.md`.

## Existing paper comparison

The current paper PDF already compares official ASYRP and CASL in main
Table 1 (not Table 2 in this version). Under that table's original metrics,
CASL has higher EPR, lower LPIPS, and higher ArcFace similarity on all five
concepts. CLIP is equal for Smiling and favors CASL only for Beards, so it
should not be described as an across-the-board CLIP win. These published
numbers are kept separate from the new four-classifier ablation because the
EPR evaluator/aggregation is not identical.

## Reproducibility

```bash
python novelty_ablation/evaluate_attribute_metrics.py \
  --split test --sae-prefix sae_only_v2 --exclude-dense \
  --include-official-asyrp --device cuda:0 \
  --selection novelty_ablation/results/sae_only_v2/selected_strengths.json \
  --output-dir novelty_ablation/results/sae_only_v2

python novelty_ablation/summarize_ablation.py \
  --per-image novelty_ablation/results/sae_only_v2/test_with_asyrp_per_image.csv \
  --output novelty_ablation/results/sae_only_v2/asyrp_sae_only_casl_table.md \
  --bootstrap 5000
```

Raw per-image results and aggregates are under `results/sae_only_v2/`.
