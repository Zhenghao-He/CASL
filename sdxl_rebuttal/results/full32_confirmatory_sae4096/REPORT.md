# CASL–SDXL rebuttal experiment

## Bottom line

CASL transfers end-to-end to the frozen `stabilityai/stable-diffusion-xl-base-1.0`
text-conditioned latent U-Net, but the specificity advantage is concept-dependent.
The 32-image confirmatory Smiling experiment is a clear positive result. At closely
matched target strength, CASL improves off-target change, LPIPS, ArcFace identity,
and the four-classifier EPR over the matched dense H-space comparator. Young is
mixed and favors the comparator on most preservation metrics after a fair
matched-target baseline refinement. Blond Hair looked positive on 8 images but
did not remain a stable semantic edit on 32 images.

This supports transfer to a large pretrained text-conditioned latent U-Net whose
backbone was not trained or fine-tuned on FFHQ/CelebA. It does **not** support
empirical claims about DiTs, SD3, FLUX, flow matching, or rectified flow. CASL
still uses target-domain CelebA-HQ images to train its SAE and semantic alignment,
so this is not fully zero-shot domain generalization.

## Setup

- Backbone: SDXL-base 1.0, base model only; diffusion backbone frozen.
- Intervention: forward hook on `unet.mid_block` output, shape
  `[batch, 1280, 32, 32]` at 1024×1024 image resolution.
- Inversion/editing: 50-step DDIM, guidance scale 1.0, identical fixed neutral
  prompt, image IDs, inversion latents, layer, and timestep range `[5, 25]`.
- SAE: trained anew for SDXL on 4,096 CelebA-HQ images; 1,280 input dimensions,
  10,240 dictionary features (8× expansion), 26,225,920 parameters, 2,000 steps,
  token batch 4,096, learning rate `3e-4`, L1 coefficient 1.0.
- Stage 2: 64 images, 120 steps, learning rate `1e-4`, preservation weight 0.8.
  Full map is a bias-free `10240→1280` linear layer (13,107,200 parameters);
  low-rank map is `10240→256→1280` (2,949,120 parameters).
- Comparator: adapted official Asyrp DeltaBlock architecture at the same SDXL
  mid-block (4,924,160 parameters), with matched data, CLIP/preservation losses,
  layer, timestep range, and tuning budget. This is labelled
  **Asyrp-style matched dense H-space**, not an exact official Asyrp reproduction.

## Metrics

Edit-induced metrics use the inversion reconstruction as reference. The source
image is used only for inversion error.

- Signed target change and off-target change are computed independently with
  ResNet18, VGG16, MobileNetV2, and ViT-B/16 CelebA classifiers.
- The reported target and off-target values are equal-weight arithmetic means
  across the four classifiers. Per-classifier results remain in every CSV.
- EPR follows the paper: mean absolute target-logit change divided by mean
  absolute non-target-logit change. The main EPR is the equal-weight mean of the
  four backbone-specific EPRs.
- Preservation: LPIPS (AlexNet) and ArcFace (InsightFace buffalo_l).
- AugCLIP/directional CLIP is retained only as an auxiliary metric.

## 32-image confirmatory result

### Smiling: strongest fair comparison

| metric | CASL top-8, α=28 | dense, α=4 | CASL − dense | paired-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| four-model signed target ↑ | 0.2831 | 0.2757 | +0.0074 | [-0.0580, +0.0653] |
| four-model off-target ↓ | 0.0784 | 0.1090 | -0.0307 | [-0.0414, -0.0208] |
| LPIPS ↓ | 0.0600 | 0.1125 | -0.0525 | [-0.0680, -0.0381] |
| ArcFace ↑ | 0.5271 | 0.4066 | +0.1205 | [+0.0248, +0.2019] |
| four-model EPR ↑ | 4.0165 | 2.3536 | +1.6628 | [+1.0772, +2.2772] |

The target-strength difference is only 2.7%, and its paired-bootstrap CI includes
zero. Thus the preservation advantage is not explained by CASL making a weaker
edit. All four classifiers agree on the positive target direction. A second
comparison, CASL α=28 versus dense α=2, is also within 10% target strength and
CASL wins all four preservation/specificity metrics.

### Young: mixed/negative specificity result

The matched pair after explicitly refining the baseline sweep is low-rank CASL
top-32 α=112 versus dense α=0.875:

| metric | CASL | dense |
|---|---:|---:|
| four-model signed target ↑ | 0.0820 | 0.0831 |
| four-model off-target ↓ | 0.0679 | 0.0628 |
| LPIPS ↓ | 0.0493 | 0.0526 |
| ArcFace ↑ | 0.6973 | 0.7202 |
| four-model EPR ↑ | 1.4826 | 1.6237 |

The target mismatch is 1.27%. CASL improves LPIPS only; the dense comparator is
better on off-target change, ArcFace, and EPR. Young therefore does not support a
specificity-superiority claim.

### Blond Hair: exploratory result did not confirm

At 8 images, CASL top-1 α=24 and dense α=0.5 were matched within 1.24%, and CASL
improved off-target, LPIPS, ArcFace, and EPR. At 32 images both effects became
very small and ViT-B/16 changed in the opposite direction. This concept is
reported as an exploratory failure, not a confirmatory win.

## Concept exploration retained in full

The original five concepts were Smiling, Young, Big Nose, Beards, and Blond Hair.
Six additional concepts were trained and swept: Eyeglasses, Male, Bangs,
Gray Hair, Wearing Lipstick, and Black Hair.

- Big Nose and Beards: no stable four-classifier positive edit.
- Eyeglasses: CASL can become stronger than the comparator only with large
  perceptual/identity cost; no matched preservation win.
- Male: classifier directions disagree; strong settings have large identity cost.
- Bangs: promising on 2 images, but failed on the fixed 8-image validation set.
- Gray Hair: both methods produce negligible target change.
- Wearing Lipstick: the comparator is better at matched/near-matched strengths.
- Black Hair: CASL has a low-strength preservation advantage, but does not reach
  the comparator's target strength on 8 images.

Every attempted output and strength remains under
`results/expanded_smoke_sae4096/` and
`results/validation8_expanded_sae4096/`; failed and reverse-direction settings
were not deleted.

## Failed/limited baselines

- The current dense comparator uses the official Asyrp DeltaBlock architecture,
  adapted to the SDXL mid-block and trained with the matched CASL supervision.
  It is not yet a complete reproduction of the official Asyrp training and
  inference pipeline in `Asyrp_official/`.
- Concept Slider/MasaCtrl were not added to this continuation. MasaCtrl's
  supported SDXL path does not provide a directly matched official real-image
  editing route without substantial unsupported reimplementation.
- No claim is made that CASL beats the dense comparator for every concept.

## Compute and reproducibility

This continuation trained 18 new Stage-2 weights and generated/evaluated roughly
900 SDXL edits. It used up to four A100-80GB GPUs for about 1.5 hours wall time
(approximately 5–6 A100 GPU-hours), excluding the previously completed 4,096-image
SAE activation collection/training and DDIM inversions. Shared-storage model
loading was a substantial fraction of wall time.

Primary artifacts:

- `Smiling/per_image_metrics.csv`
- `Smiling/sweep_summary_four_classifier.csv`
- `Smiling/matched_target_four_classifier.csv`
- `Smiling/paired_bootstrap_main.csv`
- `Smiling/plots/`
- `Smiling/qualitative/Smiling_grid.png`
- corresponding full CSVs under `Young/` and `Blond_Hair/`

Key launch pattern:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sdxl_rebuttal.experiment edit \
  --config sdxl_rebuttal/configs/validation8_sae4096.yaml \
  --concepts Smiling --methods casl --strengths 28,32 --topks 8 \
  --test-count 32 --edit-batch-size 2 \
  --result-dir sdxl_rebuttal/results/full32_confirmatory_sae4096/Smiling
```

Evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sdxl_rebuttal.evaluate \
  --config sdxl_rebuttal/configs/validation8_sae4096.yaml \
  --result-dir sdxl_rebuttal/results/full32_confirmatory_sae4096/Smiling \
  --test-count 32 --batch-size 16
```

## Recommended rebuttal wording

We added an end-to-end experiment on the frozen SDXL-base text-conditioned latent
U-Net, using 32 fixed real CelebA-HQ images, 50-step DDIM inversion, and identical
prompts/latents/timesteps across methods. For Smiling, at matched target strength
(four-classifier signed change 0.283 versus 0.276), CASL reduced four-classifier
off-target change from 0.109 to 0.078 and LPIPS from 0.113 to 0.060, while
improving ArcFace identity from 0.407 to 0.527 and mean four-classifier EPR from
2.354 to 4.016. Paired-bootstrap 95% intervals exclude zero for all four
specificity/preservation improvements. Results are concept-dependent: Young is
mixed at matched strength and Blond Hair did not confirm at 32 images. We
therefore claim evidence that CASL can transfer to a large pretrained
text-conditioned latent U-Net without backbone fine-tuning, not universal
superiority, fully zero-shot domain generalization, or empirical transfer to DiT
or rectified-flow architectures.
