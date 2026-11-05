# CASL on frozen SDXL-base: rebuttal experiment

## Status and question tested

This is a rebuttal experiment, not a paper-wide revision. The implemented path
uses `stabilityai/stable-diffusion-xl-base-1.0` (base only), a fixed neutral
prompt, 50-step DDIM inversion, and the SDXL U-Net mid-block. The diffusion
backbone is frozen and is not fine-tuned on FFHQ/CelebA. A new SAE and new
concept-alignment maps are trained for SDXL; no old CASL weights are reused.

The completed validation run contains 8 fixed CelebA-HQ images (`29501.jpg`–
`29508.jpg`), all three concepts, 7 intervention strengths, CASL top-k
{16, 64, 256}, and a matched dense comparator. It contains 768 evaluated
outputs after adding official Concept Slider SDXL weights for Smiling and
Age/Young. The completed 32-image confirmation uses the recovered main-paper
IDs `29501.jpg`–`29532.jpg`; its fresh SAE, six alignment maps, inversions, and
image generation are kept separate from validation. It evaluates 1,024
manifested outputs: 768 CASL/dense and 256 official Concept Slider outputs.

## Controlled comparison

The dense method is labeled **Asyrp-style matched dense H-space**, not an exact
Asyrp reproduction. It bypasses the SAE but shares CASL's training images,
CLIP directional and target losses, preservation loss, optimizer, step budget,
mid-block layer, timestep range, and norm-matched strength sweep. All edit
metrics compare against the inversion reconstruction. Only inversion LPIPS and
ArcFace compare against the original.

The main comparison is interpolation along each method's Pareto envelope over
their common target-strength support. This prevents a weaker edit from
appearing more specific merely because it changes less.
Validation sweeps top-k 16/64/256 and seven strengths. The 32-image run fixes
top-k 64 and steps 5–25 globally for every concept, then confirms four
predeclared strengths (0/.2/.4/.8); it does not choose a different favorable
top-k per concept.

| Concept | Matched target change | Leakage CASL / dense ↓ | LPIPS CASL / dense ↓ | ArcFace CASL / dense ↑ | Diagnosis |
|---|---:|---:|---:|---:|---|
| Smiling | 0.1602 | **0.0336 / 0.0359** | **0.0298 / 0.0356** | **0.7579 / 0.7328** | small, consistent CASL advantage |
| Young | 0.0718 | 0.0522 / **0.0450** | 0.0495 / **0.0416** | 0.7164 / **0.7679** | dense is better |
| Eyeglasses | 0.00905 | 0.0248 / **0.0165** | 0.0233 / **0.0149** | 0.8396 / **0.9134** | weak edit; dense is better |
| Macro | — | 0.0369 / **0.0325** | 0.0342 / **0.0307** | 0.7713 / **0.8047** | no overall CASL superiority |

As an alternate fixed-strength summary at alpha 0.2, the three-concept macro
target change is 0.0608 for CASL (95% clustered bootstrap CI 0.0365–0.0912)
and 0.0654 for dense (0.0390–0.0955); leakage is 0.0259 vs 0.0283, LPIPS
0.0188 vs 0.0224, and ArcFace 0.836 vs 0.830. The intervals overlap, and the
target strengths are not identical, so the matched-target table remains the
primary specificity test.

The two independent attribute classifiers agree on the Smiling result. At
top-k 64 and strength 0.2, CASL changes Smiling probability by 0.159
(95% bootstrap CI 0.083–0.251; MobileNetV2: 0.188), with leakage 0.0307,
LPIPS 0.0249, and ArcFace 0.775. Dense changes the target by 0.147
(CI 0.079–0.226; MobileNetV2: 0.170), with leakage 0.0306, LPIPS 0.0264,
and ArcFace 0.772. Across the matched-target curve, CASL is slightly better
on all three preservation measures for Smiling.

The negative results matter. Young favors dense at matched strength.
Eyeglasses did not become meaningful in validation even after testing ranges
0–15, 5–25, 10–35, and 0–49. In the 32-image run its largest CASL target
change is only 0.018 (CI 0.0003–0.0418) and requires leakage 0.050, LPIPS
0.050, and ArcFace 0.674; dense is stronger but likewise poorly specific.
The official Concept Slider produces strong Smiling and Young edits, but
strong settings visibly alter geometry and identity; no official Eyeglasses
SDXL weight was available.

EPR is computed exactly as paper Eq. 12–14: aggregate mean absolute target
classifier-logit change divided by aggregate mean absolute non-target
classifier-logit change (`epsilon=1e-8`). Signed probability change is reported
separately because EPR alone can reward an edit in the wrong direction. At
Smiling alpha 0.2, EPR is 3.66 for CASL and 3.44 for dense; all EPR values and
both mean/max off-target changes are retained in `sweep_summary.csv`.

## Inversion and engineering checks

Full-run inversion reconstruction has mean LPIPS 0.0628 and mean ArcFace
0.8424 versus the source (ArcFace averages detected faces). The hook fires
exactly 50 times. A zero-strength
intervention is bit-exact to inversion-only in latent space, all methods use
the same prompt/seed/inversion latent, and finite/shape checks pass.

The full SAE uses 256 training images and 393,216 sampled 1,280-dimensional
mid-block tokens, with expansion 8 and 1,200 steps. Its final objective is
0.275 and it averages about 721 active features out of 10,240. Expansion 8,
rather than the paper's default 128, is a compute-driven SDXL limitation.

Recorded wall times on four A100-80GB GPUs include: full activation collection
232 s, SAE training 81 s, 32-image inversion 236 s, and 272–313 s per
300-step alignment map. Validation evaluation (768 rows, including LPIPS,
ArcFace, two classifiers, and auxiliary directional CLIP) took 453 s. The
full run omits only auxiliary directional CLIP; all 1,024 primary-metric rows
are complete (110 s using the same ArcFace model through its GPU provider).

## Baseline feasibility and failed attempts

- Official Concept Slider commit `710c995c34652371444ce041c4890e821d0a7a91`
  and released `smiling.pt`/`age.pt` weights run on the identical inversion
  latents and prompt. Age uses the released negative slider direction.
- MasaCtrl is excluded. Its official SDXL path is shared-noise synthesis,
  whereas its official real-image editing path targets SD1.4. Including it
  would require a substantial unsupported port.
- The original directional-only alignment worked for Smiling but failed on
  Young/Eyeglasses. Adding symmetric source-versus-target CLIP supervision
  improved Young but did not rescue Eyeglasses.
- The neutral-source attempt, binary-direction probe, global-target probe, and
  Eyeglasses timestep-range probe are retained rather than discarded.

## Claim and limitation

The defensible claim is narrow: CASL can be trained and applied end-to-end in a
large pretrained, text-conditioned latent U-Net without fine-tuning the
diffusion backbone on the target face datasets, and it yields a small,
consistent specificity advantage for Smiling. The experiment does **not** show a universal
advantage across concepts. CASL still learns the SAE and alignment on
target-domain images, so this is not fully zero-shot domain generalization.
Nothing here is empirical evidence for DiT, SD3, FLUX, flow matching, or
rectified flow; those remain plausible but unverified extensions.

## Rebuttal paragraph

We added an end-to-end experiment on frozen SDXL-base (32 real images), using a fixed neutral
prompt and 50-step DDIM inversion of the same real CelebA-HQ images. We train a
fresh SAE and alignment at the SDXL U-Net mid-block; the diffusion backbone is
not fine-tuned on FFHQ/CelebA. Against a norm-, data-, layer-, timestep-, and
optimization-matched dense H-space intervention, CASL shows a clear
matched-target specificity gain for Smiling (off-target change 0.034 vs 0.036,
LPIPS 0.030 vs 0.036, ArcFace 0.758 vs 0.733). The broader result is mixed:
dense is better for Young and neither method produces a meaningful Eyeglasses
edit, so we do not claim universal superiority. This supports an end-to-end
transfer path to a large text-conditioned latent U-Net, not zero-shot domain
generalization and not empirical transfer to DiTs or rectified-flow models.

## Reproduction

Run from `Asyrp_SAE` with the Python interpreter and cache variables in
`sdxl_rebuttal/README.md`. The main commands are:

```bash
python -m sdxl_rebuttal.experiment all --config sdxl_rebuttal/configs/validation.yaml --device cuda:0
python -m sdxl_rebuttal.external_concept_slider --config sdxl_rebuttal/configs/validation.yaml --device cuda:0 --concept Smiling
python -m sdxl_rebuttal.external_concept_slider --config sdxl_rebuttal/configs/validation.yaml --device cuda:0 --concept Young
python -m sdxl_rebuttal.external_concept_slider --config sdxl_rebuttal/configs/validation.yaml --merge-only
python -m sdxl_rebuttal.evaluate --config sdxl_rebuttal/configs/validation.yaml --device cuda:0
python -m sdxl_rebuttal.analyze --config sdxl_rebuttal/configs/validation.yaml
```

Replace `validation.yaml` with `full.yaml` for the 32-image confirmation.
Every stage is resumable. Per-image results, sweep summaries, matched-target
tables, fixed-budget summaries, Pareto AUCs, plots, and qualitative grids are
under `sdxl_rebuttal/results/validation/` and `sdxl_rebuttal/results/full/`.
