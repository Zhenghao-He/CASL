# Minimal rebuttal evidence: CASL on transformer flow + Euler

## Setup

- Frozen public LFM CelebA-HQ-256 DiT-L/2 flow-matching backbone.
- Intervention at pre-patchification u-space `[B,4,32,32]`, before the DiT
  patch embedder.
- Newly trained token-wise TopK SAE `4 -> 128 -> 4` and newly trained
  CLIP-supervised Stage-2 maps `Linear(128,4)`.
- 100-step explicit Euler inversion and reverse sampling, following the solver
  setting in the u-space paper.
- Five concepts, 32 target-attribute-absent CelebA-HQ images per concept,
  selected from image IDs >=10000. The five sets contain 73 unique images.
- Target metrics are the mean of ResNet18, MobileNetV2, VGG16, and ViT-B/16
  CelebA attribute classifiers. EPR is computed separately per classifier as
  aggregate mean absolute target-logit change divided by aggregate mean
  absolute off-target-logit change, then averaged across classifiers.
- Edit LPIPS and ArcFace use the Euler reconstruction as reference. Original
  images are used only for inversion reconstruction error.

## Reconstruction

| Euler steps | Unique images | Reconstruction LPIPS ↓ | ArcFace ↑ |
|---:|---:|---:|---:|
| 100 | 73 | 0.133 | 0.783 |

ArcFace is averaged over the 52/73 pairs for which InsightFace detected a face
in both images. An 8-image solver sweep gave LPIPS 0.159/0.056/0.040 for
50/100/200 steps. We use 100 steps because it matches the cited u-space paper
and captures most of the numerical-integration improvement.

## Minimal steering table

The reported CASL setting uses the validation-selected direction and top-k.
For Young, Beard, Blond Hair, and Male, a common moderate `alpha=2` is reported
instead of the stronger `alpha=4`, because the latter produced larger target
changes but poor identity preservation. Smiling uses its validation-selected
sign and `alpha=-1`.

| Concept | Target Δ ↑ | Success ↑ | 4-clf EPR ↑ | LPIPS ↓ | ArcFace ↑ | Dense target Δ |
|---|---:|---:|---:|---:|---:|---:|
| Smiling | 0.036 | 0.068 | 1.469 | 0.072 | 0.921 | 0.034 |
| Young | 0.139 | 0.385 | 1.338 | 0.163 | 0.677 | 0.002 |
| Beard | 0.034 | 0.065 | 1.165 | 0.221 | 0.647 | 0.029 |
| Blond Hair | 0.110 | 0.125 | 1.550 | 0.144 | 0.761 | 0.038 |
| Male | 0.119 | 0.202 | 1.806 | 0.246 | 0.644 | 0.004 |
| **Macro** | **0.087** | **0.169** | **1.466** | **0.169** | **0.730** | **0.021** |

All four classifiers give a positive mean target direction for every reported
CASL concept. Success is the negative-to-positive conversion rate among
examples that each classifier itself considered negative before editing,
averaged across the four classifiers. ArcFace is averaged over detected edit
pairs; missing detections remain NaN in the per-image CSV.

The dense column is only a compact sanity control at its independently
validation-selected setting. It is not target-matched, so preservation metrics
must not be compared against that column and the table does not claim CASL
superiority over dense intervention.

## Interpretation and limitations

These results support the narrow architectural claim: CASL's SAE, CLIP
alignment, sparse coordinate selection, and activation injection can be
trained and executed end-to-end at pre-patchification u-space in a latent DiT
trained with flow matching and sampled/inverted with Euler. The claim does not
depend on beating a dense baseline.

The experiment does not transfer ASYRP's DDIM-specific asymmetric update.
CASL changes the point at which the frozen velocity field is evaluated and
Euler integrates the resulting velocity normally. The evidence is for this
public DiT flow model only, not for every DiT, SD3, FLUX, or rectified-flow
architecture. Strong edits (`alpha=4`) substantially reduced identity for
several concepts and are retained in the result directory rather than used in
the compact table.

## Rebuttal paragraph

> Following the reviewer, we replaced the ad-hoc block-20 probe with the
> pre-patchification u-space and evaluated CASL on a frozen latent DiT trained
> with flow matching. We trained a new token-wise SAE and CLIP-supervised
> alignment module at this location and used 100-step Euler inversion and
> sampling. On 32 target-attribute-absent real images per concept, all four
> independent classifiers agreed on the positive editing direction for all
> five concepts; the macro signed target change was 0.087, with four-classifier
> EPR 1.466, edit LPIPS 0.169, and ArcFace 0.730. Euler reconstruction achieved
> LPIPS 0.133 and ArcFace 0.783. This demonstrates that CASL's sparse
> concept-aligned intervention is compatible with the evaluated
> transformer-based flow model and Euler solver. We do not transfer ASYRP's
> DDIM-specific asymmetric update or claim universal generalization to all
> DiT/rectified-flow architectures.

## Artifacts

- Compact table: `results/flow_uspace_large/minimal32_e100_compact.csv`
- Per-image results:
  `results/flow_uspace_large/minimal32_e100_{concept}_per_image.csv` for
  Smiling and
  `results/flow_uspace_large/minimal32_e100_a2_{concept}_per_image.csv` for the
  other four concepts.
- Full strong-edit results:
  `results/flow_uspace_large/minimal32_e100_{concept}_{per_image,summary}.csv`.
- Fixed attribute-filtered IDs:
  `flow_uspace_rebuttal/manifests/{concept}_negative32.txt`.
- Solver sweep:
  `results/flow_uspace_large/minimal_reconstruction_step_sweep.json`.
