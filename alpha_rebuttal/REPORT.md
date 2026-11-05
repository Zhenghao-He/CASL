# Minimal alpha experiments for rebuttal

## Setup

We used the same eight CelebA-HQ images (IDs 29501--29508) for Smiling,
Young, Big Nose, Beards, and Blond Hair. Edit-induced metrics use each
method's inversion reconstruction, rather than the original image, as the
reference. EPR and attribute changes are averaged over ResNet-18, VGG-16,
MobileNetV2, and ViT-B/16. CLIP-Score uses CLIP ViT-L/14 and LPIPS uses
AlexNet.

## Reviewer 1: selecting CASL alpha

CASL was evaluated at alpha = 0, 32, 64, 96, 128, 160, and 192. Macro
results are:

| alpha | EPR ↑ | CLIP-Score ↑ | LPIPS ↓ | Target delta ↑ | Off-target delta ↓ |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.188 | 0.000 | 0.000 | 0.000 |
| 32 | 1.743 | 0.203 | 0.156 | 0.152 | 0.085 |
| 64 | 1.708 | 0.209 | 0.227 | 0.224 | 0.120 |
| 96 | 1.600 | 0.216 | 0.315 | 0.218 | 0.153 |
| 128 | 1.722 | 0.220 | 0.378 | 0.242 | 0.159 |
| 160 | 1.689 | 0.220 | 0.443 | 0.226 | 0.170 |
| 192 | 1.620 | 0.218 | 0.493 | 0.154 | 0.178 |

LPIPS and off-target change increase with alpha, while CLIP-Score and target
change saturate at approximately 128. This supports selecting alpha on a
held-out set by the CLIP--LPIPS Pareto trade-off instead of qualitative
inspection. A concrete rule, maximizing CLIP-Score subject to LPIPS <= 0.40,
selects alpha 96/64/128/64/160 for
Smiling/Young/Big Nose/Beards/Blond Hair.

The per-concept table also exposes an important limitation: for Big Nose,
CLIP-Score improves while the independent target classifier changes slightly
in the wrong direction. Therefore CLIP alone should not be claimed as a
universally reliable criterion. The defensible rule is a preservation
constraint plus validation with an independent target metric.

OS, MES, and OMES characterize which frozen SAE coordinates are selected.
They are fixed for a given top-1 selection and therefore should not change
when only the inference multiplier alpha changes. The alpha-dependent
disentanglement measure is EPR.

## Reviewer 2: method-specific intensity ranges

We swept the native CASL multiplier, official ASYRP delta interpolation, and
the archived Boundary strength grid. The five-concept macro trajectories are:

| Method | Evaluated settings | EPR trajectory ↑ | Target-delta trajectory ↑ |
|---|---|---|---|
| CASL | 0/32/64/128/192 | 0.000/1.743/1.708/1.722/1.620 | 0.000/0.152/0.224/0.242/0.154 |
| ASYRP | 0/0.5/1/1.5/2 | 0.000/1.454/1.554/1.385/1.334 | 0.000/0.103/0.171/0.126/0.089 |
| Boundary | grid 0/5/9/15/19 | 1.196/1.100/1.120/1.403/1.289 | -0.157/0.063/0.000/-0.026/-0.117 |

The coefficients are method-specific and are not directly equated. The
trajectory, rather than an arbitrary single point, shows that CASL's EPR
advantage over ASYRP persists across the tested operating range. CASL is not
merely preserving more through weak editing: its macro target change is also
higher at corresponding sampled operating points. Boundary is reported by
grid index because its archived sweep is two-sided and does not expose the
same alpha semantics.

## Failed/excluded attempt

We tested a quick Concept Slider sweep, but excluded it from the rebuttal
table after auditing the path: the prototype started denoising from the clean
VAE code rather than the method's inverted real-image trajectory. Its very
small measured changes are therefore not a valid Concept Slider intensity
comparison. The images remain under `results/slider/` for auditability and
are not used in the reported conclusions.

## Concise rebuttal wording

**Alpha selection.** We thank the reviewer for pointing out the subjectivity
of per-concept alpha selection. We swept alpha in
{0,32,64,96,128,160,192} on the same eight held-out CelebA-HQ images for all
five concepts. Macro CLIP-Score increases from 0.188 to 0.220 up to
alpha=128 and then saturates, whereas LPIPS increases monotonically from
0.000 to 0.493; target change similarly peaks at alpha=128. Thus alpha can be
selected reproducibly on a held-out set from the CLIP--LPIPS Pareto frontier
(e.g., maximize CLIP-Score under LPIPS <= 0.40), rather than by qualitative
inspection. EPR remains stable across the useful range
(1.60--1.74 for alpha=32--192). OS/MES/OMES measure the fixed selected SAE
coordinates and are consequently independent of the inference multiplier.
We will add the per-concept sweep and clarify that CLIP should be paired with
an independent target check, since Big Nose reveals one classifier
disagreement.

**Fairness across editing ranges.** We additionally swept every baseline
with a defensible native scalar in this setup instead of comparing one
hand-picked setting. Across five concepts, CASL's macro EPR is
1.74/1.71/1.72/1.62 at alpha=32/64/128/192, compared with
1.45/1.55/1.39/1.33 for official ASYRP at delta multipliers
0.5/1.0/1.5/2.0. CASL also produces larger macro signed target changes at
the corresponding sampled operating points. We will report the complete
per-concept trajectories and method-native coefficients, without treating
their numerical scales as equivalent.

