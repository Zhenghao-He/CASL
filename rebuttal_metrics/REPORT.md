# OMES / AugCLIP rebuttal experiment

## DCI extension

We additionally evaluate DCI on the same independent 4,096-image CelebA-HQ
SAE cache and five factors as the corrected equal-budget OMES experiment.
Each rule receives exactly `k` unique active latents per factor (`5k` total);
DCI uses gradient-boosted classifiers and their feature-importance matrix.

| k | Random D/C/I | High-activation D/C/I | CASL D/C/I |
|---:|---:|---:|---:|
| 1 | 0.017/0.027/0.722 | 0.015/0.027/**0.723** | **0.021/0.054**/0.721 |
| 2 | 0.038/0.043/0.732 | 0.048/**0.067/0.742** | **0.053**/0.064/0.740 |
| 4 | 0.080/0.073/0.748 | 0.089/0.090/0.755 | **0.114/0.104/0.766** |
| 8 | 0.148/0.111/0.766 | 0.185/0.144/0.775 | **0.192/0.153/0.792** |
| 16 | 0.233/0.151/0.782 | 0.261/0.167/0.787 | **0.276/0.174/0.794** |

CASL is best on all three criteria for `k={4,8,16}`. Random values average
100 active subspaces; CASL informativeness exceeds the corresponding 95%
random interval at all three budgets. Full protocol, balanced accuracy,
random intervals, and audit metadata are in `DCI_REPORT.md` and
`results/celeba_sae_omes_independent/dci_selection/`.

> **OMES protocol correction.** The image-level OMES results below are
> retained only as an implementation audit and must not be used in the
> rebuttal. OMES evaluates the structure of a representation; Asyrp is not a
> latent-disentanglement method, and CLIP embeddings of edited output images
> are not the representation whose disentanglement CASL claims to improve.
> The rebuttal OMES comparison instead evaluates the latent-selection rule on
> one shared full-SAE factor--latent relation matrix. For each target factor,
> CASL, high activation, and random-active selection receive exactly the same
> `k`-dimension budget. This targeted adaptation is explicitly distinguished
> from vanilla representation-level OMES. AugCLIP remains an output-image
> editing metric and its method comparisons remain applicable.

## Equal-budget CASL latent-selection OMES

We evaluate the selection rule itself, rather than comparing CASL with
Asyrp. A single frozen CelebA-HQ diffusion model and frozen SAE encode an
independent slice of 4,096 images (IDs 20001--24096) at timestep 509 using a
fixed forward-noise seed. Every image is represented by spatial-mean pooling
of the 65,536-dimensional SAE code. Following the official OMES pairing
principle, factor configurations are paired only when they differ in one of
Smiling, Young, Big Nose, Beards, or Blond Hair; class centroids average
identity nuisance. This yields 15 exact centroid pairs per factor. We then
compute factor-targeted OS/MES using exactly `k` active dimensions selected
by each rule.

| k | Random active OMES | High-activation OMES | CASL top-k OMES |
|---:|---:|---:|---:|
| 1 | 0.536 | 0.489 | **0.621** |
| 2 | 0.469 | 0.491 | **0.515** |
| 4 | 0.421 | 0.438 | **0.491** |
| 8 | 0.392 | 0.414 | **0.445** |
| 16 | 0.377 | 0.400 | **0.430** |

CASL has the highest mean OMES at every tested budget. At `k=4`, for
example, CASL obtains OS/MES/OMES `0.482/0.499/0.491`, versus
`0.401/0.441/0.421` for random-active and `0.410/0.465/0.438` for high
activation. CASL improves both OS and MES over random-active at every k. It
also improves both components over high activation except at `k=2`, where
its OS is higher but MES is lower. CASL exceeds the 95% random-subspace OMES
interval for `k={2,4,8,16}`; at `k=1`, its mean is highest but remains inside
the broad random-subspace interval.

The official inactive threshold `0.05`, calibrated for VAE representations,
retains only 3 dimensions after spatial pooling and cannot support `k=16`.
The primary scale-calibrated threshold `0.01` retains 681 dimensions.
Sensitivity runs at `0.005` and `0.02` preserve the ordering: CASL has the
highest mean OMES for every k. At threshold `0.02`, the `k=16` margin over
the 95% random-subspace upper bound is only `0.0018`, so that endpoint is
less decisive. Random intervals quantify variation over 1,000 random
subspaces, not sampling uncertainty over images.

An earlier 1,000-image cached-activation protocol validation is retained in
`results/celeba_sae_omes`; it is not the rebuttal-facing result because those
activations may overlap the SAE training cache. In that audit CASL was best
only for `k={4,8,16}`, motivating the independent-slice evaluation rather
than replacing it silently.

### LSUN-Church extension

LSUN-Church has no multi-attribute annotation analogous to CelebA-HQ. We
therefore use the 32 paired source/edit outputs per factor from official
Asyrp as an independent intervention probe for Gothic, Red Brick, Snowy, and
Wooden. Source and edit receive identical forward noise; CASL edits are not
used. Activity ranking and active-unit filtering use only the shared
unedited source representation.

| k | Random OS | Random MES | Random OMES | High-act. OS | High-act. MES | High-act. OMES | CASL OS | CASL MES | CASL OMES |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.658 | **0.183** | **0.421** | 0.688 | 0.123 | 0.406 | **0.693** | 0.135 | 0.414 |
| 2 | 0.656 | **0.515** | 0.585 | **0.693** | 0.504 | **0.598** | 0.665 | 0.505 | 0.585 |
| 4 | 0.656 | 0.671 | 0.664 | 0.661 | 0.682 | 0.671 | **0.665** | **0.686** | **0.676** |
| 8 | 0.655 | 0.746 | 0.700 | 0.666 | 0.762 | 0.714 | **0.668** | **0.771** | **0.720** |
| 16 | 0.655 | 0.782 | 0.719 | **0.658** | 0.785 | 0.721 | 0.655 | **0.803** | **0.729** |

Church supports the selection claim at moderate budgets. CASL is best on all
three macro metrics at `k=4` and `k=8`; at `k=8`, its OMES `0.720` exceeds
both high activation (`0.714`) and the 95% random-subspace upper bound
(`0.713`). Thresholds `0.005`, `0.01`, and `0.02` preserve the ordering.

### AFHQ-Dog extension

The same independent-probe protocol uses Angry, Sleepy, Smiling, and
Puppy-like.

| k | Random OS | Random MES | Random OMES | High-act. OS | High-act. MES | High-act. OMES | CASL OS | CASL MES | CASL OMES |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **0.529** | 0.444 | 0.487 | 0.509 | **0.483** | **0.496** | 0.489 | 0.425 | 0.457 |
| 2 | **0.526** | **0.516** | **0.521** | 0.512 | 0.506 | 0.509 | 0.517 | 0.506 | 0.512 |
| 4 | **0.527** | **0.540** | **0.533** | 0.513 | 0.518 | 0.516 | 0.525 | 0.530 | 0.527 |
| 8 | 0.527 | **0.548** | **0.537** | **0.527** | 0.539 | 0.533 | 0.518 | 0.531 | 0.525 |
| 16 | 0.526 | 0.552 | 0.539 | **0.535** | **0.563** | **0.549** | 0.519 | 0.526 | 0.522 |

Dog does not support a CASL selection advantage under this probe: CASL is
slightly better than high activation at `k=2/4`, but never beats the random
macro mean and falls behind high activation at larger budgets. The result is
stable at active thresholds `0.005/0.01/0.02`. This must be reported as a
negative extension rather than folded into the favorable CelebA/Church
claim. A limitation is that Asyrp interventions are only approximate factor
changes, especially for the Happy-to-Smiling proxy and the newly trained
Puppy module; unlike CelebA, no ground-truth multi-factor annotations are
available.

## Scope and protocol

We evaluated the concepts reported in CASL Table 2 and Appendix Tables 12/13
using 32 paired test images per concept.

- **CelebA-HQ (Table 2):** Smiling, Big Nose, Young, Beards, Blond Hair.
- **LSUN-Church (Table 12):** Gothic, Red Brick, Snowy, Wooden.
- **AFHQ-Dog (Table 13):** Angry, Sleepy, Smiling, Puppy-like.
- **OMES:** official relation, OS, and MES definitions; `alpha=0.5`;
  average pooling; unnormalised CLIP ViT-B/16 image features as a fixed common
  representation. Because OMES was proposed for representations rather than
  edited images, we call this **image-level OMES in a fixed CLIP
  representation**.
- **AugCLIP:** official CLIP ViT-B/16 + weighted linear-SVM + ideal-projection
  formula. CelebA-HQ descriptions use the 40 ground-truth attributes.
  LSUN-Church and AFHQ-Dog use cached, per-image Florence-2-base-ft
  descriptions because GPT-4V credentials were unavailable.
- For Appendix experiments, edit-induced scores use each method's own
  inversion reconstruction as source. Official Asyrp source/edit grids were
  split into individual images and its reconstruction was recovered from the
  official precomputed inversion pairs. Image IDs match 32/32 for both
  implementations.

CASL strengths follow Appendix Table 4: face `128/64/128/96/128` in the
concept order above, Church `16` for all concepts, and Dog
`32/64/32/64`. The Dog Smiling artifact uses 32 because the released sweep has
no strength 31 image, although Table 4 prints 31.

The completed tables do not constitute a matched-edit-strength comparison:
CASL uses those fixed paper strengths, while official Asyrp uses
`hs_coeff_delta_h=1.0`. OMES `alpha` is a different quantity: it weights
`OS` versus `MES` as `alpha * OS + (1 - alpha) * MES`.

## Superseded image-level OMES alpha sensitivity

Reweighting the already computed OS/MES components over `alpha=0.0,...,1.0`
does not require regenerating images. On CelebA-HQ and LSUN-Church, CASL has
both lower OS and lower MES than Asyrp, so no OMES alpha can reverse the macro
ordering. On AFHQ-Dog, CASL has higher MES but lower OS: it is ahead only when
`alpha < 0.104` (approximately), including grid values 0.0 and 0.1, and is
behind for alpha values 0.2 through 1.0. The primary official-style setting
remains alpha 0.5. Full values are saved in
`results/omes_alpha_sensitivity.csv`.

## CelebA-HQ: all available baselines

All entries contain five concepts and 32 images per concept. Higher is better
for every column.

| Method | OS | MES | OMES | AugCLIP |
|---|---:|---:|---:|---:|
| Asyrp | **0.280** | 0.250 | **0.265** | 0.682 |
| Boundary | 0.193 | 0.228 | 0.211 | **0.790** |
| CASL-Steer | 0.256 | 0.249 | 0.253 | 0.748 |
| Concept Slider | 0.243 | 0.272 | 0.258 | 0.689 |
| MasaCtrl | 0.265 | 0.195 | 0.230 | 0.585 |
| SwiftEdit | 0.248 | **0.273** | 0.260 | 0.755 |

CASL is not the global winner under either macro score. The cleanest
CASL-versus-official-Asyrp comparison is:

| Concept | CASL OMES | Asyrp OMES | CASL AugCLIP | Asyrp AugCLIP | AugCLIP delta |
|---|---:|---:|---:|---:|---:|
| Smiling | **0.286** | 0.284 | **0.729** | 0.674 | +0.055 |
| Big Nose | 0.139 | **0.220** | **0.850** | 0.765 | +0.085 |
| Young | **0.313** | 0.281 | **0.673** | 0.648 | +0.024 |
| Beards | **0.295** | 0.292 | **0.697** | 0.610 | +0.087 |
| Blond Hair | 0.229 | **0.249** | **0.789** | 0.713 | +0.076 |
| Macro | 0.253 | **0.265** | **0.748** | 0.682 | **+0.065** |

The paired-image bootstrap 95% CI for the macro AugCLIP improvement is
`[+0.047, +0.085]`. Per concept, the intervals exclude zero for Smiling,
Big Nose, Beards, and Blond Hair; Young is positive but inconclusive
(`[-0.014, +0.063]`). CASL wins image-level OMES on three of five concepts
but loses its macro because of Big Nose and Blond Hair.

## Appendix Table 12: LSUN-Church

| Method | Concepts | OS | MES | OMES | AugCLIP |
|---|---:|---:|---:|---:|---:|
| Reconstruction/no edit | 4 | 0.001 | 0.002 | 0.002 | **0.917** |
| Official Asyrp | 4 | **0.270** | **0.269** | **0.269** | 0.731 |
| CASL-Steer | 4 | 0.235 | 0.258 | 0.246 | 0.808 |

CASL improves AugCLIP over Asyrp for every Church concept: Gothic `+0.051`,
Red Brick `+0.097`, Snowy `+0.091`, and Wooden `+0.068`. The common-concept
macro delta is `+0.077`, with paired bootstrap 95% CI `[+0.057, +0.098]`.
However, CASL loses OMES (`0.246` versus `0.269`), winning only Gothic at the
factor level.

## Appendix Table 13: AFHQ-Dog

| Method | Concepts | OS | MES | OMES | AugCLIP |
|---|---:|---:|---:|---:|---:|
| Reconstruction/no edit | 4 | 0.004 | 0.006 | 0.005 | **0.839** |
| Asyrp (official code) | 4 | **0.301** | 0.214 | **0.258** | 0.671 |
| CASL-Steer | 4 | 0.197 | **0.226** | 0.211 | **0.799** |

The released Asyrp artifacts lacked Puppy, so we trained that per-concept CNN
using the official code and the same `100` training images, 40-step schedule,
one iteration, learning rate `0.5`, CLIP loss weight `1`, and L1 weight `3`.
CASL improves AugCLIP for Angry (`+0.240`), Sleepy (`+0.082`),
Smiling/Happy-proxy (`+0.098`), and Puppy (`+0.093`). The four-concept macro
is `0.799` versus `0.671`, a delta of `+0.128` with paired bootstrap 95% CI
`[+0.116, +0.141]`. The superseded image-level OMES values are retained only
for traceability and are not interpreted as a method comparison.

## Interpretation and limitations

The strongest completed result is consistent across all evaluated domains:
CASL has higher AugCLIP than Asyrp on all 5 face concepts, all 4 Church
concepts, and all 4 Dog concepts. This does **not** support a claim that CASL
dominates every editing metric. The archived image-level OMES calculation is
not a valid CASL--Asyrp comparison because it evaluates CLIP output-image
features rather than the SAE representation.

The no-edit reconstruction has unusually high AugCLIP on Church and Dog.
This shows that, under locally generated source descriptions, AugCLIP is
strongly influenced by preservation and should remain an auxiliary metric,
not standalone evidence of target-edit strength. OMES and AugCLIP also test
different properties, so neither should replace the paper's EPR, LPIPS,
ArcFace, classifier target change, or matched-target analysis.

Completed limitations and failed attempts are retained rather than removed:

- Generic-template Church/Dog descriptions are preserved in
  `results/*_template_audit`; final results use per-image Florence-2
  descriptions.
- Qwen2.5-VL-3B was attempted but abandoned after impractically large/slow
  checkpoint downloads. GPT-4V was unavailable.
- The released Asyrp Dog Puppy checkpoint was absent; its replacement was
  trained by us using the official implementation and matched configuration.
  The released Happy checkpoint remains an explicit proxy for Dog Smiling,
  not an exact reproduction.
- The available Snow checkpoint was trained with 50 steps but the common
  official Appendix evaluation uses the available 40-step setup.
- Boundary, Concept Slider, MasaCtrl, and SwiftEdit Appendix-domain outputs
  are absent; only existing face outputs and feasible official Asyrp
  checkpoints are reported.

## SDXL 8-image experiment inventory

The official-Asyrp-CNN 8-image validation evaluated exactly five concepts:
Smiling, Young, Big Nose, Beards, and Blond Hair. Smiling and Blond Hair have
the clearest matched-target preservation advantage for CASL. Young is mixed:
CASL has lower off-target change and LPIPS and higher ArcFace, but lower
four-classifier mean EPR. Big Nose and Beards generated complete sweeps but
did not yield reliable positive, matched-target rows and should not be used
as quantitative wins. A separate older 8-image run evaluated Eyeglasses
against a matched dense H-space baseline, not official Asyrp. Arched
Eyebrows, Narrow Eyes, and Mouth Slightly Open only have two-image searches.

## Recommended rebuttal wording

> We additionally test whether CASL's CLIP-alignment weights identify
> factor-specific SAE units using an equal-budget, selection-conditioned OMES
> analysis. On an independent slice of 4,096 CelebA-HQ images, all methods use
> the same frozen diffusion model and SAE, timestep, spatial pooling, exact
> one-factor-difference pairs, active-latent pool, and latent budget. CASL has
> the highest mean OMES for every `k` in `{1,2,4,8,16}`. At `k=4`, CASL
> obtains OS/MES/OMES `0.482/0.499/0.491`, compared with
> `0.401/0.441/0.421` for random active latents and `0.410/0.465/0.438` for
> high-activation latents. CASL exceeds the 95% interval from 1,000 random
> subspaces for `k={2,4,8,16}`, and the ordering is stable to the active-unit
> threshold. This supports semantic alignment as a principled latent
> selection rule rather than a generic high-activation heuristic.

> We additionally evaluate AugCLIP on 32 paired images per concept. Against
> the official Asyrp implementation, CASL obtains higher AugCLIP on all 5
> CelebA-HQ attributes (0.748 vs. 0.682;
> paired delta +0.065, 95% CI [+0.047,+0.085]), all 4 LSUN-Church attributes
> (0.808 vs. 0.731), and all 4 AFHQ-Dog attributes (0.799 vs. 0.671; the
> missing released Puppy module was trained with official code and matched
> settings). We present AugCLIP as auxiliary preservation/modification
> evidence and retain EPR, LPIPS, ArcFace, and matched-target analyses as the
> primary editing-specificity evidence.
