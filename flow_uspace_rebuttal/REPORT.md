# Transformer-flow u-space extension: live rebuttal report

## Large-data, multi-concept run (latest)

This section supersedes the implementation choices in the original smoke test
below. The experiment uses the public LFM CelebA-HQ-256 DiT-L/2 checkpoint,
pre-patchification u-space `[B,4,32,32]`, and 50-step explicit Euler inversion
and editing.

### Correct paper-aligned SAE and Stage 2

Re-reading the CASL implementation exposed an incorrect initial adaptation:
the smoke SAE flattened all `4x32x32` values into one 4096-vector. The paper
instead treats spatial locations as tokens, shares the SAE over the channel
dimension, and trains a linear Stage-2 map from SAE codes to the intervention.
The corrected flow adaptation is:

- SAE: token-wise `4 -> 128 -> 4`, expansion 32, TopK=20, 1,156 parameters.
- SAE data: 4,096 CelebA-HQ images, four random flow-path samples per image,
  producing 16,777,216 spatial tokens.
- Split: 15,099,494 train and 1,677,722 held-out validation tokens.
- Train/validation explained variance: 0.999688/0.999691.
- Train/validation L0: 19.073/19.073 active coordinates out of 128.
- Stage 2: frozen SAE plus `Linear(128,4)` applied per spatial token, 516
  trainable parameters. Feature importance is the squared column norm of this
  map; inference retains only the selected top-k SAE coordinates.

The failed flattened SAEs and their diagnostics are retained. The original
67.1M-parameter flattened SAE reached validation EV 0.956 but L0 7,149/8,192;
stronger L1 and fixed TopK substantially degraded validation EV. This explains
the weak and non-semantic constant-direction smoke result.

### Scope and concepts

The 8-image validation sweep covers 12 concepts:
`Smiling`, `Young`, `Big Nose`, `Beard`, `Blond Hair`, `Eyeglasses`, `Male`,
`Mustache`, `No Beard`, `Bangs`, `Heavy Makeup`, and `Chubby`. Every attempted
alpha/top-k output is retained. Five concepts received a finer low-strength
sweep after the initial grid showed that `|alpha|=1` was too strong for the
input-dependent mapping.

The dense comparator is a CLIP-supervised dense u-space direction trained with
the same images, loss, intervention time, and 160-step optimization budget. It
is a matched dense u-space control, not an official reproduction of [3] or
ASYRP.

### Results

The 8-image sweep suggested preservation gains for Smiling, No Beard, and Blond
Hair, but Beard and Young failed. The three candidates were then evaluated on
32 images. The validation-selected Blond Hair result did not replicate: matching
the dense target-CLIP change required CASL alpha 4 and caused substantially
worse LPIPS and identity.

| Concept (32 images) | Dense alpha | CASL k / alpha | Dense target CLIP change | CASL target CLIP change | Dense LPIPS | CASL LPIPS | Dense ArcFace | CASL ArcFace |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Smiling | 4 | 16 / -0.43 | 0.000263 | 0.000186 | 0.0734 | **0.0200** | 0.8614 | **0.9682** |
| No Beard | -4 | 8 / -0.075 | 0.000528 | 0.000768 | 0.0713 | **0.0053** | 0.8670 | **0.9789** |
| Blond Hair | 4 | 8 / 4 | 0.005912 | 0.004961 | **0.0824** | 0.2586 | **0.8512** | 0.5642 |

Paired bootstrap 95% intervals for CASL-minus-dense do not cross zero for
Smiling (LPIPS `[-0.0658,-0.0417]`, ArcFace `[+0.0807,+0.1337]`) or No Beard
(LPIPS `[-0.0764,-0.0571]`, ArcFace `[+0.0755,+0.1538]`). They strongly favor
dense for Blond Hair.

These two positive rows are not yet sufficient for a rebuttal claim of general
semantic superiority. Target-CLIP changes are small and directional CLIP is
weak (and negative for some settings), so an independent attribute classifier
or randomized qualitative audit is required before presenting them as
successful target edits. Inspection of `qualitative_matched32.jpg` adds a more
specific concern: the offset-based random test set contains sources that already
smile or already have no beard. It is valid for preservation measurement but
not for target success. A rebuttal table must therefore rerun target success on
attribute-filtered sources (e.g. `Smiling=-1`, `No_Beard=-1`,
`Blond_Hair=-1`) and use an independent CelebA attribute classifier. The
current Smiling/No-Beard rows must not be quoted as final wins.

The defensible result now is:

1. the CASL machinery can be implemented end-to-end at pre-patchification
   u-space in a transformer flow model with Euler inversion/sampling;
2. the correct token-wise SAE scales cleanly to millions of tokens without
   overfitting;
3. sparse Stage-2 intervention has strong preservation at approximately matched
   auxiliary target score for two evaluated concepts, but the effect is not
   concept-universal and Blond Hair is a clear counterexample.

### Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 python flow_uspace_rebuttal/run.py sae \
  --config flow_uspace_rebuttal/config_large.yaml

CUDA_VISIBLE_DEVICES=0 python flow_uspace_rebuttal/run.py align \
  --config flow_uspace_rebuttal/config_large.yaml \
  --concepts smiling young big_nose

CUDA_VISIBLE_DEVICES=0 python flow_uspace_rebuttal/run.py edit \
  --config flow_uspace_rebuttal/config_large.yaml --test-images 8 \
  --concepts smiling young big_nose

CUDA_VISIBLE_DEVICES=0 python flow_uspace_rebuttal/evaluate_large.py \
  --config flow_uspace_rebuttal/config_large.yaml
```

Key artifacts are under `results/flow_uspace_large/`, including all manifests,
per-image metrics, SAE sweeps, alignment curves, 32-image curves, matched-target
identity JSON files, and logs.

## Original smoke-test status

This is a completed **2-image smoke test**, not yet rebuttal evidence. It verifies
the implementation and identifies the comparison that the validation run must
test. No number below should be inserted into the rebuttal as a final result.

## Setup

- Backbone: the public LFM CelebA-HQ-256 `DiT-L/2` checkpoint (FID 5.26).
- Objective/sampler: latent flow matching; 50 explicit Euler steps in both
  inversion (`t=0 -> 1`) and reconstruction/editing (`t=1 -> 0`).
- Intervention: the `[B,4,32,32]` latent immediately before DiT patch embedding,
  i.e. the pre-patchification location called *u-space* in [3]. The DiT then
  produces 256 patch tokens.
- SAE: newly trained `4096 -> 8192 -> 4096` ReLU SAE (67,121,152 parameters) on
  512 u-space samples constructed from 32 images, random times, and random
  flow-path noise. Final train explained variance is 0.993. This small cache is
  suitable only for smoke testing; the validation/full run must use a larger
  independent train/validation cache.
- Concepts aligned: Smiling, Young, and Eyeglasses. End-to-end editing was first
  smoke-tested on Smiling.
- Matched methods: a learned dense u-space direction and CASL SAE-coordinate
  direction. Both use the same two alignment images, CLIP objective, preservation
  loss, 80 optimizer steps, intervention interval `[0.35,0.75]`, inversion
  latents, Euler trajectory, and strength scale.
- Sweep: `alpha in {-4,-2,-1,0,1,2,4}` and CASL
  `k in {1,2,4,8,16}`. One alpha unit is calibrated to u-space RMS 0.05.

The released CelebA U-ViT checkpoint used by [3] is not available in its official
repository. Therefore this experiment uses [3]'s pre-patchification location on
the independently released LFM DiT-flow backbone; it must not be described as an
exact reproduction of [3].

## Implementation verification

- Zero intervention is bit-identical to inversion-only:
  maximum latent absolute difference `0.0`.
- All checked trajectories are finite.
- Shape assertion: u-space `[1,4,32,32]`, SAE input 4096, SAE latent 8192.
- The original and edited methods reuse exactly the same inverted latent.
- Nonzero edits are evaluated relative to the inversion reconstruction.

## Smoke metrics

The closest observed target-strength pair is dense `alpha=1` and CASL
`k=8, alpha=1`. Directional CLIP is auxiliary and the sample contains only two
images.

| Method | k | alpha | Directional CLIP ↑ | LPIPS ↓ | ArcFace ↑ |
|---|---:|---:|---:|---:|---:|
| dense u-space | all | 1 | **0.0888** | 0.0177 | **0.9696** |
| CASL u-space | 8 | 1 | 0.0875 | **0.0169** | 0.9462 |

This does **not** establish a CASL advantage: target strength is closely matched
and LPIPS is slightly lower, but ArcFace is worse. With `n=2`, neither difference
is meaningful. The qualitative grid shows that CASL top-4/8 can cause a clear
smile on one image at `alpha=4`, but is weaker on the other image.

## Failed configurations retained

1. Hard top-8 during alignment starved non-selected coordinates of gradients.
   Smiling directional loss changed only `0.973 -> 0.942`, versus
   `0.973 -> 0.513` for dense. Checkpoints are retained under
   `failed_hard_topk_train/`.
2. Training the full SAE direction optimized well but distributed the semantic
   signal across too many coordinates; top-1/8/16 inference produced almost no
   edit. Artifacts are retained under `failed_full_alignment/`.
3. Unit-RMS perturbations made every nonzero edit collapse. The failed grid and
   trajectories are retained under `failed_rms1/`. The corrected base RMS is
   0.05.
4. Straight-through top-8 is stable and produces sparse edits, but converges more
   weakly than the dense alignment in this smoke test. It remains the current
   candidate for the larger run.

## Correct theoretical formulation

At this location, u-space is the current flow latent itself, not an activation
extracted by an internal transformer block. Write the pretrained velocity
predictor as

\[
v_\theta(x_t,t)=F_\theta(x_t,t).
\]

For frozen SAE encoder and decoder \(E,D\), CASL defines a concept shift in SAE
coordinates:

\[
z_t=E(x_t),\qquad
\Delta u_t^c =
D(z_t+\gamma_t\delta z^c)-D(z_t).
\]

For the linear decoder used here, the shift is equivalently
\(D(\gamma_t\delta z^c)-D(0)\). The shift changes the input at which the frozen
velocity field is evaluated, while the Euler state itself is updated normally:

\[
\tilde v_\theta(x_t,t)
=F_\theta(x_t+\Delta u_t^c,t),
\qquad
x_{k+1}=x_k+\Delta t_k\tilde v_\theta(x_k,t_k).
\]

Here \(\Delta t_k<0\) for reconstruction/editing from noise (`t=1`) to data
(`t=0`), and \(\Delta t_k>0\) for real-image inversion. For a small
intervention,

\[
\Delta v_t
\approx J_{F,x}(x_t,t)\,\Delta u_t^c.
\]

Thus CASL locally perturbs the learned velocity field through sparse SAE
coordinates at u-space. It does not reuse ASYRP's DDIM-specific
predicted-clean-image/direction decomposition or asymmetric update.

## Rebuttal wording (conditional on a successful larger run)

> **Extension to transformer-based flow models.** We agree that block 20 in
> App. I was only a fixed probe and is not a canonical analogue of the U-Net
> bottleneck. Following the reviewer, we instead intervene in the
> pre-patchification latent u-space of a pretrained DiT latent-flow model and
> train a new SAE and concept alignment at this location. Real images are
> inverted and edited with the same 50-step Euler trajectory. Compared with a
> matched dense u-space direction, CASL achieves [TARGET] at matched target
> strength while changing [PRESERVATION/OFF-TARGET RESULTS]. This extension does
> not transfer ASYRP's DDIM-specific asymmetric update: the decoded sparse
> u-space shift changes the point at which the learned velocity field is
> evaluated, and the resulting field is integrated by the standard Euler
> update. These results support the evaluated transformer-flow model only; we do
> not claim that u-space is universally optimal across DiT, SD3, or FLUX.

Do not fill the brackets from this smoke test. If the 8-image matched-target
analysis retains the current ArcFace disadvantage, the correct conclusion is
that the experiment demonstrates *applicability* but not improved specificity.

## Commands and artifacts

```bash
CUDA_VISIBLE_DEVICES=0 python flow_uspace_rebuttal/run.py all \
  --config flow_uspace_rebuttal/config_smoke.yaml
CUDA_VISIBLE_DEVICES=0 python flow_uspace_rebuttal/evaluate_smoke.py \
  --result-dir results/flow_uspace_smoke
```

- `verification.json`: hook/location/zero-intervention checks.
- `sae.pt`, `sae_curve.json`: SAE and training diagnostics.
- `alignment/<concept>/{dense,casl}.pt`: alignment weights and curves.
- `manifest.csv`: every per-image/per-strength output.
- `metrics_per_image.csv`, `metrics_summary.csv`: smoke metrics.
- `qualitative_smoke.jpg`: randomly ordered two-image qualitative grid.
