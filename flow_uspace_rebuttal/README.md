# CASL in pre-patchification u-space of a latent DiT flow model

This experiment replaces the ad-hoc Appendix-I block-20 probe with the
pre-patchification latent location called *u-space* in [3]. The public backbone is
the official LFM CelebA-HQ-256 DiT-L/2 checkpoint. It predicts a flow-matching
velocity field and is integrated in both directions with 50 explicit Euler
steps.

The experiment does **not** claim to use the unreleased CelebA U-ViT checkpoint
from [3]. It uses [3]'s intervention location on a separate, public DiT latent
flow backbone.

```bash
CUDA_VISIBLE_DEVICES=0 python flow_uspace_rebuttal/run.py all \
  --config flow_uspace_rebuttal/config_smoke.yaml
```

Stages can also be run separately: `collect`, `sae`, `align`, `verify`, and
`edit`. Artifacts, curves, image grids inputs, and a per-edit manifest are saved
under `results/flow_uspace_smoke`.

The u-space tensor is `[B,4,32,32]` (4096 dimensions per image). The smoke
configuration trains a 4096→8192→4096 ReLU SAE, then learns concept directions
for a matched dense-u control and CASL. Every edit starts from the same inverted
latent and uses the same Euler trajectory and intervention interval.
