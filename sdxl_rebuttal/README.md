# CASL on SDXL — rebuttal experiment

This directory is deliberately isolated from the original CASL research
snapshot. It evaluates a newly trained SAE and semantic alignment map at the
`stabilityai/stable-diffusion-xl-base-1.0` U-Net mid-block. The SDXL backbone is
never fine-tuned.

The matched dense baseline is **not** claimed as an exact Asyrp reproduction.
It is named **Asyrp-style matched dense H-space**: it replaces the SAE code
with the raw mid-block representation and otherwise receives the same images,
CLIP directional objective, preservation loss, optimizer steps, intervention
layer, timestep range, and strength/norm tuning budget.

Run from `Asyrp_SAE`:

```bash
export HF_HOME=/tmp/casl_hf
export HF_HUB_CACHE=/tmp/casl_hf/hub
PYTHON_BIN=/u/wff7ad/program/anaconda3/bin/python
"$PYTHON_BIN" -m pip install --user -r sdxl_rebuttal/requirements.txt
"$PYTHON_BIN" -m sdxl_rebuttal.experiment all --config sdxl_rebuttal/configs/smoke.yaml --device cuda:0
"$PYTHON_BIN" -m sdxl_rebuttal.evaluate --config sdxl_rebuttal/configs/smoke.yaml --device cuda:0
"$PYTHON_BIN" -m sdxl_rebuttal.analyze --config sdxl_rebuttal/configs/smoke.yaml
```

The explicit interpreter is intentional: the repository's legacy activated
environment is Python 3.9 / PyTorch 2.1 and cannot import the pinned modern
Diffusers release. The interpreter above is Python 3.12 / PyTorch 2.7 on this
machine and leaves the legacy environment unchanged.

Individual resumable stages are `collect`, `train-sae`, `invert`,
`train-align`, `edit`, and `verify`. Every command writes a JSONL event log;
existing complete artifacts are reused unless `--force` is passed.

Validation and full commands:

```bash
# Core CASL + matched dense sweep
"$PYTHON_BIN" -m sdxl_rebuttal.experiment all \
  --config sdxl_rebuttal/configs/validation.yaml --device cuda:0

# Official Concept Slider weights (Smiling and Age/Young only)
mkdir -p /tmp/casl_sdxl/concept_sliders
curl -L --fail -o /tmp/casl_sdxl/concept_sliders/smiling.pt \
  https://sliders.baulab.info/weights/xl_sliders/smiling.pt
curl -L --fail -o /tmp/casl_sdxl/concept_sliders/age.pt \
  https://sliders.baulab.info/weights/xl_sliders/age.pt
"$PYTHON_BIN" -m sdxl_rebuttal.external_concept_slider \
  --config sdxl_rebuttal/configs/validation.yaml --device cuda:0 --concept Smiling
"$PYTHON_BIN" -m sdxl_rebuttal.external_concept_slider \
  --config sdxl_rebuttal/configs/validation.yaml --device cuda:0 --concept Young
"$PYTHON_BIN" -m sdxl_rebuttal.external_concept_slider \
  --config sdxl_rebuttal/configs/validation.yaml --merge-only

"$PYTHON_BIN" -m sdxl_rebuttal.evaluate \
  --config sdxl_rebuttal/configs/validation.yaml --device cuda:0
"$PYTHON_BIN" -m sdxl_rebuttal.analyze \
  --config sdxl_rebuttal/configs/validation.yaml
```

Replace `validation.yaml` with `full.yaml` for the 32-image run. The full
config uses a separate activation cache, SAE, six semantic maps, inversion
latents, and result directory; no validation checkpoint is silently reused.
For safe multi-GPU execution on this server, run no more than two concurrent
SDXL processes. Per-concept edit commands may be parallelized, then rerun one
all-concept `edit` command without `--force` to consolidate the core manifest.
The full configuration uses `edit.batch_size: 16`; this batches sixteen identical
intervention settings (never different methods or strengths) through the U-Net
and only changes throughput, not the paired inputs or intervention.

The fixed test manifest (`29501.jpg` through `29532.jpg`) was recovered from
the existing Smiling, Young, and Eyeglasses main-experiment output folders.
Edit metrics use the SDXL inversion reconstruction as the reference. The
original source is used only for inversion LPIPS/ArcFace.
