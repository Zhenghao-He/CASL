# CASL–SDXL smoke report

## Completed setup

- Backbone: `stabilityai/stable-diffusion-xl-base-1.0` base model only; frozen throughout.
- Concepts completed: Smiling. Methods completed: casl, dense.
- Images: 2 fixed CelebA-HQ IDs; 50-step DDIM inversion with the same neutral prompt for reconstruction and editing.
- Primary space: SDXL U-Net mid-block. Resolution: 512×512.
- Dense comparator label: **Asyrp-style matched dense H-space** (not an exact Asyrp reproduction).
- All edit-induced metrics use the inversion reconstruction as reference; only inversion error uses the source image.

## Inversion

Mean inversion LPIPS: **0.2658**. Mean ArcFace similarity: **0.2803**.

## Matched-target and Pareto results

No non-degenerate common target-strength interval was available in this stage.

The complete sweep, including zero/failed/weak settings, is retained in `sweep_summary.csv`; the comparison above is not a single arbitrarily selected strength.

## Baseline feasibility and failed attempts

- Concept Slider is not reported unless an official SDXL real-image path can consume the same inversion latent, prompt, image IDs, and 50-step schedule without retraining on evaluation images.
- MasaCtrl is not reported unless its official real-image SDXL path runs without a substantial unsupported port.
- The dense control is the controlled causal comparator; it shares the alignment data, CLIP loss, preservation loss, optimizer budget, layer, timestep candidates, and norm-matched strength sweep.

## Limitations and claim scope

This experiment tests transfer to a large-scale text-conditioned latent diffusion U-Net whose backbone was not trained or fine-tuned on FFHQ/CelebA. The SAE and semantic map do use target-domain images, so this is not fully zero-shot domain generalization. It provides no empirical evidence for DiTs, SD3, FLUX, flow matching, or rectified flow.

## Rebuttal-ready wording

The completed run did not provide a non-degenerate matched-target interval, so it does not yet support a comparative specificity claim. It verifies the SDXL execution path only.

## Reproducibility

- `run_manifest.json` records the exact config and software versions.
- `per_image_metrics.csv` contains every image/strength/configuration result.
- `matched_target_grid.csv`, `fixed_budget_summary.csv`, and `pareto_auc.csv` contain alternate fair summaries.
- `events.jsonl` records completed stages and wall-clock costs.
