# CASL–SDXL validation report

## Completed setup

- Backbone: `stabilityai/stable-diffusion-xl-base-1.0` base model only; frozen throughout.
- Concepts completed: Eyeglasses, Smiling, Young. Methods completed: casl, concept_slider, dense.
- Images: 8 fixed CelebA-HQ IDs; 50-step DDIM inversion with the same neutral prompt for reconstruction and editing.
- Primary space: SDXL U-Net mid-block. Resolution: 1024×1024.
- Dense comparator label: **Asyrp-style matched dense H-space** (not an exact Asyrp reproduction).
- All edit-induced metrics use the inversion reconstruction as reference; only inversion error uses the source image.

## Inversion

Mean inversion LPIPS: **0.0573**. Mean ArcFace similarity: **0.9355**.

## Matched-target and Pareto results

Across the common target-strength support of each concept (five-point interpolation on each method's Pareto envelope):

- CASL mean off-target change: 0.0184; dense: 0.0169.
- CASL LPIPS: 0.0162; dense: 0.0183.
- CASL ArcFace: 0.9001; dense: 0.9252.

- Eyeglasses: matched-target mean leakage CASL 0.0084 vs dense 0.0084; LPIPS 0.0093 vs 0.0112; ArcFace 0.9383 vs 0.9520.
- Smiling: matched-target mean leakage CASL 0.0170 vs dense 0.0225; LPIPS 0.0152 vs 0.0260; ArcFace 0.9333 vs 0.9102.
- Young: matched-target mean leakage CASL 0.0298 vs dense 0.0197; LPIPS 0.0240 vs 0.0178; ArcFace 0.8287 vs 0.9133.

The complete sweep, including zero/failed/weak settings, is retained in `sweep_summary.csv`; the comparison above is not a single arbitrarily selected strength.

## Baseline feasibility and failed attempts

- Official Concept Slider SDXL weights are included for Smiling and Age/Young on exactly the same inversion latents, neutral prompt, IDs, and 50-step schedule; no official Eyeglasses SDXL weight was available.
- MasaCtrl was excluded: its official SDXL code path is shared-noise synthesis, while its official real-image path targets SD1.4; porting it would require a substantial unsupported reimplementation.
- The dense control is the controlled causal comparator; it shares the alignment data, CLIP loss, preservation loss, optimizer budget, layer, timestep candidates, and norm-matched strength sweep.

## Limitations and claim scope

This experiment tests a large-scale text-conditioned latent diffusion U-Net whose backbone was not fine-tuned on FFHQ/CelebA. It gives a positive end-to-end result for Smiling, a weaker mixed result for Young, and a failed Eyeglasses edit; it therefore does not establish concept-universal superiority. The SAE and semantic map do use target-domain images, so this is not fully zero-shot domain generalization. It provides no empirical evidence for DiTs, SD3, FLUX, flow matching, or rectified flow.

## Rebuttal-ready wording

We added an end-to-end experiment on the frozen SDXL-base text-conditioned latent U-Net (8 fixed real CelebA-HQ images; Eyeglasses, Smiling, Young; 50-step DDIM inversion). At matched target strength over the common Pareto support, CASL obtained mean off-target attribute change 0.0184 versus 0.0169 for a norm- and budget-matched dense H-space intervention, with ArcFace identity 0.9001 versus 0.9252. CASL produced a clear Smiling edit, but the aggregate comparison was mixed: dense was better on Young and neither method produced a meaningful Eyeglasses edit. Thus the result demonstrates an end-to-end SDXL transfer path rather than universal specificity superiority. The backbone was not fine-tuned on the face datasets, but CASL still learns its SAE/alignment on target-domain images, and we do not claim empirical transfer to DiTs or rectified-flow models.

## Reproducibility

- `run_manifest.json` records the exact config and software versions.
- `per_image_metrics.csv` contains every image/strength/configuration result.
- `matched_target_grid.csv`, `fixed_budget_summary.csv`, and `pareto_auc.csv` contain alternate fair summaries.
- `events.jsonl` records completed stages and wall-clock costs.
