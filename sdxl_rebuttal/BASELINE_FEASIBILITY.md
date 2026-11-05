# External baseline feasibility log

Checked on 2026-07-24 against the official repositories.

## Concept Sliders

- Official source: <https://github.com/rohitgandikota/sliders>
- Vendored commit: `710c995c34652371444ce041c4890e821d0a7a91`.
- The official repository has an SDXL text-slider trainer
  (`trainscripts/textsliders/train_lora_xl.py`) and SDXL inference notebook.
- The official real-image notebook uses 50-step null-text inversion, but only
  with `CompVis/stable-diffusion-v1-4` at 512 px. It is not itself an SDXL
  real-image implementation.
- The official weights index
  (<https://sliders.baulab.info/weights/xl_sliders/>) provides `smiling.pt`
  and `age.pt`, but no Eyeglasses slider. Their SHA-256 hashes are
  `6430ab47393ba15222ea0988c3479f547c8b59f93a41024bcddd7121ef7147d1`
  and `8c1c096f7cc1109b4072cbc604c811a5f0ff034fc0f6dc7cf66a558550aa4890`.
- We load those two weights through the official `LoRANetwork` wrapper, then
  apply them to the exact CASL SDXL DDIM inversion latents and neutral prompt.
  This preserves the official learned intervention while equalizing the source
  trajectory. Positive scale is used for Smiling and negative scale for Young
  because the released age slider's positive direction is old.
- Eyeglasses remains N/A for this external baseline rather than substituting a
  non-official weight or training on the evaluation images.

## MasaCtrl

- Official source: <https://github.com/TencentARC/MasaCtrl>
- The official `run_synthesis_sdxl_processor.py` supports SDXL only for paired
  text-to-image synthesis from shared random noise. It explicitly notes that
  its SDXL hyperparameters may not be optimal.
- The official real-image notebook performs DDIM/null-text inversion with
  Stable Diffusion v1.4 at 512 px. The repository does not provide a combined
  official SDXL real-image path.
- Excluded: combining the old real-image notebook with the newer SDXL
  attention processor would be a substantial unsupported reimplementation and
  would not consume the same inversion trajectory as the controlled methods.

The controlled external-free comparator is therefore always the
**Asyrp-style matched dense H-space** intervention. It is not described as an
exact Asyrp reproduction.
