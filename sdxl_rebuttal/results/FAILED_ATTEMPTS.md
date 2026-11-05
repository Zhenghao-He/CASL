# Failed / excluded attempts

This file is append-only for the rebuttal experiment; failures are not removed
when later fixed.

## 2026-07-24 — smoke attempt 1

- Stage: activation collection, SDXL loaded successfully.
- Failure: after moving SDXL text encoders to CPU, Diffusers reported
  `pipe.device=cpu`; activation noise was consequently created on CPU while the
  latent was on `cuda:0`.
- Error: `RuntimeError: Expected all tensors to be on the same device`.
- Resolution: bind activation-cache tensors and scheduler timesteps to
  `pipe.unet.device`. This is an implementation-device bug, not a failed
  scientific configuration.

## 2026-07-24 — smoke scientific configuration

- Configuration: 2 images, Smiling, 512 px, 8 SAE images / 50 SAE steps,
  4 semantic-alignment steps, top-k 16, norm-matched alpha 0.05.
- Result: both methods moved the two-classifier Smiling score slightly in the
  wrong direction. ResNet-18 signed target change was -0.0063 for CASL and
  -0.0037 for dense. This is retained as a pipeline smoke test and is not used
  as rebuttal evidence.
- Inversion diagnosis: mean LPIPS was 0.2658; image 29502 (extreme upward pose)
  reconstructed poorly at 512 px. Validation uses native SDXL 1024 px.
- SAE diagnosis: the initial L1 coefficient 1e-4 left about 1,758/5,120
  features active. A retained calibration sweep gave (coefficient,
  explained variance, active features): (0.001, .845, 2126), (0.01, .841,
  2043), (0.03, .833, 1886), (0.1, .804, 1500), (0.3, .815, 907), (1.0,
  .596, 257), and (3.0, .244, 23). Validation/full therefore use 1.0, a
  reconstruction/sparsity compromise selected before seeing validation edits.

## 2026-07-24 — validation attempt 1 (neutral-to-target CLIP direction)

- Configuration: 8 images, 1024 px, 64 SAE images, 400 SAE steps, 100
  alignment steps per method/concept, step indices 5–25, strengths
  0/.025/.05/.1/.2, CASL top-k 16/64/256.
- Preserved at: `results/validation_attempt_neutral_source/` (480 edited
  images plus all per-image metrics/plots/tables) and
  `/tmp/casl_sdxl/validation/attempt_neutral_source/alignment/` (six
  checkpoints).
- Inversion succeeded: mean LPIPS 0.0573 and ArcFace 0.9355.
- Smiling succeeded. At alpha .2: CASL top-k64 target change .1259,
  mean leakage .0193; dense target .1110, leakage .0226.
- Young was weak and favored dense at the strongest point (CASL top-k256
  target .0196/leakage .0172; dense target .0222/leakage .0139).
- Eyeglasses failed for both methods (maximum target change about .0014).
- Diagnosis fixed before attempt 2: neutral-to-target CLIP text differences
  are weak/ambiguous for binary age and glasses. Attempt 2 uses explicit
  old-to-young and without-to-with-glasses directions, 200 steps for every
  concept and both methods. This is a symmetric, predeclared change; no method
  receives extra tuning.

## 2026-07-24 — CLI alias failure

- Parallel `train-align` initially failed before model training because the
  CLI mapped it to `train_align` rather than `train_alignment`.
- The alias was corrected; no checkpoint or image from the failed invocation
  was used.

## 2026-07-24 — validation attempt 2 binary-direction probe

- Configuration: explicit neutral-expression-to-smiling, old-to-young, and
  without-glasses-to-with-eyeglasses CLIP directions; 200 alignment steps for
  both methods; 2 fixed images; alpha 0.1/0.2/0.4/0.8; top-k 64.
- Preserved at: `results/validation_binary_probe/` and
  `/tmp/casl_sdxl/validation/attempt_binary_direction/alignment/`.
- Smiling moved in the requested direction (CASL target change .0409 at
  alpha .2; dense .0490). Young was weak and inconsistent.
- Eyeglasses failed decisively: CASL target change was only .0002 at alpha
  .8 while mean off-target change reached .0248 and LPIPS .0404; the dense
  control behaved similarly. Visual inspection confirmed that no eyeglasses
  were generated.
- Diagnosis: increasing intervention strength only increases unrelated
  changes. A directional CLIP objective alone is insufficient for this local
  object edit. Attempt 3 adds the same binary source-vs-target global CLIP
  classification term to CASL and dense, with all other budgets matched.

## 2026-07-24 — Eyeglasses global-target and schedule probes

- Adding an equal-weight binary source-vs-target CLIP classification term to
  both methods did not make eyeglasses appear in the 2-image visual probe.
- A retained schedule sweep then tested step ranges 0–15, 5–25, 10–35, and
  0–49 at alpha .2/.4/.8 for both methods
  (`results/validation_schedule_probe/`, 48 edited outputs).
- No range succeeded. The largest CASL ResNet-18 target change was .00022
  (all 50 steps, alpha .8), at mean off-target change .03282, LPIPS .05051,
  and ArcFace .86773. Dense reached only .00132 at leakage .07619 and
  ArcFace .83995. MobileNetV2 agreed.
- Conclusion: this SDXL mid-block alignment does not learn the Eyeglasses
  edit under the tested supervision and budgets. The concept remains in the
  validation/full evaluation as a negative result; no strength or time range
  is selected as a successful setting.

## 2026-07-24 — full-run throughput/configuration interruption

- The initial full Concept Slider launch inherited the six-point validation
  grid. Smiling scale 2 completed before the full confirmation grid was reduced
  to the predeclared validation-selected points 0/.5/1/3; its 32 images remain
  under `results/full/edits/Smiling/concept_slider/.../a2/` and were not
  deleted. Scale 4 was interrupted before a batch completed and produced no
  images. Neither setting is silently mixed into the 1,024-row confirmation
  manifest.
- Core full generation was also interrupted once to increase same-setting
  image batching from 4 to 16. All completed images were retained and reused;
  the final consolidated manifest verifies exactly 768 core rows with no
  duplicates or missing configurations.
