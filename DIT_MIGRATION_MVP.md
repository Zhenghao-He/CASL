# DiT Migration MVP

## Goal

Build a small transferability proof experiment that keeps the core idea of `Asyrp_SAE`:

1. extract an internal hidden representation from a diffusion model,
2. train an SAE on that representation,
3. identify concept-aligned sparse features,
4. verify that a small subset of those features predicts a concept.

For the first DiT experiment, do not port the full `delta_h` editing pipeline yet. Port the representation-analysis pipeline first.

## What `Asyrp_SAE` is actually doing

The current project mixes several things, but the SAE path is structurally:

1. Collect hidden states `middle_h` from the diffusion backbone at a chosen timestep.
2. Flatten spatial tokens from shape `[C, H, W]` to `[H*W, C]`.
3. Train an SAE on these hidden vectors.
4. Train an `XBlock` decoder that maps SAE latents back to a hidden-space direction.
5. Rank latent dimensions by decoder weight magnitude.
6. Keep top-k latent dimensions and test whether they align with a semantic concept.
7. Optionally inject the decoded hidden direction back into the diffusion process.

For a migration proof, steps 1/2/3/5/6 are enough.

## Mapping from UNet version to DiT

### Representation source

- `Asyrp_SAE`: uses UNet `middle_h` from the bottleneck.
- DiT: use one transformer block output, e.g. `model.blocks[20]`.

### Tokenization

- `Asyrp_SAE`: spatial positions become tokens after reshaping `[C, H, W] -> [H*W, C]`.
- DiT: tokens already exist as `[B, T, D]`.

### SAE input

Two valid choices:

1. Token-level SAE:
   - input dim = `D`
   - each token is one training sample
   - this is the closest analogue to `Asyrp_SAE`

2. Pooled-image SAE:
   - mean-pool over tokens to `[B, D]`
   - each image is one training sample
   - this is simpler and cheaper, good for MVP

Your current `dit_preliminary_validation.py` uses pooled-image SAE. That is acceptable for a transferability proof.

### Concept scoring

- `Asyrp_SAE`: rank latent dimensions through the trained decoder / downstream classifier.
- DiT MVP: rank latent dimensions with a linear probe (`LogisticRegression`) and then run top-k `LinearSVC`.

This keeps the same core claim: sparse latent features learned on the diffusion hidden space retain semantic structure.

## Recommended MVP experiment

### Claim

"The SAE-based concept alignment used in `Asyrp_SAE` is not specific to UNet hidden maps; the same phenomenon appears on DiT hidden activations."

### Minimal protocol

1. Use ImageNet validation images.
2. Define coarse semantic concepts, e.g. `dog`, `cat`, `bird`.
3. Extract DiT hidden activations from one fixed block and one fixed timestep.
4. Train a simple ReLU SAE on those activations.
5. For each concept:
   - fit a linear classifier on SAE codes,
   - sort code dimensions by coefficient magnitude,
   - evaluate top-k subsets with a linear SVM.
6. Report accuracy for `k in {1, 2, 4, 8, 16}`.
7. Visualize top-activating images for each concept.

If top-k sparse features work on DiT too, that is already a defensible "transferability" result.

## Why this is the right first step

The hardest part of a full port is not SAE training. It is intervention:

- `Asyrp_SAE` injects a decoded direction back into UNet bottleneck features.
- DiT has different residual topology, token layout, and class conditioning.

If you try to port intervention first, you will spend most of the time reengineering model internals instead of proving transferability.

The MVP avoids that trap and isolates the scientific question:

"Does sparse concept structure persist in DiT hidden states?"

## Concrete correspondence to your existing files

### In `Asyrp_SAE`

- `main.py`
  - `--train_sae` trains SAE.
  - `--sae_analysis` runs importance extraction + editing.
- `diffusion_latent.py`
  - `train_sae()` trains the SAE on collected hidden states.
  - `get_important_z()` ranks latent dimensions.
  - `sample_z()` keeps top-k latent codes and prepares SVM features.
  - `svm_z()` runs linear classification.
- `models/xblock/XBlock.py`
  - wraps `SAE encode -> sparse latent select -> direction decode`.

### In your DiT validation code

- `src/dit_preliminary_validation.py`
  - `extract_single_concept()` is the DiT analogue of hidden extraction.
  - `train_sae()` is the DiT analogue of SAE fitting.
  - `fit_alignment_and_probe()` is the DiT analogue of `get_important_z() + svm_z()`.

So the current DiT script is already the correct structural MVP, just simplified.

## Suggested settings for the first run

- model: `DiT-XL/2`
- image size: `256`
- block index: `20`
- fixed timestep: one mid/high-noise timestep (keep fixed across runs)
- samples per class: `200` to `500`
- SAE input dim: `1152`
- SAE expansion (`gamma_sae`): `64` or `128`
- sparsity weight (`lambda_sparse`): `16` to `32`
- concepts: `dog`, `cat`, `bird`

## Success criteria

The experiment is successful if all three are true:

1. SAE reconstructs hidden activations reasonably (loss decreases stably).
2. A small number of latent dimensions gives non-trivial concept accuracy.
3. Top-activating images are visually coherent for the target concept.

You do not need editing results for the first transferability proof.

## Stage 2: full intervention port

Once the MVP works, the next step is to approximate `XBlock` on DiT:

1. switch from pooled-image SAE to token-level SAE,
2. add a small decoder `z -> delta_token`,
3. inject `delta_token` into a chosen transformer block residual stream,
4. only perturb top-k SAE dimensions,
5. compare generated samples before/after intervention.

That would be the real DiT analogue of `edit_by_sae()`.

## Practical recommendation

Do the project in two milestones:

1. Representation transfer proof:
   - use the existing DiT validation script,
   - produce accuracy tables and top-activation images.

2. Editing transfer proof:
   - implement token-level SAE and block residual injection.

The first milestone is enough for a clean small experiment and is the highest-leverage next step.
