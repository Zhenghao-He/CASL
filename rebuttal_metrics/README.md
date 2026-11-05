# OMES and AugCLIP evaluation

This directory contains the additional metrics requested during rebuttal.

- OMES follows the official NeurIPS 2024 implementation in
  `third_party/transfer_disentanglement`. We use unnormalised CLIP ViT-B/16
  image embeddings as the common representation for every editing method.
  The source/edit pairs for one concept are treated as the weakly supervised
  pairs that differ in one factor. The primary setting is `alpha=0.5` with
  average pooling, matching the paper's main setting. OS, MES, and their
  average (OMES) are all reported.
- AugCLIP follows the official CVPR 2025 implementation in
  `third_party/augclip_eval`. The script uses CLIP ViT-B/16, weighted linear
  SVM separation, and projection onto the separating hyperplane. Attribute
  descriptions are loaded from a reproducible JSON cache. CelebA-HQ source
  descriptions use its ground-truth visual attributes. LSUN-Church and
  AFHQ-Dog descriptions are generated per image with Florence-2-base-ft,
  because no GPT-4V/OpenAI API key is available on the experiment server.

OMES was proposed for learned representations, not image-editing outputs.
The existing `evaluate_omes_augclip.py` image-level OMES output is retained
only as an implementation audit and must not be reported as a CASL-versus-
baseline result. The rebuttal-facing OMES protocol builds one factor--latent
relation matrix from the full SAE code, then evaluates factor-targeted
subsets selected under the same per-concept dimension budget, including
random-active and high-activation controls.
AugCLIP remains an output-image editing metric and can compare editing
methods.

The equal-budget SAE-selection experiment is reproduced with:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  rebuttal_metrics/extract_celeba_sae_direct.py \
  --start-index 20000 --n-images 4096 --batch-size 32

python rebuttal_metrics/evaluate_selected_omes.py \
  --representation \
    rebuttal_metrics/results/celeba_sae_omes_independent/representations_t509_mean.npy \
  --classes \
    rebuttal_metrics/results/celeba_sae_omes_independent/classes.csv \
  --output-dir \
    rebuttal_metrics/results/celeba_sae_omes_independent/selection \
  --inactive-threshold 0.01 \
  --ks 1 2 4 8 16 \
  --random-seeds 1000
```

This evaluates exactly `k` active dimensions per concept for random-active,
global high-activation, and CASL alignment-weight selection. It first builds
one shared factor--latent relation matrix, so all selection rules use the
same frozen SAE, samples, spatial pooling, factor pairs, and OS/MES code.
Because the official VAE-scale inactive threshold of `0.05` leaves only
three mean-pooled SAE dimensions, the primary scale-calibrated threshold is
`0.01`; results at `0.005` and `0.02` are retained as sensitivity audits.

LSUN-Church and AFHQ-Dog do not provide CelebA-style multi-attribute labels.
For these domains, official Asyrp source/edit outputs serve as independent
factor-intervention probes; CASL edits are not used to construct the OMES
relation matrix. Reproduce both domains with:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  rebuttal_metrics/extract_paired_domain_sae.py --domain church
CUDA_VISIBLE_DEVICES=1 python \
  rebuttal_metrics/extract_paired_domain_sae.py --domain dog

python rebuttal_metrics/evaluate_paired_selection_omes.py \
  --domain church \
  --representation \
    rebuttal_metrics/results/lsun_church_selection_omes/paired_representations.npz \
  --output-dir \
    rebuttal_metrics/results/lsun_church_selection_omes/selection

python rebuttal_metrics/evaluate_paired_selection_omes.py \
  --domain dog \
  --representation \
    rebuttal_metrics/results/afhq_dog_selection_omes/paired_representations.npz \
  --output-dir \
    rebuttal_metrics/results/afhq_dog_selection_omes/selection
```

Run CelebA-HQ:

```bash
CUDA_VISIBLE_DEVICES=0 python rebuttal_metrics/evaluate_omes_augclip.py \
  --config rebuttal_metrics/configs/celeba_hq_table2.json \
  --output-dir rebuttal_metrics/results/celeba_hq_table2
```

Run Appendix Tables 12 and 13:

```bash
# If the released Dog Puppy checkpoint is not present, train the same
# per-concept Asyrp CNN using the cached 40-step AFHQ inversions.
CUDA_DEVICE=1 bash rebuttal_metrics/train_asyrp_dog_puppy.sh

# Generate every available/retrained official-code baseline.
CUDA_DEVICE=1 bash rebuttal_metrics/run_asyrp_appendix.sh all

# The official Appendix path saves a 258x515 source/edit grid. Split it before
# evaluation; configs point to the resulting source/ and edit/ directories.
python rebuttal_metrics/extract_asyrp_grids.py \
  --input-dir ../Asyrp_official/runs/rebuttal_church_gothic_LC_church_outdoor_t999_ninv40_ngen40/test_images/40 \
  --output-dir ../Asyrp_official/runs/rebuttal_church_gothic_split \
  --precomputed-pairs ../Asyrp_official/precomputed/church_outdoor_test_t999_nim32_ninv40_pairs.pth

CUDA_VISIBLE_DEVICES=0 python rebuttal_metrics/evaluate_omes_augclip.py \
  --config rebuttal_metrics/configs/lsun_church_table12.json \
  --output-dir rebuttal_metrics/results/lsun_church_table12 \
  --batch-size 64

CUDA_VISIBLE_DEVICES=0 python rebuttal_metrics/evaluate_omes_augclip.py \
  --config rebuttal_metrics/configs/afhq_dog_table13.json \
  --output-dir rebuttal_metrics/results/afhq_dog_table13 \
  --batch-size 64
```

The cached per-image Florence-2 descriptions are reproducible with:

```bash
python rebuttal_metrics/generate_vlm_descriptions.py \
  --backend florence2 --device cuda --domain church \
  --image-dir runs/church_gothic_LC_church_outdoor_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0/images \
  --output rebuttal_metrics/descriptions/lsun_church_florence2.json

python rebuttal_metrics/generate_vlm_descriptions.py \
  --backend florence2 --device cuda --domain dog \
  --image-dir runs/dog_angry_LC_dog_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0/images \
  --output rebuttal_metrics/descriptions/afhq_dog_florence2.json
```

The evaluator writes per-image AugCLIP scores, method summaries, OMES
association matrices, and a machine-readable audit of missing pairs.
Paired CASL-minus-Asyrp confidence intervals can be regenerated with:

```bash
python rebuttal_metrics/bootstrap_augclip.py \
  --input rebuttal_metrics/results/celeba_hq_table2/augclip_per_image.csv \
  --output rebuttal_metrics/results/celeba_hq_table2/augclip_casl_vs_asyrp_bootstrap.csv
```

Reweight the completed OS/MES results over the full OMES alpha grid without
rerunning CLIP:

```bash
python rebuttal_metrics/omes_alpha_sensitivity.py
```

Here `alpha` is the OMES weight in
`alpha * OS + (1 - alpha) * MES`; it is unrelated to the editing
intervention strength.

For a strict common-concept comparison (for example, before Puppy training
finishes), pass a subset without duplicating the config:

```bash
CUDA_VISIBLE_DEVICES=0 python rebuttal_metrics/evaluate_omes_augclip.py \
  --config rebuttal_metrics/configs/afhq_dog_table13.json \
  --concepts Angry Sleepy Smiling --skip-augclip \
  --output-dir rebuttal_metrics/results/afhq_dog_table13_common3
```
