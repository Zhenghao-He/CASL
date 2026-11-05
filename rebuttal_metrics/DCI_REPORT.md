# DCI rebuttal experiment

## Protocol

DCI is evaluated on the same independent CelebA-HQ slice used for the
corrected equal-budget OMES experiment:

- 4,096 images, IDs 20001--24096;
- one frozen diffusion model and frozen 65,536-dimensional SAE;
- fixed timestep 509, noise seed 2026, and spatial-mean pooling;
- five binary factors: Smiling, Young, Big Nose, Beards, and Blond Hair;
- 679 active latents at the primary scale-calibrated threshold 0.01.

DCI requires one joint representation for all factors. For each budget `k`,
CASL contributes `k` unique active latents from each concept ranking. If two
rankings select the same latent, the later ranking advances until it
contributes `k` new latents. High activation takes the top `5k` active
latents, and each random control samples `5k` active latents without
replacement. Thus every selection uses exactly the same total dimension.

Following the repository's official
`third_party/transfer_disentanglement` implementation, five
`GradientBoostingClassifier` models predict the ground-truth factors and
their feature importances form the DCI importance matrix. The first
two-thirds of the independent slice are used for training and the final
third for testing. DCI-I is mean test accuracy. Balanced accuracy is
reported additionally because the CelebA binary factors are imbalanced.
Random results use 100 independently sampled active subspaces.

## Results

| k | Random D/C/I | High-activation D/C/I | CASL D/C/I |
|---:|---:|---:|---:|
| 1 | 0.017/0.027/0.722 | 0.015/0.027/**0.723** | **0.021/0.054**/0.721 |
| 2 | 0.038/0.043/0.732 | 0.048/**0.067/0.742** | **0.053**/0.064/0.740 |
| 4 | 0.080/0.073/0.748 | 0.089/0.090/0.755 | **0.114/0.104/0.766** |
| 8 | 0.148/0.111/0.766 | 0.185/0.144/0.775 | **0.192/0.153/0.792** |
| 16 | 0.233/0.151/0.782 | 0.261/0.167/0.787 | **0.276/0.174/0.794** |

CASL is consistently strongest at the practically useful moderate budgets
`k={4,8,16}`, improving disentanglement, completeness, and informativeness
over both high activation and the random mean. At `k=8`, for example, CASL
obtains D/C/I `0.192/0.153/0.792`, compared with
`0.185/0.144/0.775` for high activation and
`0.148/0.111/0.766` for random active latents. Its informativeness exceeds
the random 95% interval for every `k` from 4 to 16. The low-budget result is
more mixed: CASL has the best D and C at `k=1`, while high activation is
slightly more informative; at `k=2`, CASL has the best D but not C or I.

## Rebuttal wording

```markdown
We further evaluate the selected SAE representation with DCI on an
independent set of 4,096 CelebA-HQ images. Random active, high-activation,
and CASL selection use the same frozen SAE, five factors, train/test split,
and exactly the same latent budget. At `k=8` per factor, CASL obtains
disentanglement/completeness/informativeness of `0.192/0.153/0.792`, versus
`0.185/0.144/0.775` for high activation and `0.148/0.111/0.766` for random
active latents. CASL is best on all three DCI criteria for
`k={4,8,16}`, supporting that Stage 2 selects a representation that is not
only more factor-specific but also retains predictive information. We will
include DCI together with OMES in the revision.
```

