# Equal-budget DCI on CASL-selected SAE latents

The same frozen SAE and independent 4,096-image CelebA-HQ representation
cache used by the corrected OMES experiment are evaluated with five binary
factors: Smiling, Young, Big Nose, Beards, and Blond Hair. Every selection
rule receives exactly `5k` unique active latents (`k` per factor). DCI uses
gradient-boosted classifiers and their feature-importance matrix, following
the implementation bundled with `transfer_disentanglement`. Informativeness
is held-out mean classification accuracy.

| k per factor | Selection | DCI-D ↑ | DCI-C ↑ | DCI-I ↑ | Balanced-I ↑ |
|---:|---|---:|---:|---:|---:|
| 1 | Random active | 0.017 | 0.027 | 0.722 | 0.522 |
| 1 | High-activation | 0.015 | 0.027 | **0.723** | 0.529 |
| 1 | CASL | **0.021** | **0.054** | 0.721 | **0.536** |
| 2 | Random active | 0.038 | 0.043 | 0.732 | 0.548 |
| 2 | High-activation | 0.048 | **0.067** | **0.742** | **0.588** |
| 2 | CASL | **0.053** | 0.064 | 0.740 | 0.582 |
| 4 | Random active | 0.080 | 0.073 | 0.748 | 0.586 |
| 4 | High-activation | 0.089 | 0.090 | 0.755 | 0.613 |
| 4 | CASL | **0.114** | **0.104** | **0.766** | **0.625** |
| 8 | Random active | 0.148 | 0.111 | 0.766 | 0.625 |
| 8 | High-activation | 0.185 | 0.144 | 0.775 | 0.652 |
| 8 | CASL | **0.192** | **0.153** | **0.792** | **0.671** |
| 16 | Random active | 0.233 | 0.151 | 0.782 | 0.655 |
| 16 | High-activation | 0.261 | 0.167 | 0.787 | 0.674 |
| 16 | CASL | **0.276** | **0.174** | **0.794** | **0.681** |

Random-active values are means over 100 subspaces. CASL is best on all four
reported quantities for `k={4,8,16}`. Its DCI-I exceeds the random-subspace
95% interval at all three budgets. CASL also exceeds the random interval for
DCI-D at `k=4` and `k=16`, and for DCI-C at `k=8`; the remaining scores are
above the random mean but inside its interval.

