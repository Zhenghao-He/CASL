# Reviewer 2: native intensity sweeps

Eight fixed CelebA-HQ images and five concepts. The normalized level is only an index over five sampled settings from each method's own native sweep; raw coefficients (and Boundary grid indices) are not assumed to be comparable across methods.

| Method | Native strengths at normalized 0/.25/.5/.75/1 | EPR at normalized 0/.25/.5/.75/1 ↑ | Target delta at normalized 0/.25/.5/.75/1 ↑ |
|---|---|---|---|
| CASL | 0/32/64/128/192 | 0.000/1.743/1.708/1.722/1.620 | 0.000/0.152/0.224/0.242/0.154 |
| ASYRP | 0/0.5/1/1.5/2 | 0.000/1.454/1.554/1.385/1.334 | 0.000/0.103/0.171/0.126/0.089 |
| Boundary | 0/5/9/15/19 | 1.196/1.100/1.120/1.403/1.289 | -0.157/0.063/0.000/-0.026/-0.117 |

This table reports the full trajectories rather than each method's best point, so preservation caused merely by weak editing is visible. Per-concept values and off-target changes are retained in `results/epr_sweeps/summary.csv`.
