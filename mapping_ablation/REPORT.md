# Stage-2 linear vs. nonlinear alignment (quick ablation)

We replaced CASL's linear alignment map with a parameter-matched two-layer MLP
(`Linear-SiLU-Linear`). Both variants use the same frozen SAE, eight training
images, eight disjoint test images, five concepts, CLIP/reconstruction losses,
optimizer, seed, inversion cache, sampling configuration, and one training pass. The linear map has
33.55M parameters; the MLP has 33.82M (+0.8%).

|Concept|Mapping|Target $\Delta\uparrow$|Off-target $\Delta\downarrow$|EPR $\uparrow$|CLIP-Score $\uparrow$|LPIPS $\downarrow$|
|---|---|---:|---:|---:|---:|---:|
|Smiling|Linear|-0.008|0.058|**1.196**|**0.198**|0.121|
||MLP|**0.020**|**0.031**|0.981|0.193|**0.113**|
|Young|Linear|0.004|0.037|1.242|**0.202**|0.112|
||MLP|**0.009**|**0.030**|**1.263**|0.197|**0.112**|
|Big Nose|Linear|**0.031**|0.042|0.964|**0.178**|0.137|
||MLP|0.016|**0.031**|**1.163**|0.171|**0.112**|
|Beards|Linear|**0.001**|0.044|**1.206**|**0.166**|0.131|
||MLP|-0.006|**0.032**|1.173|0.159|**0.112**|
|Blond Hair|Linear|-0.015|0.045|**0.960**|**0.204**|0.141|
||MLP|**-0.008**|**0.033**|0.886|0.200|**0.112**|
|**Macro**|**Linear**|0.003|0.045|**1.113**|**0.189**|0.129|
||**MLP**|**0.006**|**0.031**|1.093|0.184|**0.112**|

The MLP does not provide a consistent alignment gain. Its macro target change is
numerically higher but the paired 95% bootstrap interval includes zero
(`+0.0033`, `[-0.0083, 0.0159]`). In contrast, its CLIP-Score is lower for all
five concepts (`-0.0057`, 95% CI `[-0.0085, -0.0030]`) and its macro EPR is also
slightly lower. Its lower off-target change and LPIPS accompany this weaker
CLIP alignment. Thus, this quick ablation gives no evidence that adding
nonlinearity improves Stage 2; the linear map remains the simpler default.

This is an eight-image diagnostic, not evidence that nonlinear mappings can
never help. The archived run used late-step stochastic noise; metrics use each
run's own inversion reconstruction as reference. A deterministic checkpoint-only
rerun was prepared, but the shared checkpoint filesystem returned I/O-busy errors.

## Concise rebuttal text

We additionally replaced the Stage-2 linear map with a parameter-matched
two-layer SiLU MLP (33.82M vs. 33.55M parameters), keeping the frozen SAE,
training data, losses, and optimization budget fixed. Across five concepts, the
MLP did not improve macro EPR (1.093 vs. 1.113) and reduced CLIP-Score for every
concept (0.184 vs. 0.189 macro), while its target-change difference was not
significant. We therefore retain the linear map: the semantic shift is already
represented in the SAE coordinate system, and additional nonlinear capacity
does not yield more effective alignment in this ablation.
