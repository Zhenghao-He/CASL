**W1 (novelty).** We added an ablation that separates the two stages on the
five CelebA-HQ concepts. **Official ASYRP** learns a dense CNN-based
$\Delta h$ without an SAE; **SAE-only** uses the frozen SAE and selects a
top-1 unit by target-positive activation separation, without Stage-2 CLIP
alignment; **CASL** uses both sparse disentanglement and supervised concept
alignment. We report 24 held-out images and average ResNet18, MobileNetV2,
VGG16, and ViT-B/16 attribute evaluators.

| Method | $\Delta$target $\uparrow$ | $\Delta$non-target $\downarrow$ | EPR $\uparrow$ |
|---|---:|---:|---:|
| Official ASYRP | 0.167 | **0.126** | 1.702 |
| SAE-only | 0.002 | **0.069** | 0.666 |
| **CASL** | **0.302** | 0.155 | **1.832** |

SAE-only does not yield reliable human-aligned edits, showing that the SAE is
not sufficient by itself; its low off-target change reflects a near-zero
target edit. Relative to the official dense ASYRP intervention, CASL
increases the signed target change and macro EPR. This supports the role of
Stage 2: it aligns sparse SAE coordinates with the requested human concept
rather than merely combining an SAE with an existing dense edit.
