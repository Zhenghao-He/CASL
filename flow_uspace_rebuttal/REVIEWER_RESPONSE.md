### Extension to principled \(u\)-space and transformer-based flow models

We thank the reviewer for pointing us to the principled \(u\)-space introduced
in [3]. We agree that block 20 in App. I was used only as a fixed probing
location and should not be interpreted as a canonical Transformer analogue of
the U-Net bottleneck. We therefore replace this ad-hoc location with the
pre-patchification \(u\)-space of [3] and perform a new end-to-end experiment
on a latent Diffusion Transformer trained with flow matching and sampled with
an Euler solver.

#### Experimental setup

We use the public LFM CelebA-HQ-256 DiT-L/2 checkpoint as the frozen
flow-matching backbone. The model operates on latents produced by
`stabilityai/sd-vae-ft-mse`; neither the DiT nor the VAE is fine-tuned during
CASL training. In this architecture, the current flow state has shape

\[
x_t\in\mathbb{R}^{4\times32\times32},
\]

and is subsequently converted into 256 Transformer tokens by the DiT patch
embedder. Following [3], we define \(u\)-space as this continuous latent tensor
*immediately before patchification*. Thus, the intervention is no longer
attached to an arbitrarily selected Transformer block.

We train all CASL components anew at this location:

- **SAE.** A spatially shared token-wise TopK SAE
  \(4\!\rightarrow\!128\!\rightarrow\!4\), with TopK \(=20\), is trained while
  the flow backbone remains frozen. We use 4,096 CelebA-HQ training images and
  four independently sampled flow-path states per image, corresponding to
  16,384 activation tensors or approximately 16.8M spatial tokens. The held-out
  validation explained variance is 0.9997, with 19.1 active coordinates on
  average out of 128.
- **Concept alignment.** With the SAE frozen, each concept learns a
  token-wise linear map \(M_c:\mathbb{R}^{128}\rightarrow\mathbb{R}^{4}\).
  This Stage-2 module has only 516 trainable parameters. It is optimized for
  160 steps on 16 separate training images using the same directional CLIP
  supervision as CASL and an image-preservation term. The importance of each
  SAE coordinate is the squared column norm of \(M_c\), and inference retains
  only the selected top-\(k\) coordinates.
- **Inversion and editing.** We use 100 explicit Euler steps for both
  real-image inversion and reverse editing, following the Euler setting in
  [3]. Under the convention of the evaluated LFM checkpoint, \(t=0\) denotes
  data and \(t=1\) denotes noise; inversion integrates \(0\!\rightarrow\!1\),
  while reconstruction/editing integrates \(1\!\rightarrow\!0\). The
  intervention is active over \(t\in[0.35,0.75]\). Every edited image starts
  from exactly the same inverted latent as its unedited reconstruction.
- **Validation and test separation.** We first use eight validation images to
  determine the sign, top-\(k\), and a reasonable intervention-strength range.
  The confirmatory evaluation then uses 32 target-attribute-absent CelebA-HQ
  images per concept, selected using ground-truth attributes from image IDs
  \(\ge 10000\). These images are disjoint from the 4,096 SAE images, the
  Stage-2 alignment images, and the validation images. We report a common
  moderate \(\alpha=2\) for Young, Beard, Blond Hair, and Male; the
  validation-selected Smiling direction uses \(\alpha=-1\). Stronger
  \(\lvert\alpha\rvert=4\) results are retained but are not used in the main
  table because they produce larger identity changes.
- **Evaluation.** Attribute changes are independently evaluated by ResNet18,
  MobileNetV2, VGG16, and ViT-B/16 CelebA classifiers. We report their mean
  signed target-probability change and the mean of their four EPR values.
  LPIPS uses AlexNet and identity is evaluated with ArcFace. All
  edit-induced metrics compare the edited image with its Euler inversion
  reconstruction; the original image is used only to measure inversion error.

The reported target change is

\[
\Delta_{\mathrm{target}}^c
=
\frac{1}{4}\sum_{m=1}^{4}
s_c\left[
p_m^c(I_{\mathrm{edit}})
-
p_m^c(I_{\mathrm{recon}})
\right],
\]

where \(s_c\) specifies the intended direction. For classifier \(m\), EPR is
computed using classifier logits as

\[
\mathrm{EPR}_m^c
=
\frac{
\mathbb{E}_i\left[
\left|\ell_{m,c}(I_{\mathrm{edit}}^i)
-\ell_{m,c}(I_{\mathrm{recon}}^i)\right|
\right]
}{
\mathbb{E}_i\left[
\frac{1}{C-1}\sum_{j\neq c}
\left|\ell_{m,j}(I_{\mathrm{edit}}^i)
-\ell_{m,j}(I_{\mathrm{recon}}^i)\right|
\right]
+10^{-8}
},
\qquad
\mathrm{EPR}^c=\frac{1}{4}\sum_{m=1}^{4}\mathrm{EPR}_m^c.
\]

Using the reconstruction as the reference separates the intended CASL edit
from numerical inversion and VAE reconstruction errors.

#### From DDIM activation injection to flow-matching dynamics

A flow-matching model learns a time-dependent velocity field

\[
\frac{d x_t}{dt}=v_\theta(x_t,t),
\]

which is integrated by the standard Euler update

\[
x_{k+1}=x_k+\Delta t_k\,v_\theta(x_k,t_k).
\]

At pre-patchification \(u\)-space, CASL first encodes each spatial latent
vector with the frozen SAE and retains the concept-selected sparse
coordinates:

\[
z_{t,p}=E_{\mathrm{SAE}}(u_{t,p}),\qquad
\Delta u_{t,p}^{c}=M_c(P_{K_c}z_{t,p}),
\]

where \(p\) indexes spatial positions and \(P_{K_c}\) retains the top-\(k\)
coordinates selected by the Stage-2 alignment weights. After RMS
normalization, the edited velocity is evaluated at

\[
\widetilde v_t
=
v_\theta\!\left(x_t+\gamma_t\alpha\Delta u_t^c,t\right),
\]

where \(\gamma_t\) is the intervention schedule. The flow state itself is then
advanced using the unmodified Euler rule:

\[
\widetilde x_{k+1}
=
\widetilde x_k
+\Delta t_k\,
v_\theta\!\left(
\widetilde x_k+\gamma_{t_k}\alpha\Delta u_{t_k}^c,t_k
\right).
\]

For a sufficiently small intervention, its local effect on the learned
velocity field is

\[
\widetilde v_t-v_t
\approx
J_{v_\theta,x}(x_t,t)\,
\gamma_t\alpha\Delta u_t^c.
\]

Thus, the sparse semantic intervention remains defined in CASL's SAE
coordinates, while its effect on a flow model is a local, concept-aligned
perturbation of the learned velocity field that is integrated by Euler.

This extension deliberately does **not** copy ASYRP's asymmetric DDIM update
verbatim. ASYRP's update relies on the DDIM-specific decomposition

\[
x_{t-1}
=
\sqrt{\bar\alpha_{t-1}}\,\widehat x_0
+
\sqrt{1-\bar\alpha_{t-1}}\,\epsilon_\theta,
\]

which allows edited and unedited branches to be assigned asymmetrically to the
predicted-clean-image and noise-direction terms. A velocity-based
flow-matching model with Euler integration has no canonical
\(\widehat x_0/\epsilon\) decomposition. We therefore transfer the
activation-space intervention principle, not the DDIM-specific algebra:
CASL modifies the point at which the frozen velocity field is evaluated, and
the resulting velocity is integrated normally.

#### Results

First, the unedited 100-step Euler round trip gives the following
reconstruction quality:

| Euler steps | Unique test images | Reconstruction LPIPS \(\downarrow\) | Reconstruction ArcFace \(\uparrow\) |
|---:|---:|---:|---:|
| 100 | 73 | 0.133 | 0.783 |

The five concept-specific test sets contain 32 images each and 73 unique
images in total. ArcFace is averaged over pairs in which a face is detected in
both images. A solver calibration on eight fixed images yielded reconstruction
LPIPS of 0.159, 0.056, and 0.040 for 50, 100, and 200 Euler steps,
respectively; we use 100 steps because it follows [3] and captures most of the
integration improvement without doubling the cost.

| Concept | Signed target \(\Delta\) \(\uparrow\) | Four-classifier EPR \(\uparrow\) | Edit LPIPS \(\downarrow\) | Edit ArcFace \(\uparrow\) |
|---|---:|---:|---:|---:|
| Smiling | 0.036 | 1.469 | 0.072 | 0.921 |
| Young | 0.139 | 1.338 | 0.163 | 0.677 |
| Beard | 0.034 | 1.165 | 0.221 | 0.647 |
| Blond Hair | 0.110 | 1.550 | 0.144 | 0.761 |
| Male | 0.119 | 1.806 | 0.246 | 0.644 |
| **Macro average** | **0.087** | **1.466** | **0.169** | **0.730** |

All four independent classifiers give a positive mean target direction for
every reported concept. The results exhibit different strength-preservation
regimes rather than a single degenerate behavior. Smiling produces a modest
but highly identity-preserving edit (ArcFace 0.921, LPIPS 0.072). Young and
Blond Hair show substantially larger target changes (0.139 and 0.110) while
retaining ArcFace similarities of 0.677 and 0.761. Male gives the largest
EPR (1.806) and a clear target response (0.119), but also has the largest
perceptual and identity changes, exposing the expected strength-preservation
trade-off. Beard is the weakest result: its target change remains consistently
positive across all four classifiers, but its LPIPS is relatively high
compared with its target change. We include it rather than selecting only the
strongest concepts. ArcFace values are averaged over edit pairs in which faces
are detected in both images.

The macro target change of 0.087 and EPR of 1.466 show that the intervention is
not merely reproducing the inversion output: it changes the intended
attribute more than the average non-target classifier logit. At the same time,
the macro LPIPS of 0.169 and ArcFace of 0.730 show that the moderate-strength
setting does not obtain its target changes solely by completely replacing the
input identity. Stronger settings increase target scores further but degrade
identity, so they are not used for the main claim.

#### Scope of the conclusion

This experiment establishes that CASL's full pipeline—SAE learning,
CLIP-supervised concept alignment, sparse coordinate selection,
activation-space injection, real-image inversion, and end-to-end
steering—can be instantiated at the principled pre-patchification \(u\)-space
of the evaluated latent DiT and used with a flow-matching objective and Euler
sampling. It also shows that the intervention does not require the
DDIM-specific asymmetric update.

We do not claim that \(u\)-space is universally optimal for every Transformer,
that the result proves transfer to every rectified-flow architecture, or that
it empirically covers SD3 or FLUX. The evaluated flow backbone is trained on
CelebA-HQ, so this experiment addresses architecture/objective/sampler
transfer, not zero-shot domain transfer. The separate SDXL experiment addresses
transfer to a large text-conditioned latent U-Net whose backbone was not
fine-tuned on the target face datasets.
