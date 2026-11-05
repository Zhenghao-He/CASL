### DiT $u$-space and flow-matching extension

We thank the reviewer for suggesting the $u$-space in [3]. We agree that
block 20 in App. I was only a preliminary probing location. We replace it with
the pre-patchification $u$-space and evaluate CASL on a frozen latent
DiT-L/2 trained with flow matching. Here, the continuous VAE latent immediately
before patch embedding is

$$
u_t=x_t\in\mathbb{R}^{4\times32\times32}.
$$

We treat its 32-by-32 spatial positions as tokens and share the SAE across
positions. Hence, the SAE is 4-to-128-to-4 per token: 128 is a 32-fold channel
expansion, rather than a 128-dimensional representation of the whole image.
This preserves the spatial layout and avoids a large image-specific flattened
SAE. We train this TopK SAE and a CLIP-supervised Stage-2 alignment map anew;
the flow backbone remains frozen.

For the flow dynamics

$$
\frac{d x_t}{dt}=v_\theta(x_t,t),
$$

let $E_{\mathrm{SAE}}$ be the frozen token-wise SAE encoder,
$P_{K_c}$ the concept-specific projector that retains the top-$k$ coordinates
selected for concept $c$, and $M_c$ the CLIP-supervised linear map from the
128-dimensional sparse code back to the four $u$-space channels. CASL produces

$$
z_{t,p}=E_{\mathrm{SAE}}(u_{t,p}),\qquad
\Delta u_{t,p}^c=M_c\left(P_{K_c}z_{t,p}\right),
$$

where $p$ indexes the spatial location. $M_c$ is trained with CASL's
directional CLIP loss; the squared column norms of $M_c$ rank the SAE
coordinates used by $P_{K_c}$. This separates unsupervised decomposition
($E_{\mathrm{SAE}}$) from human-concept alignment ($M_c$ and $P_{K_c}$).

We inject the resulting shift before patchification, as suggested by [3]. The
shift changes the point at which the frozen velocity field is evaluated, while
the flow state is still advanced by the original Euler solver:

$$
\widetilde v_t
=v_\theta\left(x_t+\gamma_t\alpha\Delta u_t^c,t\right),
\qquad
\widetilde x_{k+1}
=\widetilde x_k+\Delta t_k\widetilde v_{t_k}.
$$

Locally,

$$
\widetilde v_t-v_t
\approx J_{v_\theta,x}\gamma_t\alpha\Delta u_t^c.
$$

Thus, a sparse semantic displacement in $u$-space induces a local perturbation
of the learned velocity field. This is the flow counterpart of
activation-space intervention: it preserves the pretrained vector field and
the Euler integrator rather than introducing a new sampler.

We test whether this construction supports real-image steering, rather than
only reconstructing the input. We use 100-step Euler inversion and sampling
and evaluate Smiling, Young, Beard, Blond Hair, and Male on 32
target-attribute-absent CelebA-HQ images per concept. The intervention settings
are selected on a separate eight-image validation set. Target change and EPR
are averaged over four independent attribute classifiers (ResNet18,
MobileNetV2, VGG16, and ViT-B/16). All edit metrics use the Euler
reconstruction as reference, while the original image is used only for
reconstruction LPIPS and ArcFace.

| Concept | Target change $\uparrow$ | EPR $\uparrow$ | LPIPS $\downarrow$ | ArcFace $\uparrow$ |
|---|---:|---:|---:|---:|
| Smiling | 0.036 | 1.469 | 0.072 | 0.921 |
| Young | 0.139 | 1.338 | 0.163 | 0.677 |
| Beard | 0.034 | 1.165 | 0.221 | 0.647 |
| Blond Hair | 0.110 | 1.550 | 0.144 | 0.761 |
| Male | 0.119 | 1.806 | 0.246 | 0.644 |
| **Macro** | **0.087** | **1.466** | **0.169** | **0.730** |

All four classifiers agree on the positive target direction for every
concept. The Euler reconstruction itself achieves LPIPS 0.133 and ArcFace
0.783 against the original image. Smiling provides the strongest preservation;
Young and Blond Hair give larger target changes; Male gives the highest EPR
but also the clearest strength--identity trade-off.

We do not copy ASYRP's asymmetric DDIM update: it relies on a
predicted-clean/noise decomposition that has no canonical counterpart in a
velocity-based Euler solver. Therefore, transferring that algebra verbatim
would be unjustified. We instead transfer its activation-intervention
principle: CASL modifies the internal representation used to predict the
velocity, and Euler integrates the modified velocity normally. This result
supports the evaluated transformer flow model; it does not establish that
$u$-space is universally optimal or empirically cover all DiT and
rectified-flow architectures.
