The following values are transcribed from main Table 1 of
`NeurIPS_CASL (9).pdf`. They use the paper's original evaluation protocol and
must not be mixed numerically with the new four-classifier ablation.

| Concept | Method | EPR $\uparrow$ | CLIP $\uparrow$ | LPIPS $\downarrow$ | ArcFace $\uparrow$ |
|---|---|---:|---:|---:|---:|
| Smiling | ASYRP | 3.199 $\pm$ 2.182 | **0.192 $\pm$ 0.014** | 0.358 $\pm$ 0.086 | 0.238 $\pm$ 0.144 |
| Smiling | CASL | **4.465 $\pm$ 2.319** | **0.192 $\pm$ 0.012** | **0.243 $\pm$ 0.085** | **0.566 $\pm$ 0.210** |
| Big Nose | ASYRP | 0.938 $\pm$ 0.608 | **0.173 $\pm$ 0.017** | 0.275 $\pm$ 0.082 | 0.463 $\pm$ 0.182 |
| Big Nose | CASL | **1.060 $\pm$ 0.608** | 0.159 $\pm$ 0.012 | **0.191 $\pm$ 0.050** | **0.761 $\pm$ 0.059** |
| Young | ASYRP | 1.685 $\pm$ 1.220 | **0.211 $\pm$ 0.014** | 0.343 $\pm$ 0.082 | 0.306 $\pm$ 0.139 |
| Young | CASL | **1.817 $\pm$ 1.117** | 0.206 $\pm$ 0.013 | **0.305 $\pm$ 0.082** | **0.387 $\pm$ 0.142** |
| Beards | ASYRP | 1.739 $\pm$ 1.268 | 0.206 $\pm$ 0.047 | 0.353 $\pm$ 0.090 | 0.294 $\pm$ 0.164 |
| Beards | CASL | **2.161 $\pm$ 1.080** | **0.211 $\pm$ 0.043** | **0.233 $\pm$ 0.071** | **0.562 $\pm$ 0.260** |
| Blond Hair | ASYRP | 1.173 $\pm$ 1.002 | **0.186 $\pm$ 0.039** | 0.303 $\pm$ 0.082 | 0.409 $\pm$ 0.210 |
| Blond Hair | CASL | **1.386 $\pm$ 1.213** | 0.170 $\pm$ 0.036 | **0.221 $\pm$ 0.072** | **0.686 $\pm$ 0.155** |
