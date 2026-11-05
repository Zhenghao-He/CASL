# Reviewer 1: CASL alpha sweep

Eight fixed CelebA-HQ images per concept. EPR and attribute changes are averaged over ResNet-18, VGG-16, MobileNetV2, and ViT-B/16; LPIPS uses AlexNet; CLIP-Score uses ViT-L/14. All edit-induced quantities use the alpha=0 reconstruction as reference.

| Concept | alpha | EPR ↑ | CLIP-Score ↑ | LPIPS ↓ | Target delta ↑ | Off-target delta ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Smiling | 0 | 0.000 | 0.197 | 0.000 | 0.000 | 0.000 |
| Smiling | 32 | 3.056 | 0.215 | 0.187 | 0.217 | 0.099 |
| Smiling | 64 | 2.609 | 0.225 | 0.267 | 0.323 | 0.134 |
| Smiling | 96 | 2.208 | 0.237 | 0.383 | 0.307 | 0.175 |
| Smiling | 128 | 2.254 | 0.233 | 0.456 | 0.243 | 0.176 |
| Smiling | 160 | 2.017 | 0.248 | 0.545 | 0.229 | 0.192 |
| Smiling | 192 | 1.943 | 0.238 | 0.585 | 0.102 | 0.192 |
| Young | 0 | 0.000 | 0.201 | 0.000 | 0.000 | 0.000 |
| Young | 32 | 0.962 | 0.203 | 0.171 | -0.002 | 0.084 |
| Young | 64 | 0.929 | 0.208 | 0.255 | 0.062 | 0.143 |
| Young | 96 | 0.772 | 0.203 | 0.332 | 0.039 | 0.184 |
| Young | 128 | 0.818 | 0.211 | 0.408 | 0.004 | 0.176 |
| Young | 160 | 1.063 | 0.202 | 0.502 | -0.056 | 0.195 |
| Young | 192 | 1.058 | 0.200 | 0.583 | -0.179 | 0.212 |
| Big Nose | 0 | 0.000 | 0.177 | 0.000 | 0.000 | 0.000 |
| Big Nose | 32 | 0.919 | 0.182 | 0.078 | 0.018 | 0.045 |
| Big Nose | 64 | 0.998 | 0.183 | 0.104 | -0.036 | 0.049 |
| Big Nose | 96 | 1.141 | 0.189 | 0.159 | -0.036 | 0.075 |
| Big Nose | 128 | 1.084 | 0.193 | 0.190 | -0.027 | 0.089 |
| Big Nose | 160 | 0.907 | 0.192 | 0.218 | -0.042 | 0.102 |
| Big Nose | 192 | 0.890 | 0.186 | 0.264 | -0.042 | 0.122 |
| Beards | 0 | 0.000 | 0.165 | 0.000 | 0.000 | 0.000 |
| Beards | 32 | 2.856 | 0.210 | 0.224 | 0.425 | 0.126 |
| Beards | 64 | 2.849 | 0.225 | 0.308 | 0.607 | 0.168 |
| Beards | 96 | 2.388 | 0.237 | 0.408 | 0.543 | 0.195 |
| Beards | 128 | 2.472 | 0.252 | 0.489 | 0.616 | 0.215 |
| Beards | 160 | 2.230 | 0.231 | 0.559 | 0.558 | 0.222 |
| Beards | 192 | 1.903 | 0.242 | 0.601 | 0.456 | 0.226 |
| Blond Hair | 0 | 0.000 | 0.202 | 0.000 | 0.000 | 0.000 |
| Blond Hair | 32 | 0.923 | 0.204 | 0.120 | 0.103 | 0.072 |
| Blond Hair | 64 | 1.155 | 0.203 | 0.202 | 0.165 | 0.104 |
| Blond Hair | 96 | 1.488 | 0.211 | 0.291 | 0.237 | 0.135 |
| Blond Hair | 128 | 1.982 | 0.213 | 0.348 | 0.375 | 0.141 |
| Blond Hair | 160 | 2.225 | 0.226 | 0.393 | 0.441 | 0.139 |
| Blond Hair | 192 | 2.306 | 0.224 | 0.434 | 0.432 | 0.138 |

OS/MES/OMES characterize the selected SAE coordinates and therefore do not change when only the inference multiplier alpha changes. Their fixed top-1 values are:

| Concept | OS ↑ | MES ↑ | OMES ↑ |
|---|---:|---:|---:|
| Smiling | 0.493 | 0.727 | 0.610 |
| Young | 0.274 | 0.395 | 0.334 |
| Big Nose | 0.395 | 0.697 | 0.546 |
| Beards | 0.464 | 0.655 | 0.559 |
| Blond Hair | 0.552 | 0.450 | 0.501 |

A reproducible validation rule, rather than visual inspection, is to maximize CLIP-Score subject to LPIPS <= 0.40. It selects:

| Concept | Selected alpha | EPR ↑ | CLIP-Score ↑ | LPIPS ↓ | Target delta ↑ |
|---|---:|---:|---:|---:|---:|
| Smiling | 96 | 2.208 | 0.237 | 0.383 | 0.307 |
| Young | 64 | 0.929 | 0.208 | 0.255 | 0.062 |
| Big Nose | 128 | 1.084 | 0.193 | 0.190 | -0.027 |
| Beards | 64 | 2.849 | 0.225 | 0.308 | 0.607 |
| Blond Hair | 160 | 2.225 | 0.226 | 0.393 | 0.441 |

Important diagnostic: for Big Nose, absolute CLIP-Score increases while the independent classifier target delta is slightly negative. Thus CLIP alone is not a universally reliable selector; a preservation constraint plus an independent target check is safer.
