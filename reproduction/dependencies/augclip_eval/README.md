## 💡 Preserve or Modify? Context-Aware Evaluation for Balancing Preservation and Modification in Text-Guided Image Editing
### CVPR 2025 | 📖 [Paper](https://arxiv.org/abs/2410.11374) | ✨ [Project page](https://augclip.github.io/)

> #### Authors &emsp;&emsp; [Yoonjeon Kim](https://akatigre.github.io/)<sup>1&#42;</sup>, [Soohyun Ryu]()<sup>1&#42;</sup>, [Yeonsung Jung]()<sup>1</sup>, [Hyunkoo Lee]()<sup>1</sup>, [Joowon Kim]()<sup>1</sup>, June Yong Yang<sup>1</sup>, Jaeryong Hwang<sup>2</sup>, Eunho Yang<sup>1,3&dagger;</sup> <br> <sub> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <sup>1</sup>KAIST, <sup>2</sup>Korea Navy Academy <sup>3</sup>AITRICS</sub> <br> <sub> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <sup>&#42;</sup>Equal Contribution, <sup>&dagger;</sup>Corresponding author</sub>

![teaser](./assets/teaser.png)


## Get Started

### Installation

1. Clone AugCLIP.
```bash
git clone https://github.com/augclip/augclip
cd augclip
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n augclip_eval python==3.8
pip3 install torch torchvision torchaudio
pip3 install matplotlib openai pillow scikit-learn torchmetrics
pip3 install git+https://github.com/openai/CLIP.git
```

### Notebook
Open [here](./eval_augclip.ipynb) and input openai token.

### Checkpoints

You can obtain the checkpoints for evaluating Segment Consistency and DINO similarity from the provided link.


| Modelname   | Download |
|-------------|----------------------|
| BiseNet-v2  | [download](https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/model_final_v2_coco.pth) |
| DINO        | [download](https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth) |


> #### 🔆 Abstract
> *The development of vision-language and generative models has significantly advanced text-guided image editing, which seeks the preservation of core elements in the source image while implementing modifications based on the target text. However, existing metrics have a context-blindness problem, indiscriminately applying the same evaluation criteria on completely different pairs of source image and target text, biasing towards either modification or preservation. Directional CLIP similarity, the only metric that considers both source image and target text, is also biased towards modification aspects and attends to irrelevant editing regions of the image. We propose **AugCLIP**, a context-aware metric that adaptively coordinates preservation and modification aspects, depending on the specific context of a given source image and target text. This is done by deriving the CLIP representation of an ideally edited image, that preserves the source image with necessary modifications to align with target text. More specifically, using a multi-modal large language model, **AugCLIP** augments the textual descriptions of the source and target, then calculates a modification vector through a hyperplane that separates source and target attributes in CLIP space. Extensive experiments on five benchmark datasets, encompassing a diverse range of editing scenarios, show that **AugCLIP** aligns remarkably well with human evaluation standards, outperforming existing metrics.*

---
### 📚 Citation
```bibtex
@inproceedings{kim2025preserve,
  title={Preserve or Modify? Context-Aware Evaluation for Balancing Preservation and Modification in Text-Guided Image Editing},
  author={Kim, Yoonjeon and Ryu, Soohyun and Jung, Yeonsung and Lee, Hyunkoo and Kim, Joowon and Yang, June Yong and Hwang, Jaeryong and Yang, Eunho},
  journal={arXiv preprint arXiv:2410.11374},
  year={2024}
}
```