# ArcFace Face Identity Preservation Evaluation - Experimental Setup

## For Paper Writing Reference

### 1. Evaluation Protocol

**Metric**: ArcFace face identity similarity (cosine similarity on face embeddings)
- **Range**: [0, 1] where 1.0 = identical identity, 0.0 = completely different identity
- **Model**: InsightFace buffalo_l (ResNet-50 backbone, 512-dim embeddings)
- **Implementation**: CPU-based for reproducible results

### 2. Technical Configuration

```python
# Critical parameters for reproducibility:
Detection threshold: 0.3 (lowered from default 0.5)
Detection size: 640×640 pixels
Face alignment: 5-point landmark to 112×112
Embedding dimension: 512 (L2-normalized)
Similarity metric: Cosine similarity
```

### 3. Dataset Specifications

- **Image resolution**: 256×256 pixels, RGB JPEG
- **Image pairing**: 5-digit filename matching (consistent with LPIPS evaluation)
- **Original images**: Diffusion reconstruction results (x0 from inversion process)
- **Evaluation concepts**: Young, Smiling, Big_Nose, Mustache/Beards, Blond_Hair

### 4. Evaluated Methods

1. **Boundary**: Latent space linear boundary manipulation
2. **Asyrp**: Asymmetric reverse process editing
3. **Ours**: SAE-based latent space editing
4. **Reconstruction**: Diffusion inversion baseline
5. **MasaCtrl**: Mutual self-attention control

### 5. Key Experimental Results

| Method | Success Rate | Mean Similarity | Std Dev | Interpretation |
|--------|--------------|-----------------|---------|----------------|
| Example (Boundary-Smiling) | 81% (26/32) | 0.566 | 0.144 | Good identity preservation with variation |

**Success Rate Improvement**: 
- Original setup: 34% success (11/32 pairs)
- Optimized setup: 81% success (26/32 pairs) 
- **Key optimization**: Lowered detection threshold for edited faces

### 6. Interpretation Guidelines for Results

- **>0.8**: Excellent identity preservation (minimal perceptual change)
- **0.5-0.8**: Good identity preservation (noticeable but acceptable changes)
- **<0.5**: Significant identity alteration (substantial identity change)

### 7. Implementation Details for Methods Section

```
We evaluate face identity preservation using ArcFace [Deng et al., CVPR 2019] 
face recognition similarity. The InsightFace buffalo_l model extracts 512-dimensional 
face embeddings after SCRFD face detection and 5-point landmark alignment. 
Identity similarity is measured as cosine similarity between normalized embeddings.

To accommodate edited faces with altered appearance, we lowered the face detection 
threshold to 0.3 (from default 0.5), improving evaluation coverage from 34% to 81% 
of image pairs. All experiments use CPU execution for reproducible results.
```

### 8. Common Failure Cases

- **Face detection failure**: Heavily edited faces that alter facial structure
- **Multiple faces**: Algorithm uses largest/most confident face detection
- **Extreme edits**: Significant structural changes (age, facial hair) may reduce similarity

### 9. Statistical Reporting Template

```
Face identity preservation was evaluated using ArcFace similarity scores. 
Method X achieved a mean similarity of μ ± σ (n=N successful pairs out of M total), 
indicating [excellent/good/poor] identity preservation across [concept] editing tasks.
```

### 10. File Organization for Paper

```
Results structure:
arcface_results/
├── boundary/         # Method 1 results
├── asyrp/           # Method 2 results  
├── ours/            # Our method results
├── reconstruction/  # Baseline results
└── masactrl/        # Method 5 results

Each contains:
- arcface_{concept}.csv        # Raw similarity scores
- arcface_{concept}_statistics.csv  # Summary statistics
- arcface_{concept}_errors.txt      # Failed cases log
```

### 11. Citation Information

```bibtex
@inproceedings{deng2019arcface,
  title={ArcFace: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={4690--4699},
  year={2019}
}

@misc{insightface,
  title={InsightFace},
  author={Deng, Jiankang and Guo, Jia and others},
  url={https://github.com/deepinsight/insightface},
  year={2021}
}
```