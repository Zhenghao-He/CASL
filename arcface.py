import cv2
import numpy as np
import torch
import torch.nn.functional as F
from insightface.app import FaceAnalysis

# 初始化 ArcFace（自动下载权重）
app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

def arcface_similarity(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    f1 = app.get(img1)
    f2 = app.get(img2)
    if len(f1) == 0 or len(f2) == 0:
        raise ValueError("No face detected.")
    e1 = torch.tensor(f1[0].embedding)
    e2 = torch.tensor(f2[0].embedding)
    sim = F.cosine_similarity(e1, e2, dim=0).item()
    return sim

sim = arcface_similarity("before.png", "after.png")
print(f"ArcFace similarity: {sim:.4f}")
