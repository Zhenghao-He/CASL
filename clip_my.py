import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

def compute_clip_score(image_dir, prompt, model_name="openai/clip-vit-large-patch14", output_csv=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载模型与预处理器
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # 2. 预处理 prompt
    text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # 3. 遍历文件夹计算每张图的 score
    scores = []
    filenames = []
    for fname in tqdm(sorted(os.listdir(image_dir))):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_dir, fname)
            image = Image.open(image_path).convert("RGB")

            inputs = processor(images=image, return_tensors="pt").to(device)
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            # cosine similarity
            score = (image_features @ text_features.T).item()
            scores.append(score)
            filenames.append(fname)

    if not scores:
        print("No valid images found!")
        return 0, []

    # 计算统计信息
    avg_score = np.mean(scores)
    var_score = np.var(scores)
    std_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    print(f"CLIP Score Statistics:")
    print(f"  Mean: {avg_score:.6f}")
    print(f"  Variance: {var_score:.6f}")
    print(f"  Standard Deviation: {std_score:.6f}")
    print(f"  Min: {min_score:.6f}")
    print(f"  Max: {max_score:.6f}")
    print(f"  Count: {len(scores)}")

    # 保存结果到CSV
    if output_csv:
        # 保存详细分数
        df_scores = pd.DataFrame({
            'filename': filenames,
            'clip_score': scores
        })
        df_scores.to_csv(output_csv, index=False)
        
        # 保存统计信息
        stats_csv = output_csv.replace('.csv', '_statistics.csv')
        df_stats = pd.DataFrame({
            'metric': ['mean', 'variance', 'std', 'min', 'max', 'count'],
            'value': [avg_score, var_score, std_score, min_score, max_score, len(scores)]
        })
        df_stats.to_csv(stats_csv, index=False)
        
        print(f"\nResults saved to: {output_csv}")
        print(f"Statistics saved to: {stats_csv}")

    return avg_score, scores

if __name__ == "__main__":
    # 设置图片目录和提示词
    image_dir = "/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/boundary_good/Blond_Hair_500Range_7/images"
    prompt = "people with blond hair"
    
    # 设置输出CSV文件路径
    output_csv = "clip_scores_blond_hair.csv"
    
    # 计算CLIP分数
    avg_score, all_scores = compute_clip_score(image_dir, prompt, output_csv=output_csv)
    
    print(f"\nFinal Average CLIP-Score: {avg_score:.6f}")
