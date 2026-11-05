#!/usr/bin/env python3
"""
CLIP Score Calculator with Statistics

This script calculates CLIP scores for images in a directory and saves results to CSV.

Usage:
    python clip_batch.py --image_dir /path/to/images --prompt "description" --output output.csv
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse


def compute_clip_score(image_dir, prompt, model_name="openai/clip-vit-large-patch14", output_csv=None):
    """计算CLIP分数并保存统计结果"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 加载模型与预处理器
    print(f"Loading CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # 2. 预处理 prompt
    text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # 3. 遍历文件夹计算每张图的 score
    scores = []
    filenames = []
    image_files = [f for f in sorted(os.listdir(image_dir)) 
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
    
    print(f"Found {len(image_files)} images to process")
    
    for fname in tqdm(image_files, desc="Processing images"):
        try:
            image_path = os.path.join(image_dir, fname)
            image = Image.open(image_path).convert("RGB")

            inputs = processor(images=image, return_tensors="pt").to(device)
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            # cosine similarity
            score = (image_features @ text_features.T).item()
            scores.append(score)
            filenames.append(fname)
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

    if not scores:
        print("No valid images found!")
        return 0, []

    # 计算统计信息
    avg_score = np.mean(scores)
    var_score = np.var(scores)
    std_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    print(f"\nCLIP Score Statistics:")
    print(f"  Prompt: '{prompt}'")
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
        df_scores = df_scores.sort_values('filename')  # 按文件名排序
        df_scores.to_csv(output_csv, index=False)
        
        # 保存统计信息
        stats_csv = output_csv.replace('.csv', '_statistics.csv')
        df_stats = pd.DataFrame({
            'metric': ['mean', 'variance', 'std', 'min', 'max', 'count'],
            'value': [avg_score, var_score, std_score, min_score, max_score, len(scores)]
        })
        # 添加元信息
        df_stats = pd.concat([
            pd.DataFrame({'metric': ['prompt', 'image_dir'], 'value': [prompt, image_dir]}),
            df_stats
        ])
        df_stats.to_csv(stats_csv, index=False)
        
        print(f"\nResults saved to: {output_csv}")
        print(f"Statistics saved to: {stats_csv}")
        
        # 显示前几个结果
        print(f"\nTop 5 scores:")
        print(df_scores.nlargest(5, 'clip_score')[['filename', 'clip_score']])

    return avg_score, scores


def main():
    parser = argparse.ArgumentParser(description='Calculate CLIP scores for images')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text prompt for CLIP scoring')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file path')
    parser.add_argument('--model', type=str, default="openai/clip-vit-large-patch14",
                       help='CLIP model name (default: openai/clip-vit-large-patch14)')
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory '{args.image_dir}' does not exist!")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Image directory: {args.image_dir}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Output file: {args.output}")
    print(f"Model: {args.model}")
    print("-" * 60)
    
    # 计算CLIP分数
    avg_score, all_scores = compute_clip_score(
        args.image_dir,
        args.prompt, 
        args.model,
        args.output
    )
    
    print(f"\nFinal Average CLIP Score: {avg_score:.6f}")


if __name__ == "__main__":
    main()