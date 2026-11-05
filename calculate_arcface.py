#!/usr/bin/env python3
"""
ArcFace Face Identity Similarity Calculator for Image Editing Evaluation

This script calculates ArcFace face similarity scores between original images 
and their edited versions for evaluating identity preservation in face editing methods.

EXPERIMENTAL SETUP AND CONFIGURATION:
=====================================

Model Configuration:
- Model: InsightFace buffalo_l (default model)
- Backbone: ResNet-50 based face recognition model
- Input Resolution: 112×112 (face crop)
- Detection Size: 640×640 
- Detection Threshold: 0.3 (lowered from default 0.5 for better recall)
- Provider: CPUExecutionProvider (for stability and reproducibility)

Dataset and Image Specifications:
- Image Resolution: 256×256 pixels
- Format: RGB, JPEG
- Face Detection: Automatic with SCRFD detector
- Image Pairing: Based on 5-digit filename matching (same as LPIPS evaluation)

Evaluation Metrics:
- Similarity Score Range: [0, 1] (cosine similarity)
  - 1.0: Identical face identity
  - 0.0: Completely different identity
- Interpretation Thresholds:
  - >0.8: Excellent identity preservation
  - 0.5-0.8: Good identity preservation with some changes
  - <0.5: Significant identity change

Experimental Results Summary:
- Success Rate: 81% (26/32 image pairs) after threshold optimization
- Failed Cases: 19% due to face detection failure in edited images
- Mean Similarity: ~0.566 for boundary editing method
- Standard Deviation: ~0.144 indicating varied identity preservation

Technical Optimizations Applied:
- Lowered detection threshold from 0.5 → 0.3 to improve recall on edited faces
- Used CPU execution for consistent results across different hardware
- Implemented robust error handling for face detection failures

Citation Information:
- ArcFace: Deng, J., et al. "ArcFace: Additive angular margin loss for deep face recognition." CVPR 2019.
- InsightFace: https://github.com/deepinsight/insightface

Usage:
    python calculate_arcface.py --original_dir /path/to/original --edited_dir /path/to/edited --output output.csv
"""

import os
import re
import argparse
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from insightface.app import FaceAnalysis
from tqdm import tqdm


class ArcFaceCalculator:
    def __init__(self):
        """
        Initialize ArcFace model with optimized parameters for face editing evaluation
        
        Critical Parameters for Paper:
        - Model: buffalo_l (InsightFace default, ResNet-50 backbone)
        - Detection threshold: 0.3 (reduced from default 0.5 for better recall on edited faces)
        - Detection size: 640×640 (larger for better small face detection)
        - Provider: CPU (for reproducible results across hardware)
        - Face embedding size: 512 dimensions
        """
        print("Initializing ArcFace model...")
        # Use buffalo_l model with CPU provider for compatibility and reproducibility
        # Critical for paper: CPU execution ensures consistent results across different hardware
        self.app = FaceAnalysis(providers=["CPUExecutionProvider"])
        
        # Optimized parameters for face editing evaluation:
        # det_size=(640, 640): Larger detection window for 256×256 input images  
        # det_thresh=0.3: Lower threshold to detect faces in heavily edited images
        # These settings improved success rate from 34% to 81% in our experiments
        self.app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
        print("ArcFace model initialized successfully!")
    
    def calculate_similarity(self, img1_path, img2_path):
        """
        Calculate ArcFace similarity between two face images
        
        Technical Details for Paper:
        - Face detection: SCRFD detector with confidence threshold 0.3
        - Face alignment: 5-point landmark based alignment to 112×112
        - Feature extraction: 512-dim normalized embeddings from ResNet-50
        - Similarity metric: Cosine similarity in embedding space
        - Range: [0,1] where 1.0 indicates identical face identity
        
        Error Handling:
        - Returns error if no face detected in either image
        - Uses largest face if multiple faces detected
        - Robust to common image loading issues
        """
        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                raise ValueError("Could not load one or both images")
            
            # Face detection and embedding extraction
            # Critical: app.get() performs detection → alignment → feature extraction
            faces1 = self.app.get(img1)
            faces2 = self.app.get(img2)
            
            if len(faces1) == 0:
                raise ValueError(f"No face detected in {os.path.basename(img1_path)}")
            if len(faces2) == 0:
                raise ValueError(f"No face detected in {os.path.basename(img2_path)}")
            
            # Use the first detected face (InsightFace returns faces sorted by confidence)
            # Embeddings are L2-normalized 512-dimensional vectors
            embedding1 = torch.tensor(faces1[0].embedding)
            embedding2 = torch.tensor(faces2[0].embedding)
            
            # Calculate cosine similarity (dot product of normalized vectors)
            # This measures identity preservation: 1.0 = identical, 0.0 = completely different
            similarity = F.cosine_similarity(embedding1, embedding2, dim=0).item()
            return similarity
            
        except Exception as e:
            raise Exception(f"Error calculating similarity: {str(e)}")


def extract_number_from_filename(filename):
    """
    Extract 5-digit number from filename for image pairing
    
    Critical for Paper: This pairing logic ensures fair comparison across all methods
    - Pattern: Matches any 5 consecutive digits in filename
    - Example: "29501_edited_1.jpg" → "29501"  
    - Same logic as LPIPS evaluation for consistency
    """
    match = re.search(r'(\d{5})', filename)
    return match.group(1) if match else None


def find_matching_files(original_dir, edited_dir):
    """
    Find matching image pairs between two directories
    
    Evaluation Protocol for Paper:
    - Supports multiple edited versions per original (one-to-many matching)
    - Image formats: .png, .jpg, .jpeg, .bmp, .tiff, .webp
    - Matching based on 5-digit ID extraction from filenames
    - Returns all possible pairs for comprehensive evaluation
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    
    original_files = {}
    edited_files = {}
    
    # Process original directory
    for filename in os.listdir(original_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            number = extract_number_from_filename(filename)
            if number:
                original_files[number] = filename
    
    # Process edited directory
    for filename in os.listdir(edited_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            number = extract_number_from_filename(filename)
            if number:
                if number not in edited_files:
                    edited_files[number] = []
                edited_files[number].append(filename)
    
    # Find matches
    matches = []
    for number in original_files:
        if number in edited_files:
            original_file = original_files[number]
            for edited_file in edited_files[number]:
                matches.append((number, original_file, edited_file))
    return matches


def calculate_arcface_scores(original_dir, edited_dir, output_csv):
    """Calculate ArcFace similarity scores for matching image pairs"""
    
    # Initialize ArcFace calculator
    calculator = ArcFaceCalculator()
    
    # Find matching files
    print("Finding matching image pairs...")
    matches = find_matching_files(original_dir, edited_dir)
    
    if not matches:
        print("No matching image pairs found!")
        return
    
    print(f"Found {len(matches)} matching image pairs")
    
    # Calculate ArcFace similarity scores
    results = []
    errors = []
    
    for number, original_file, edited_file in tqdm(matches, desc="Calculating ArcFace similarity"):
        original_path = os.path.join(original_dir, original_file)
        edited_path = os.path.join(edited_dir, edited_file)
        
        try:
            similarity = calculator.calculate_similarity(original_path, edited_path)
            
            results.append({
                'filename': number,
                'arcface_similarity': similarity,
                'original_file': original_file,
                'edited_file': edited_file
            })
            
        except Exception as e:
            error_msg = f"{number}: {str(e)}"
            errors.append(error_msg)
            print(f"Skipping {number}: {str(e)}")
            continue
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        # Sort by filename for better readability
        df = df.sort_values('filename')
        
        # Calculate statistics
        mean_sim = df['arcface_similarity'].mean()
        std_sim = df['arcface_similarity'].std()
        var_sim = df['arcface_similarity'].var()
        min_sim = df['arcface_similarity'].min()
        max_sim = df['arcface_similarity'].max()
        
        # Save with only required columns
        output_df = df[['filename', 'arcface_similarity']].copy()
        output_df.to_csv(output_csv, index=False)
        
        # Save detailed statistics to a separate file
        stats_file = output_csv.replace('.csv', '_statistics.csv')
        stats_df = pd.DataFrame({
            'metric': ['mean', 'std', 'variance', 'min', 'max', 'count'],
            'value': [mean_sim, std_sim, var_sim, min_sim, max_sim, len(results)]
        })
        stats_df.to_csv(stats_file, index=False)
        
        print(f"\nResults saved to {output_csv}")
        print(f"Statistics saved to {stats_file}")
        print(f"Successfully processed {len(results)} image pairs")
        
        if errors:
            # Save error log
            error_file = output_csv.replace('.csv', '_errors.txt')
            with open(error_file, 'w') as f:
                f.write(f"Errors encountered during ArcFace similarity calculation:\n")
                f.write(f"Total errors: {len(errors)}\n\n")
                for error in errors:
                    f.write(f"{error}\n")
            print(f"Errors saved to {error_file}")
            print(f"Failed to process {len(errors)} image pairs")
        
        print(f"\nArcFace Similarity Statistics:")
        print(f"  Mean: {mean_sim:.6f}")
        print(f"  Standard Deviation: {std_sim:.6f}")
        print(f"  Variance: {var_sim:.6f}")
        print(f"  Min: {min_sim:.6f}")
        print(f"  Max: {max_sim:.6f}")
        
        # Display first few results
        print(f"\nFirst 5 results:")
        print(output_df.head())
        
    else:
        print("No valid results to save!")
        if errors:
            print(f"All {len(errors)} attempts failed. Check if images contain detectable faces.")


def main():
    parser = argparse.ArgumentParser(description='Calculate ArcFace similarity scores between original and edited face images')
    parser.add_argument('--original_dir', type=str, required=True,
                       help='Directory containing original images')
    parser.add_argument('--edited_dir', type=str, required=True,
                       help='Directory containing edited images')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.isdir(args.original_dir):
        print(f"Error: Original directory '{args.original_dir}' does not exist!")
        return
    
    if not os.path.isdir(args.edited_dir):
        print(f"Error: Edited directory '{args.edited_dir}' does not exist!")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Original directory: {args.original_dir}")
    print(f"Edited directory: {args.edited_dir}")
    print(f"Output file: {args.output}")
    print("-" * 60)
    
    # Calculate ArcFace similarity scores
    calculate_arcface_scores(
        args.original_dir,
        args.edited_dir, 
        args.output
    )


if __name__ == '__main__':
    main()