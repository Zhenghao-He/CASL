#!/usr/bin/env python3
"""
LPIPS Calculator for Image Pairs

This script calculates LPIPS (Learned Perceptual Image Patch Similarity) scores 
between original images and their edited versions.

Usage:
    python calculate_lpips.py --original_dir /path/to/original --edited_dir /path/to/edited --output output.csv
"""

import os
import re
import argparse
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms
import lpips
from tqdm import tqdm


def setup_image_processor(imsize=512):
    """Setup image preprocessing pipeline"""
    return transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Convert to [-1, 1]
    ])


def load_image(image_path, processor):
    """Load and preprocess a single image"""
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = processor(image).unsqueeze(0)  # Add batch dimension
        return tensor
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def extract_number_from_filename(filename):
    """Extract 5-digit number from filename"""
    # Look for exactly 5 consecutive digits in the filename
    match = re.search(r'(\d{5})', filename)
    return match.group(1) if match else None


def find_matching_files(original_dir, edited_dir):
    """Find matching image pairs between two directories"""
    # Get all image files from both directories
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


def calculate_lpips_scores(original_dir, edited_dir, output_csv, imsize=512, net='alex'):
    """Calculate LPIPS scores for matching image pairs"""
    
    # Initialize LPIPS model
    print(f"Initializing LPIPS model with {net} network...")
    loss_fn = lpips.LPIPS(net=net)
    
    # Setup image processor
    processor = setup_image_processor(imsize)
    
    # Find matching files
    print("Finding matching image pairs...")
    matches = find_matching_files(original_dir, edited_dir)
    
    if not matches:
        print("No matching image pairs found!")
        return
    
    print(f"Found {len(matches)} matching image pairs")
    
    # Calculate LPIPS scores
    results = []
    
    for number, original_file, edited_file in tqdm(matches, desc="Calculating LPIPS"):
        original_path = os.path.join(original_dir, original_file)
        edited_path = os.path.join(edited_dir, edited_file)
        
        # Load images
        original_tensor = load_image(original_path, processor)
        edited_tensor = load_image(edited_path, processor)
        
        if original_tensor is None or edited_tensor is None:
            print(f"Skipping {number} due to loading error")
            continue
        
        try:
            # Calculate LPIPS
            with torch.no_grad():
                lpips_score = loss_fn(original_tensor, edited_tensor).item()
            
            results.append({
                'filename': number,
                'lpips': lpips_score,
                'original_file': original_file,
                'edited_file': edited_file
            })
            
        except Exception as e:
            print(f"Error calculating LPIPS for {number}: {e}")
            continue
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        # Sort by filename for better readability
        df = df.sort_values('filename')
        
        # Calculate statistics
        mean_lpips = df['lpips'].mean()
        std_lpips = df['lpips'].std()
        var_lpips = df['lpips'].var()
        min_lpips = df['lpips'].min()
        max_lpips = df['lpips'].max()
        
        # Save with only required columns
        output_df = df[['filename', 'lpips']].copy()
        output_df.to_csv(output_csv, index=False)
        
        # Save detailed statistics to a separate file
        stats_file = output_csv.replace('.csv', '_statistics.csv')
        stats_df = pd.DataFrame({
            'metric': ['mean', 'std', 'variance', 'min', 'max', 'count'],
            'value': [mean_lpips, std_lpips, var_lpips, min_lpips, max_lpips, len(results)]
        })
        stats_df.to_csv(stats_file, index=False)
        
        print(f"\nResults saved to {output_csv}")
        print(f"Statistics saved to {stats_file}")
        print(f"Processed {len(results)} image pairs")
        print(f"LPIPS Statistics:")
        print(f"  Mean: {mean_lpips:.6f}")
        print(f"  Standard Deviation: {std_lpips:.6f}")
        print(f"  Variance: {var_lpips:.6f}")
        print(f"  Min: {min_lpips:.6f}")
        print(f"  Max: {max_lpips:.6f}")
        
        # Display first few results
        print(f"\nFirst 5 results:")
        print(output_df.head())
        
    else:
        print("No valid results to save!")


def main():
    parser = argparse.ArgumentParser(description='Calculate LPIPS scores between original and edited images')
    parser.add_argument('--original_dir', type=str, required=True,
                       help='Directory containing original images')
    parser.add_argument('--edited_dir', type=str, required=True,
                       help='Directory containing edited images')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file path')
    parser.add_argument('--imsize', type=int, default=512,
                       help='Image size for processing (default: 512)')
    parser.add_argument('--net', type=str, default='alex', choices=['alex', 'vgg'],
                       help='Network for LPIPS calculation (default: alex)')
    
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
    print(f"Image size: {args.imsize}")
    print(f"LPIPS network: {args.net}")
    print("-" * 50)
    
    # Calculate LPIPS scores
    calculate_lpips_scores(
        args.original_dir,
        args.edited_dir, 
        args.output,
        args.imsize,
        args.net
    )


if __name__ == '__main__':
    main()