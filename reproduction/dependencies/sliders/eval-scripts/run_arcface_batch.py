#!/usr/bin/env python3
"""
ArcFace Batch Processing Script
批量计算多个方法的ArcFace相似度分数 - 所有方法使用相同的原始图片
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    print("🔄 Starting ArcFace batch processing...")
    print("=" * 60)
    
    # 切换到正确的目录
    script_dir = Path("/p/realai/zhenghao/LoRA_SAE/sliders/eval-scripts")
    os.chdir(script_dir)
    
    # 固定的原始图片目录
    ORIGINAL_DIR = "/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0/images"
    
    # 定义不同方法的编辑后图片目录和提示词（与LPIPS脚本保持一致）
    arcface_configs = [
        # Boundary 方法
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/boundary_good/Smiling_500Range_7/images", "a smiling face", "boundary"),
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/boundary_good/Big_Nose_500Range_7/images", "face with a big nose", "boundary"),
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/boundary_good/Young_500Range_7/images", "a young face", "boundary"),
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/boundary_good/Mustache_500Range_7/images", "a face with beards", "boundary"),
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/boundary_good/Blond_Hair_500Range_7/images", "people with blond hair", "boundary"),
        
        # Asyrp 方法
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_official/runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images", "a young face", "asyrp"),
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_official/runs/smiling_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images", "a smiling face", "asyrp"),
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_official/runs/big_nose_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images", "face with a big nose", "asyrp"),
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_official/runs/blond_hair_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images", "people with blond hair", "asyrp"),
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_official/runs/beards_LC_CelebA_HQ_P2_t999_ninv50_ngen50/test_images/100/iter4/delta_h/images", "a face with beards", "asyrp"),
        
        # Ours 方法
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/delta_h/images", "a young face", "ours"),
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/smiling_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/delta_h/images", "a smiling face", "ours"),
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/big_nose_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/delta_h/images", "face with a big nose", "ours"),
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/blond_hair_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/delta_h/images", "people with blond hair", "ours"),
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/beards_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/delta_h/images", "a face with beards", "ours"),
        
        # Reconstruction 方法
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/young_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0/images", "a young face", "reconstruction"),
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/smiling_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0/images", "a smiling face", "reconstruction"),
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/big_nose_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0/images", "face with a big nose", "reconstruction"),
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/blond_hair_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0/images", "people with blond hair", "reconstruction"),
        ("/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/runs/beards_LC_CelebA_HQ_P2_t999_ninv50_ngen50/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse/xblock_test_images/100/iter4_ngen50/x0/images", "a face with beards", "reconstruction"),
        
        # MasaCtrl 方法
        ("/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/young1", "a young face", "masactrl"),
        ("/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/smiling1", "a smiling face", "masactrl"),
        ("/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/big_nose1", "face with a big nose", "masactrl"),
        ("/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/beards1", "a face with beards", "masactrl"),
        ("/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/blonde_hair1", "people with blond hair", "masactrl"),
    ]
    
    # 创建结果目录
    results_dir = Path("arcface_results")
    results_dir.mkdir(exist_ok=True)
    
    print(f"📁 Original images directory: {ORIGINAL_DIR}")
    print(f"💾 Results will be saved to: {results_dir.absolute()}")
    
    success_count = 0
    total_count = len(arcface_configs)
    
    # 处理每个配置
    for i, (edited_dir, prompt, method) in enumerate(arcface_configs, 1):
        # 从prompt生成概念名（与CLIP脚本保持一致）
        if prompt == "a young face":
            concept = "Young"
        elif prompt == "a smiling face":
            concept = "Smiling"
        elif prompt == "face with a big nose":
            concept = "Big_Nose"
        elif prompt == "a face with beards":
            concept = "Mustache"
        elif prompt == "people with blond hair":
            concept = "Blond_Hair"
        else:
            concept = "Unknown"
        
        print(f"\n👥 Processing {i}/{total_count}: {method} - {concept}")
        print(f"Original: {ORIGINAL_DIR}")
        print(f"Edited: {edited_dir}")
        
        # 创建方法特定的子目录
        method_dir = results_dir / method
        method_dir.mkdir(exist_ok=True)
        output_file = method_dir / f"arcface_{concept}.csv"
        
        print(f"Output: {output_file}")
        print("-" * 70)
        
        # 检查原始目录是否存在
        if not Path(ORIGINAL_DIR).exists():
            print(f"❌ Original directory not found: {ORIGINAL_DIR}")
            continue
            
        # 检查编辑后目录是否存在
        if not Path(edited_dir).exists():
            print(f"❌ Edited directory not found: {edited_dir}")
            continue
        
        # 构建命令
        cmd = [
            "python", "calculate_arcface.py",
            "--original_dir", ORIGINAL_DIR,
            "--edited_dir", edited_dir, 
            "--output", str(output_file)
        ]
        
        try:
            # 运行 ArcFace 计算
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)  # 20分钟超时
            
            if result.returncode == 0:
                print("✅ Successfully processed")
                success_count += 1
                
                # 显示部分输出
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-5:]:  # 显示最后5行
                        print(f"  {line}")
            else:
                print("❌ Error occurred")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            print("❌ Process timed out (20 minutes)")
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    # 显示总结
    print("\n" + "=" * 60)
    print("🎉 ArcFace batch processing completed!")
    print(f"✅ Successful: {success_count}/{total_count}")
    print(f"📁 Results saved in: {results_dir.absolute()}")
    
    # 显示生成的文件
    print("\n📄 Generated files:")
    csv_files = list(results_dir.rglob("*.csv"))
    for f in sorted(csv_files):
        if f.name.endswith('_statistics.csv') or f.name.endswith('_errors.txt'):
            continue
        size = f.stat().st_size / 1024  # KB
        print(f"  {f.relative_to(results_dir)} ({size:.1f} KB)")
    
    # 显示目录结构
    print(f"\n📂 Directory structure:")
    for method_dir in sorted(results_dir.iterdir()):
        if method_dir.is_dir():
            print(f"  {method_dir.name}/")
            for csv_file in sorted(method_dir.glob("*.csv")):
                if not csv_file.name.endswith('_statistics.csv'):
                    print(f"    ├── {csv_file.name}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()