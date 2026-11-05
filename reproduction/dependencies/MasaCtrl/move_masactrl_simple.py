#!/usr/bin/env python3
"""
简单版本：移动 MasaCtrl 编辑后的图片
直接运行即可，会移动当前目录及子目录下的所有 *_masactrl_edited.png 文件
"""

import os
import shutil
from pathlib import Path


def main():
    # 配置
    SOURCE_DIR="/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/young"
    TARGET_DIR="/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/young1"
    PATTERN = "*_masactrl_edited.png"  # 文件模式
    
    print("🖼️  移动 MasaCtrl 编辑后的图片")
    print("=" * 40)
    
    source_path = Path(SOURCE_DIR)
    target_path = Path(TARGET_DIR)
    
    # 创建目标目录（包括所有父目录）
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ 目标目录: {target_path.absolute()}")
    
    # 查找所有匹配的文件
    found_files = list(source_path.rglob(PATTERN))
    
    print(f"🔍 找到 {len(found_files)} 个匹配的文件")
    
    if not found_files:
        print("❌ 没有找到任何 *_masactrl_edited.png 文件!")
        print("请检查当前目录是否包含这些文件。")
        return
    
    moved_count = 0
    
    # 移动文件
    for file_path in found_files:
        filename = file_path.name
        target_file = target_path / filename
        
        print(f"📁 {file_path.relative_to(source_path)}", end=" -> ")
        
        if target_file.exists():
            print("⚠️ 已存在，跳过")
            continue
        
        try:
            shutil.move(str(file_path), str(target_file))
            print("✅ 完成")
            moved_count += 1
        except Exception as e:
            print(f"❌ 失败: {e}")
    
    print("\n" + "=" * 40)
    print(f"🎉 完成! 移动了 {moved_count}/{len(found_files)} 个文件")
    
    if moved_count > 0:
        print(f"📂 文件已移动到: {target_path.absolute()}")
        
        # 显示移动的文件
        print("移动的文件:")
        for f in sorted(target_path.glob("*.png")):
            size = f.stat().st_size / 1024  # KB
            print(f"  📄 {f.name} ({size:.1f} KB)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  操作被中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()