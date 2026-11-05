#!/usr/bin/env python3
"""
移动 MasaCtrl 编辑后的图片文件
移动所有匹配 *_masactrl_edited.png 的文件到指定目录
"""

import os
import shutil
import glob
import argparse
from pathlib import Path


def move_masactrl_images(source_dir, target_dir, pattern="*_masactrl_edited.png"):
    """
    移动匹配模式的图片文件
    
    Args:
        source_dir (str): 源目录路径
        target_dir (str): 目标目录路径  
        pattern (str): 文件名模式
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    print(f"移动 MasaCtrl 编辑后的图片...")
    print(f"源目录: {source_path.absolute()}")
    print(f"目标目录: {target_path.absolute()}")
    print(f"匹配模式: {pattern}")
    print("=" * 50)
    
    # 创建目标目录
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ 目标目录已创建: {target_path}")
    
    # 递归查找所有匹配的文件
    found_files = []
    
    # 使用 rglob 递归搜索
    for file_path in source_path.rglob(pattern):
        if file_path.is_file():
            found_files.append(file_path)
    
    print(f"\n找到 {len(found_files)} 个匹配的文件:")
    
    if not found_files:
        print("没有找到匹配的文件!")
        return 0
    
    moved_count = 0
    skipped_count = 0
    
    for file_path in found_files:
        filename = file_path.name
        target_file = target_path / filename
        
        print(f"\n处理: {file_path.relative_to(source_path)}")
        
        # 检查目标文件是否已存在
        if target_file.exists():
            print(f"  ⚠️  文件已存在，跳过: {filename}")
            skipped_count += 1
            continue
        
        try:
            # 移动文件
            shutil.move(str(file_path), str(target_file))
            print(f"  ✓ 已移动到: {target_file.name}")
            moved_count += 1
        except Exception as e:
            print(f"  ✗ 移动失败: {e}")
    
    # 显示结果统计
    print("\n" + "=" * 50)
    print("移动完成!")
    print(f"找到文件: {len(found_files)}")
    print(f"成功移动: {moved_count}")
    print(f"跳过文件: {skipped_count}")
    
    if moved_count > 0:
        print(f"\n目标目录内容:")
        try:
            files_in_target = list(target_path.glob("*"))
            for f in sorted(files_in_target):
                if f.is_file():
                    size = f.stat().st_size
                    print(f"  {f.name} ({size:,} bytes)")
        except Exception as e:
            print(f"  无法列出目标目录内容: {e}")
    
    return moved_count


def main():
    parser = argparse.ArgumentParser(
        description="移动 MasaCtrl 编辑后的图片文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python move_masactrl.py                                    # 当前目录到 ./masactrl_images/
  python move_masactrl.py --source /path/to/source          # 指定源目录
  python move_masactrl.py --source /path/to/source --target /path/to/target  # 指定源和目标
  python move_masactrl.py --pattern "*_edited.png"          # 自定义文件模式
        """
    )
    
    parser.add_argument(
        "--source", "-s",
        default=".",
        help="源目录路径 (默认: 当前目录)"
    )
    
    parser.add_argument(
        "--target", "-t", 
        default="./masactrl_images",
        help="目标目录路径 (默认: ./masactrl_images)"
    )
    
    parser.add_argument(
        "--pattern", "-p",
        default="*_masactrl_edited.png", 
        help="文件名模式 (默认: *_masactrl_edited.png)"
    )
    
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="试运行，只显示会移动的文件，不实际移动"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 试运行模式 - 只显示会移动的文件")
        source_path = Path(args.source)
        found_files = list(source_path.rglob(args.pattern))
        
        print(f"源目录: {source_path.absolute()}")
        print(f"目标目录: {Path(args.target).absolute()}")
        print(f"匹配模式: {args.pattern}")
        print(f"找到 {len(found_files)} 个文件:")
        
        for file_path in found_files:
            print(f"  {file_path}")
        
        return
    
    try:
        moved_count = move_masactrl_images(args.source, args.target, args.pattern)
        print(f"\n🎉 任务完成! 成功移动了 {moved_count} 个文件")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  操作被用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()