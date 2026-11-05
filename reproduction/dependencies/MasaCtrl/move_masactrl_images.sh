#!/bin/bash

# 移动 MasaCtrl 编辑后的图片文件
# Usage: ./move_masactrl_images.sh [source_dir] [target_dir]

# 设置默认路径
DEFAULT_SOURCE="/p/realai/zhenghao/LoRA_SAE/MasaCtrl/workdir/celeba/beards"
DEFAULT_TARGET="/p/realai/zhenghao/LoRA_SAE/workdir/celeba/beards1"

# 获取命令行参数或使用默认值
SOURCE_DIR="${1:-$DEFAULT_SOURCE}"
TARGET_DIR="${2:-$DEFAULT_TARGET}"

echo "Moving MasaCtrl edited images..."
echo "Source directory: $SOURCE_DIR"
echo "Target directory: $TARGET_DIR"
echo "=================================="

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 计数器
moved_count=0
found_count=0

# 查找并移动文件
echo "Searching for *_masactrl_edited.png files..."

# 使用 find 递归查找所有匹配的文件
while IFS= read -r -d '' file; do
    found_count=$((found_count + 1))
    filename=$(basename "$file")
    
    echo "Found: $file"
    
    # 检查目标文件是否已存在
    if [ -f "$TARGET_DIR/$filename" ]; then
        echo "  Warning: $filename already exists in target directory"
        echo "  Skipping to avoid overwrite..."
    else
        # 移动文件
        if mv "$file" "$TARGET_DIR/"; then
            echo "  ✓ Moved to: $TARGET_DIR/$filename"
            moved_count=$((moved_count + 1))
        else
            echo "  ✗ Error moving file: $file"
        fi
    fi
    echo ""
done < <(find "$SOURCE_DIR" -name "*_masactrl_edited.png" -type f -print0)

# 总结
echo "=================================="
echo "Search completed!"
echo "Found: $found_count files"
echo "Moved: $moved_count files"
echo "Target directory: $TARGET_DIR"

if [ $found_count -eq 0 ]; then
    echo ""
    echo "No *_masactrl_edited.png files found in $SOURCE_DIR"
    echo "Please check if:"
    echo "  1. The source directory is correct"
    echo "  2. The files exist with the correct naming pattern"
    echo "  3. You have read permissions for the source directory"
elif [ $moved_count -gt 0 ]; then
    echo ""
    echo "Files successfully moved to: $TARGET_DIR"
    echo "Contents of target directory:"
    ls -la "$TARGET_DIR"
fi