#!/bin/bash

# 简单版本：移动当前目录及子目录下的 masactrl 编辑图片

TARGET_DIR="masactrl_edited_images"

echo "Moving *_masactrl_edited.png files to $TARGET_DIR/"
echo "=============================================="

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 查找并移动文件
count=0
for file in $(find . -name "*_masactrl_edited.png" -type f); do
    filename=$(basename "$file")
    echo "Moving: $file -> $TARGET_DIR/$filename"
    
    if mv "$file" "$TARGET_DIR/"; then
        echo "✓ Success"
        count=$((count + 1))
    else
        echo "✗ Failed"
    fi
done

echo ""
echo "=============================================="
echo "Moved $count files to $TARGET_DIR/"

if [ $count -gt 0 ]; then
    echo "Files in target directory:"
    ls -la "$TARGET_DIR/"
else
    echo "No *_masactrl_edited.png files found!"
fi