import os
import shutil

from PIL import Image


boundary_dir = "results_500range"
num = 7
for result_dir in os.listdir(boundary_dir):
    full_result_path = os.path.join(boundary_dir, result_dir)
    if not os.path.isdir(full_result_path):
        continue

    # 解析新文件夹名，如 Big_Nose_0
    result_name = result_dir.replace("results_", "")
    save_dir = f"{result_name}_{num}"
    os.makedirs(os.path.join("boundary", save_dir), exist_ok=True)

    for subfolder in os.listdir(full_result_path):
        subfolder_path = os.path.join(full_result_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        src_path = os.path.join(subfolder_path, f"edited_{num}.png")

        if os.path.exists(src_path):
            dst_path = os.path.join("boundary_good", save_dir, "images",f"{subfolder}.jpg")
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            # 正确转换格式
            img = Image.open(src_path).convert("RGB")  # PNG 可能含 alpha，JPG 不支持
            img.save(dst_path, "JPEG")

            print(f"Converted and saved: {src_path} -> {dst_path}")
        else:
            print(f"Warning: {src_path} not found.")

