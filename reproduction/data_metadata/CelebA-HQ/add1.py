input_path = "./CelebA-HQ-attr.txt"      # 你的原标签文件路径
output_path = "./list_attr_celeba_shift.txt"  # 修改后的新标签文件路径

with open(input_path, "r") as fin, open(output_path, "w") as fout:
    lines = fin.readlines()
    # 如果是 CelebA 官方格式，前两行是样本数和属性名，直接写回去
    fout.write(lines[0])
    fout.write(lines[1])
    # 逐行处理后面每一行
    for line in lines[2:]:
        items = line.strip().split()
        if not items:
            continue
        # 检查是不是标签行
        img_name = items[0]
        try:
            idx = int(img_name.split(".")[0])
        except Exception as e:
            print(f"跳过异常行: {line.strip()}")
            continue
        new_idx = idx + 1
        # 生成新的图片名，位数不变
        new_img_name = f"{new_idx:05d}.jpg"
        fout.write(" ".join([new_img_name] + items[1:]) + "\n")

print(f"标签编号整体+1已完成，结果保存到 {output_path}")
