# 路径和文件名配置
mapping_file = './image_list.txt'  # 你的映射表
attr_file = './list_attr_celeba.txt'      # 原始属性标签文件
output_file = './CelebA-HQ-attr.txt'      # 输出文件

# 1. 读取mapping表：HQ idx -> 原始图片名
hq2orig = {}
with open(mapping_file, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:  # 跳过表头
        parts = line.strip().split()
        hq_idx = int(parts[0])
        orig_file = parts[2]
        hq2orig[hq_idx] = orig_file

# 2. 读取原始属性标签
attr_dict = {}
with open(attr_file, 'r') as f:
    lines = f.readlines()
    # 头两行是属性数和属性名，数据从第3行起
    for line in lines[2:]:
        parts = line.strip().split()
        fname = parts[0]
        attrs = parts[1:]
        attr_dict[fname] = attrs

# 3. 输出CelebA-HQ的属性文件
with open(output_file, 'w') as fout:
    # 写属性名
    fout.write(" ".join(lines[1].strip().split()) + "\n")
    # 写每一张HQ图像的属性
    for hq_idx in range(len(hq2orig)):
        orig_fname = hq2orig[hq_idx]
        attrs = attr_dict.get(orig_fname)
        if attrs is None:
            print(f"Warning: {orig_fname} not found in attr file")
            continue
        # HQ图片名格式：%05d.jpg 或 %06d.jpg，可自行调整
        hq_fname = f"{hq_idx:05d}.jpg"
        fout.write(hq_fname + " " + " ".join(attrs) + "\n")

print("Done! CelebA-HQ attr file saved to:", output_file)
