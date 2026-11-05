import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ConceptBalancedImageDataset(Dataset):
    def __init__(self, img_dir, attr_file, concept, transform=None, n_samples=None):
        """
        img_dir: 图片所在目录
        attr_file: CelebA属性标签文件 (如 list_attr_celeba.txt)
        concept: 关注的属性 (如 'Smiling')
        transform: torchvision.transforms
        n_samples: 每类最多采样数量 (None=全量)
        """
        self.img_dir = img_dir
        self.transform = transform
        
        # 1. 读取标签
        with open(attr_file, 'r') as f:
            lines = f.readlines()
        attr_names = lines[1].strip().split()
        concept_idx = attr_names.index(concept)
        data_lines = lines[2:]
        
        img_names, labels = [], []
        # 2. 遍历标签文件，匹配目录下实际存在的图片
        for line in data_lines:
            items = line.strip().split()
            img_name = items[0]
            attr = int(items[1 + concept_idx])
            label = 1 if attr == 1 else 0
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                img_names.append(img_name)
                labels.append(label)
        
        # 3. 正负均衡采样
        pos_idx = [i for i, l in enumerate(labels) if l == 1]
        neg_idx = [i for i, l in enumerate(labels) if l == 0]
        n = min(len(pos_idx), len(neg_idx))
        if n_samples is not None:
            n = min(n, n_samples)
        np.random.seed(42)
        pos_idx = np.random.choice(pos_idx, n, replace=False)
        neg_idx = np.random.choice(neg_idx, n, replace=False)
        all_idx = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(all_idx)
        
        self.img_names = [img_names[i] for i in all_idx]
        self.labels = [labels[i] for i in all_idx]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label, img_name
