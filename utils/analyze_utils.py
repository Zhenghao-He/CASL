import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_soft_mask_distribution_save(
    soft_mask,
    bins=50,
    threshold=0.5,
    stats_file='soft_mask_stats.txt',
    img_dir='soft_mask_dist.png',
    show_kde=True
):
    """
    可视化分析soft_mask分布，统计数据append保存到文件，图片保存到本地。

    参数:
        soft_mask: torch.Tensor or np.ndarray
        bins: int, 直方图分桶数
        threshold: float, 判定激活的阈值
        stats_file: str, 统计数据存放的文件路径
        img_file: str, 图片保存路径
        show_kde: bool, 是否画密度曲线（需要seaborn）
    """
    # 数据转换
    if isinstance(soft_mask, torch.Tensor):
        data = soft_mask.detach().cpu().numpy().flatten()
    else:
        data = np.array(soft_mask).flatten()

    # 画图并保存
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(data, bins=bins)
    plt.xlabel('soft_mask value')
    plt.ylabel('Count')
    plt.title('Histogram')

    if show_kde:
        try:
            import seaborn as sns
            plt.subplot(1, 2, 2)
            sns.kdeplot(data)
            plt.xlabel('soft_mask value')
            plt.ylabel('Density')
            plt.title('KDE')
        except ImportError:
            plt.subplot(1, 2, 2)
            plt.hist(data, bins=bins, density=True)
            plt.title('Density (hist)')
            plt.xlabel('soft_mask value')
            plt.ylabel('Density')

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'soft_mask_distribution.png'), dpi=200)  # 保存图片
    plt.close()

    # importance 是你的 tensor
    importance_sorted, _ = torch.sort(soft_mask, descending=True)

    # 1. 排序后的数值曲线
    plt.figure(figsize=(6,4))
    plt.plot(importance_sorted.cpu().numpy(), marker='o', linewidth=1)
    plt.xlabel('Sorted index (Top-K)')
    plt.ylabel('Importance value')
    plt.title('Elbow Method for Top-K Selection')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'importance_sorted_curve.png'), dpi=200)   # 保存图片
    plt.close()

    # 2. 累积占比曲线
    cumsum = torch.cumsum(importance_sorted, dim=0)
    cumsum_ratio = cumsum / importance_sorted.sum()
    plt.figure(figsize=(6,4))
    plt.plot(cumsum_ratio.cpu().numpy(), marker='.')
    plt.xlabel('Top-K')
    plt.ylabel('Cumulative ratio')
    plt.title('Cumulative Sum Ratio of Importance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'importance_cumsum_ratio.png'), dpi=200)   # 保存图片
    plt.close()


    # 统计信息
    stat_lines = []
    stat_lines.append(f"=== Soft Mask Analysis ===")
    stat_lines.append(f"min: {data.min():.6f}")
    stat_lines.append(f"max: {data.max():.6f}")
    stat_lines.append(f"mean: {data.mean():.6f}")
    stat_lines.append(f"median: {np.median(data):.6f}")
    stat_lines.append(f"std: {data.std():.6f}")
    stat_lines.append(f">{threshold} ratio: {(data > threshold).mean():.6f}")
    stat_lines.append(f"10% quantile: {np.percentile(data, 10):.6f}")
    stat_lines.append(f"90% quantile: {np.percentile(data, 90):.6f}")
    stat_lines.append(f"")

    # 追加写入文件
    with open(stats_file, 'a') as f:
        f.write('\n'.join(stat_lines) + '\n')

