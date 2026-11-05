

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import seaborn as sns
from glob import glob

# ========== 用户配置区 ==========
base_dir  = "./epr/resnet/sae"       # 你的 epr txt 文件夹
save_dir  = "./charts_results_pdf"   # 输出 PDF 的目录
os.makedirs(save_dir, exist_ok=True)

# 指定 Times New Roman 字体文件路径
font_path = '/u/wff7ad/.fonts/TIMES.TTF'
# ================================

# 全局字体 & seaborn 设置
prop        = fm.FontProperties(fname=font_path, size=46, weight='bold')
prop_t       = fm.FontProperties(fname=font_path, size=36, weight='bold')
prop_legend = fm.FontProperties(fname=font_path, size=42, weight='bold')
prop_xlabel  = fm.FontProperties(fname=font_path, size=46, weight='normal')  # 不加粗的xlabel
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.family'] = 'Times New Roman'

sns.set_style("ticks", {
    "axes.facecolor": "#f5f5f5",    # 灰白背景
    "axes.edgecolor": "black",      # 坐标轴黑边
    "grid.color": "white",          # 内部网格白色
    "grid.linestyle": "-", 
    "axes.grid": True,              # 开启网格
    "axes.axisbelow": True          # 网格在线下层
})



# 读取所有 txt 文件，提取 (concept, topk, scale, epr)
pattern = re.compile(r"epr_results_(.+?)_topk(\d+)_scale(\d+)\.txt")
records = []
for path in glob(os.path.join(base_dir, "epr_results_*_topk*_scale*.txt")):
    m = pattern.match(os.path.basename(path))
    if not m:
        continue
    concept, topk, scale = m.group(1), int(m.group(2)), int(m.group(3))
    # if concept =="beards":
    #     continue
    with open(path) as f:
        for L in f:
            if L.startswith("NLS:"):
                epr = float(L.split(":",1)[1])
                if scale == 0:
                    scale = 1

                records.append({
                    "concept": concept,
                    "topk":    topk,
                    "scale":   scale,
                    "epr":     epr
                })
                break

df = pd.DataFrame(records)

# -----------------------------------------------------------------------------
def add_stroke(ax):
    """为每条折线加黑色描边 + 粗线大点"""
    for line in ax.lines:
        line.set_linewidth(3)
        line.set_markersize(10)
        line.set_markeredgewidth(1.8)
        line.set_markeredgecolor('black')
        line.set_path_effects([pe.withStroke(linewidth=5, foreground='black'),
                               pe.Normal()])

# def style_axes(ax, xlabel, ylabel, title):
#     """统一设置标题、坐标轴标签、刻度、边框等"""
#     ax.set_title(title, fontproperties=prop, pad=8, weight='bold')
#     ax.set_xlabel(xlabel, fontproperties=prop, labelpad=6, weight='bold')
#     ax.set_ylabel(ylabel, fontproperties=prop, labelpad=6, weight='bold')
#     ax.tick_params(axis='both', which='major',
#                    labelsize=26, width=2, length=6)
#     for t in ax.get_xticklabels() + ax.get_yticklabels():
#         t.set_fontproperties(prop_t)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
def style_axes(ax, xlabel, ylabel, title):
    """统一设置标题、坐标轴标签、刻度、边框等"""
    ax.set_title(title, fontproperties=prop, pad=8, weight='bold')
    ax.set_xlabel(xlabel, fontproperties=prop_xlabel, labelpad=6)  # 使用不加粗字体
    ax.set_ylabel(ylabel, fontproperties=prop, labelpad=6, weight='bold')

    # 坐标轴
    ax.tick_params(axis='both', which='major',
                   labelsize=26, width=2.5, length=6, color='black')
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(prop_t)

    # 网格线：只要横线，白色微透明
    ax.yaxis.grid(True, color='#b0b0b0', linestyle='--', linewidth=2.0)
    ax.xaxis.grid(True, color='#b0b0b0', linestyle='--', linewidth=2.0)

    # 坐标轴边框加粗
    # for side in ['bottom', 'left']:
    #     ax.spines[side].set_linewidth(2.5)
    #     ax.spines[side].set_color('black')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # 轻微灰背景
    ax.set_facecolor('#f5f5f5')

# -----------------------------------------------------------------------------
# 1) 横轴 α，图例 TopK，Concept 固定
def plot_alpha_vs_epr_by_topk(concept_fixed):
    sub = df[df.concept == concept_fixed].sort_values(['scale','topk'])
    
    # 创建更清楚的 topk 标签
    sub = sub.copy()
    sub['topk_label'] = sub['topk'].apply(lambda x: f"k={x}")
    
    fig, ax = plt.subplots(figsize=(10,8))
    sns.lineplot(
        data=sub, x='scale', y='epr', hue='topk_label',
        marker='o', palette='Set1', ax=ax, linewidth=3
    )
    add_stroke(ax)
    # ax.xaxis.grid(False)
    # ax.yaxis.grid(True)
    style_axes(
        ax,
        xlabel="α",
        ylabel="EPR",
        title=""
    )
    ax.margins(y=0.4)
    xt = sorted(sub.scale.unique())
    ax.set_xticks(xt)
    ax.set_xticklabels(xt, fontproperties=prop_t)
    leg = ax.legend(
        prop=prop_legend,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        framealpha=0.6,  # 更透明
        facecolor='#fdfdfd',
        ncol=min(3, len(sub['topk_label'].unique())),  # 最多3列，自动换行
        columnspacing=0.4,  # 列间距
        handletextpad=0.2,  # 图例标记与文本间距
        handlelength=1,   # 图例标记长度
        borderpad=0.2,      # 图例内边距
        markerscale=0.8,    # 标记大小
       
        
    )
    leg.get_frame().set_linewidth(0.5)
    plt.tight_layout()
    fig.savefig(
        os.path.join(save_dir,
                     f"alpha_vs_epr_topk_concept_{concept_fixed}.pdf"),
        bbox_inches='tight'
    )
    plt.close(fig)

# 2) 横轴 α，图例 Concept，TopK 固定
def plot_alpha_vs_epr_by_concept(topk_fixed):
    sub = df[df.topk == topk_fixed].sort_values(['scale','concept'])
    fig, ax = plt.subplots(figsize=(10,8))
    sns.lineplot(
        data=sub, x='scale', y='epr', hue='concept',
        marker='o', palette='Set2', ax=ax, linewidth=3
    )
    add_stroke(ax)
    # ax.xaxis.grid(False)
    # ax.yaxis.grid(True)
    style_axes(
        ax,
        xlabel="α",
        ylabel="EPR",
        title=""
    )
    ax.margins(y=0.4)
    xt = sorted(sub.scale.unique())
    ax.set_xticks(xt)
    ax.set_xticklabels(xt, fontproperties=prop_t)
    leg = ax.legend(
        prop=prop_legend,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        framealpha=0.6,  # 更透明
        facecolor='#fdfdfd',
        ncol=min(3, len(sub['concept'].unique())),  # 最多3列，自动换行
        columnspacing=0.2,  # 列间距
        handletextpad=0.15,  # 图例标记与文本间距
        handlelength=0.7,   # 图例标记长度
        borderpad=0.1,      # 图例内边距
        markerscale=0.7,    # 标记大小
       
        
    )
    leg.get_frame().set_linewidth(0.5)
    plt.tight_layout()
    fig.savefig(
        os.path.join(save_dir,
                     f"alpha_vs_epr_concept_topk_{topk_fixed}.pdf"),
        bbox_inches='tight'
    )
    plt.close(fig)

# 3) 横轴 TopK，图例 Concept，α 固定
def plot_topk_vs_epr_by_concept(alpha_fixed):
    sub = df[df.scale == alpha_fixed].sort_values(['topk','concept'])
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 将 topk 转换为字符串，确保等间距显示
    sub = sub.copy()
    sub['topk_str'] = sub['topk'].astype(str)
    
    sns.lineplot(
        data=sub, x='topk_str', y='epr', hue='concept',
        marker='o', palette='Set2', ax=ax, linewidth=3
    )
    add_stroke(ax)
    # ax.xaxis.grid(False)
    # ax.yaxis.grid(True)
    style_axes(
        ax,
        xlabel=r"TopK",
        ylabel="EPR",
        title=""
    )
    ax.margins(y=0.4)
    
    # 设置 TopK 轴为等间距
    xt = sorted(sub.topk.unique())
    xt_str = [str(x) for x in xt]
    ax.set_xticks(range(len(xt_str)))
    ax.set_xticklabels(xt_str, fontproperties=prop_t)
    
    leg = ax.legend(
        prop=prop_legend,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        framealpha=0.6,  # 更透明
        facecolor='#fdfdfd',
        ncol=min(3, len(sub['concept'].unique())),  # 最多3列，自动换行
        columnspacing=0.2,  # 列间距
        handletextpad=0.15,  # 图例标记与文本间距
        handlelength=0.7,   # 图例标记长度
        borderpad=0.1,      # 图例内边距
        markerscale=0.7,    # 标记大小

    )
    leg.get_frame().set_linewidth(0.5)
    plt.tight_layout()
    fig.savefig(
        os.path.join(save_dir,
                     f"topk_vs_epr_concept_alpha_{alpha_fixed}.pdf"),
        bbox_inches='tight'
    )
    plt.close(fig)


# === 批量生成示例 ===
for c in df['concept'].unique():
    plot_alpha_vs_epr_by_topk(c)
for t in sorted(df['topk'].unique()):
    plot_alpha_vs_epr_by_concept(t)
for s in sorted(df['scale'].unique()):
    plot_topk_vs_epr_by_concept(s)

print("所有 PDF 图表已保存至:", save_dir)

