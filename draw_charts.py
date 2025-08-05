

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
prop        = fm.FontProperties(fname=font_path, size=38, weight='bold')
prop_t       = fm.FontProperties(fname=font_path, size=28, weight='bold')
prop_legend = fm.FontProperties(fname=font_path, size=36, weight='bold')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.family'] = 'Times New Roman'

sns.set(style="whitegrid")


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

def style_axes(ax, xlabel, ylabel, title):
    """统一设置标题、坐标轴标签、刻度、边框等"""
    ax.set_title(title, fontproperties=prop, pad=8, weight='bold')
    ax.set_xlabel(xlabel, fontproperties=prop, labelpad=6, weight='bold')
    ax.set_ylabel(ylabel, fontproperties=prop, labelpad=6, weight='bold')
    ax.tick_params(axis='both', which='major',
                   labelsize=26, width=2, length=6)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(prop_t)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
# -----------------------------------------------------------------------------
# 1) 横轴 α，图例 TopK，Concept 固定
def plot_alpha_vs_epr_by_topk(concept_fixed):
    sub = df[df.concept == concept_fixed].sort_values(['scale','topk'])
    fig, ax = plt.subplots(figsize=(12,8))
    sns.lineplot(
        data=sub, x='scale', y='epr', hue='topk',
        marker='o', palette='Set1', ax=ax, linewidth=3
    )
    add_stroke(ax)
    style_axes(
        ax,
        xlabel=r"$\alpha$",
        ylabel="EPR",
        title=fr"EPR vs $\alpha$    (Concept: {concept_fixed})"
    )
    ax.margins(y=0.4)
    xt = sorted(sub.scale.unique())
    ax.set_xticks(xt)
    ax.set_xticklabels(xt, fontproperties=prop_t)
    leg = ax.legend(
        title="TopK",
        prop=prop_legend,
        title_fontproperties=prop_legend,
        loc='upper left',
        bbox_to_anchor=(1.02,1),
        borderaxespad=0.
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
    fig, ax = plt.subplots(figsize=(12,8))
    sns.lineplot(
        data=sub, x='scale', y='epr', hue='concept',
        marker='o', palette='Set2', ax=ax, linewidth=3
    )
    add_stroke(ax)
    style_axes(
        ax,
        xlabel=r"$\alpha$",
        ylabel="EPR",
        title=fr"EPR vs $\alpha$    (TopK: {topk_fixed})"
    )
    ax.margins(y=0.4)
    xt = sorted(sub.scale.unique())
    ax.set_xticks(xt)
    ax.set_xticklabels(xt, fontproperties=prop_t)
    leg = ax.legend(
        title="Concept",
        prop=prop_legend,
        title_fontproperties=prop_legend,
        loc='upper left',
        bbox_to_anchor=(1.02,1),
        borderaxespad=0.
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
    fig, ax = plt.subplots(figsize=(12,8))
    sns.lineplot(
        data=sub, x='topk', y='epr', hue='concept',
        marker='o', palette='Set2', ax=ax, linewidth=3
    )
    add_stroke(ax)
    style_axes(
        ax,
        xlabel=r"TopK",
        ylabel="EPR",
        title=fr"EPR vs TopK    ($\alpha$: {alpha_fixed})"
    )
    ax.margins(y=0.4)
    xt = sorted(sub.topk.unique())
    ax.set_xticks(xt)
    ax.set_xticklabels(xt, fontproperties=prop_t)
    leg = ax.legend(
        title="Concept",
        prop=prop_legend,
        title_fontproperties=prop_legend,
        loc='upper left',
        bbox_to_anchor=(1.02,1),
        borderaxespad=0.
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

