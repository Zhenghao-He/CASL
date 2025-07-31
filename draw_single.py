import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import seaborn as sns
from glob import glob

# ==================== 配置 ======================
base_dir = "./epr/resnet/sae"
save_dir = "./charts_results_pdf/single"
os.makedirs(save_dir, exist_ok=True)
font_path = "/u/wff7ad/.fonts/TIMES.TTF"
prop = fm.FontProperties(fname=font_path, size=48, weight='bold')
prop_legend = fm.FontProperties(fname=font_path, size=26, weight='bold')
prop_t       = fm.FontProperties(fname=font_path, size=18, weight='bold')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.weight': 'bold',
    'font.size': 28,
    'axes.titlesize': 28,
    'axes.titleweight': 'bold',
    'axes.labelsize': 28,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'legend.fontsize': 36,
    'legend.title_fontsize': 36,
    'pdf.fonttype': 42,
})
sns.set(style="whitegrid")
# ===============================================

# ============= 数据读取 =========================
pattern = re.compile(r"epr_results_(.+?)_topk(\d+)_scale(\d+)\.txt")
records = []
for path in glob(os.path.join(base_dir, "epr_results_*_topk*_scale*.txt")):
    m = pattern.match(os.path.basename(path))
    if not m:
        continue
    concept, topk, scale = m.group(1), int(m.group(2)), int(m.group(3))
    with open(path) as f:
        lines = f.readlines()
        epr = None
        target = None
        non_target = None
        for l in lines:
            if "NLS:" in l:
                epr = abs(float(l.split(":",1)[1].strip()))
            if "Target score mean change" in l:
                target = abs(float(l.split(":",1)[1].strip()))
            if "Non-target mean change" in l:
                non_target = abs(float(l.split(":",1)[1].strip()))
        if epr is not None and target is not None and non_target is not None:
            records.append({
                "concept": concept,
                "topk": topk,
                "scale": scale,
                "epr": epr,
                "target": target,
                "non_target": non_target
            })
df = pd.DataFrame(records)
# ===============================================

def add_stroke(ax):
    for line in ax.lines:
        line.set_linewidth(3)
        line.set_markersize(10)
        line.set_markeredgewidth(1.8)
        line.set_markeredgecolor('black')
        line.set_path_effects([pe.withStroke(linewidth=5, foreground='black'), pe.Normal()])

def style_axes(ax, xlabel, ylabel, title):
    ax.set_title(title, fontproperties=prop, pad=8)
    ax.set_xlabel(xlabel, fontproperties=prop, labelpad=6)
    ax.set_ylabel(ylabel, fontproperties=prop, labelpad=6)
    ax.tick_params(axis='both', which='major', width=2, length=6)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(prop_t)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 1. 横轴 α，纵轴是 epr/target/non-target 的绝对值
# def plot_metrics_vs_alpha(concept_fixed, topk_fixed):
#     sub = df[(df['concept'] == concept_fixed) & (df['topk'] == topk_fixed)].sort_values('scale')
#     fig, ax = plt.subplots(figsize=(12,8))
#     ax.plot(sub['scale'], sub['epr'], marker='o', label="EPR", linewidth=3)
#     ax.plot(sub['scale'], sub['target'], marker='s', label="Target", linewidth=3)
#     ax.plot(sub['scale'], sub['non_target'], marker='^', label="Non-target", linewidth=3)
#     add_stroke(ax)
#     style_axes(ax, xlabel=r"$\alpha$", ylabel="Score (Abs.)",
#                title=f"{concept_fixed}  |  TopK={topk_fixed}")
#     ax.margins(y=0.2)
#     ax.set_xticks(sorted(sub['scale'].unique()))
#     for label in ax.get_xticklabels():
#         label.set_fontproperties(prop_t)
#     leg = ax.legend(title="Metric", prop=prop_legend,
#                     loc='upper left', bbox_to_anchor=(1.02,1))
#     # if leg.get_title(): leg.get_title().set_fontweight('bold')
#     # leg.get_frame().set_linewidth(0.5)
#     leg = ax.legend(title="Metric", prop=prop_legend, loc='upper left', bbox_to_anchor=(1.02,1))
#     if leg.get_title():
#         leg.get_title().set_fontweight('bold')
#         leg.get_title().set_fontsize(prop_legend.get_size_in_points())
#     for t in leg.get_texts():
#         t.set_fontproperties(prop_legend)
#         t.set_fontsize(prop_legend.get_size_in_points())

#     plt.tight_layout()
#     fig.savefig(os.path.join(save_dir, f"metrics_vs_alpha_{concept_fixed}_topk{topk_fixed}.pdf"),
#                 bbox_inches='tight')
#     plt.close(fig)

# # 2. 横轴 topk，纵轴是 epr/target/non-target 的绝对值
# def plot_metrics_vs_topk(concept_fixed, alpha_fixed):
#     sub = df[(df['concept'] == concept_fixed) & (df['scale'] == alpha_fixed)].sort_values('topk')
#     fig, ax = plt.subplots(figsize=(12,8))
#     ax.plot(sub['topk'], sub['epr'], marker='o', label="EPR", linewidth=3)
#     ax.plot(sub['topk'], sub['target'], marker='s', label="Target", linewidth=3)
#     ax.plot(sub['topk'], sub['non_target'], marker='^', label="Non-target", linewidth=3)
#     add_stroke(ax)
#     style_axes(ax, xlabel="TopK", ylabel="Score (Abs.)",
#                title=f"{concept_fixed}  |  α={alpha_fixed}")
#     ax.margins(y=0.2)
#     ax.set_xticks(sorted(sub['topk'].unique()))
#     for label in ax.get_xticklabels():
#         label.set_fontproperties(prop_t)
#     leg = ax.legend(title="Metric", prop=prop_legend,
#                     loc='upper left', bbox_to_anchor=(1.02,1))
#     # if leg.get_title(): leg.get_title().set_fontweight('bold')
#     # leg.get_frame().set_linewidth(0.5)
#     leg = ax.legend(title="Metric", prop=prop_legend, loc='upper left', bbox_to_anchor=(1.02,1))
#     if leg.get_title():
#         leg.get_title().set_fontweight('bold')
#         leg.get_title().set_fontsize(prop_legend.get_size_in_points())
#     for t in leg.get_texts():
#         t.set_fontproperties(prop_legend)
#         t.set_fontsize(prop_legend.get_size_in_points())

#     plt.tight_layout()
#     fig.savefig(os.path.join(save_dir, f"metrics_vs_topk_{concept_fixed}_alpha{alpha_fixed}.pdf"),
#                 bbox_inches='tight')
#     plt.close(fig)

def plot_metrics_vs_alpha(concept_fixed, topk_fixed):
    sub = df[(df['concept'] == concept_fixed) & (df['topk'] == topk_fixed)].sort_values('scale')
    fig, ax1 = plt.subplots(figsize=(16,10))
    ax2 = ax1.twinx()

    l1, = ax1.plot(sub['scale'], sub['epr'], marker='o', color='C0', label="EPR", linewidth=3)
    l2, = ax2.plot(sub['scale'], sub['target'], marker='s', color='C1', label="Target", linewidth=3)
    l3, = ax2.plot(sub['scale'], sub['non_target'], marker='^', color='C2', label="Non-target", linewidth=3)
    for line in [l1,l2,l3]:
        line.set_path_effects([pe.withStroke(linewidth=5, foreground='black'), pe.Normal()])
        line.set_markersize(10)
        line.set_markeredgewidth(1.8)
        line.set_markeredgecolor('black')

    style_axes(ax1, xlabel=r"$\alpha$", ylabel="EPR", title=f"{concept_fixed}  |  TopK={topk_fixed}")
    style_axes(ax2, xlabel="", ylabel="Score (Target/Non-target)", title="")

    ax1.set_xticks(sorted(sub['scale'].unique()))
    for label in ax1.get_xticklabels():
        label.set_fontproperties(prop_t)
    ax1.margins(y=3)
    ax2.margins(y=0.4)
    # 图例合并
    lines = [l1, l2, l3]
    labels = ["EPR", "Target", "Non-target"]
    leg = ax1.legend(lines, labels, title="Metric", prop=prop_legend, loc='upper left', bbox_to_anchor=(1.1,1))
    if leg.get_title():
        leg.get_title().set_fontweight('bold')
        leg.get_title().set_fontsize(prop_legend.get_size_in_points())
    for t in leg.get_texts():
        t.set_fontproperties(prop_legend)
        t.set_fontsize(prop_legend.get_size_in_points())
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"metrics_vs_alpha_{concept_fixed}_topk{topk_fixed}_dualY.pdf"),
                bbox_inches='tight')
    plt.close(fig)

def plot_metrics_vs_topk(concept_fixed, alpha_fixed):
    sub = df[(df['concept'] == concept_fixed) & (df['scale'] == alpha_fixed)].sort_values('topk')
    fig, ax1 = plt.subplots(figsize=(16,10))
    ax2 = ax1.twinx()
    l1, = ax1.plot(sub['topk'], sub['epr'], marker='o', color='C0', label="EPR", linewidth=3)
    l2, = ax2.plot(sub['topk'], sub['target'], marker='s', color='C1', label="Target", linewidth=3)
    l3, = ax2.plot(sub['topk'], sub['non_target'], marker='^', color='C2', label="Non-target", linewidth=3)
    for line in [l1,l2,l3]:
        line.set_path_effects([pe.withStroke(linewidth=5, foreground='black'), pe.Normal()])
        line.set_markersize(10)
        line.set_markeredgewidth(1.8)
        line.set_markeredgecolor('black')
    style_axes(ax1, xlabel="TopK", ylabel="EPR", title=f"{concept_fixed}  |  α={alpha_fixed}")
    style_axes(ax2, xlabel="", ylabel="Score (Target/Non-target)", title="")
    ax1.set_xticks(sorted(sub['topk'].unique()))
    for label in ax1.get_xticklabels():
        label.set_fontproperties(prop_t)
    ax1.margins(y=0.4)
    ax2.margins(y=0.4)
    # 图例合并
    lines = [l1, l2, l3]
    labels = ["EPR", "Target", "Non-target"]
    leg = ax1.legend(lines, labels, title="Metric", prop=prop_legend, loc='upper left', bbox_to_anchor=(1.1,1))
    if leg.get_title():
        leg.get_title().set_fontweight('bold')
        leg.get_title().set_fontsize(prop_legend.get_size_in_points())
    for t in leg.get_texts():
        t.set_fontproperties(prop_legend)
        t.set_fontsize(prop_legend.get_size_in_points())
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"metrics_vs_topk_{concept_fixed}_alpha{alpha_fixed}_dualY.pdf"),
                bbox_inches='tight')
    plt.close(fig)
# def plot_metrics_vs_alpha(concept_fixed, topk_fixed):
#     sub = df[(df['concept'] == concept_fixed) & (df['topk'] == topk_fixed)].sort_values('scale')
#     fig, ax = plt.subplots(figsize=(12,8))
#     scale_t = 10  # 放大倍数，自己可以调
#     scale_nt = 20
#     ax.plot(sub['scale'], sub['epr'], marker='o', label="EPR", linewidth=3, color='C0')
#     ax.plot(sub['scale'], sub['target']*scale_t, marker='s', label=f"Target x{scale_t}", linewidth=3, color='C1')
#     ax.plot(sub['scale'], sub['non_target']*scale_nt, marker='^', label=f"Non-target x{scale_nt}", linewidth=3, color='C2')
#     add_stroke(ax)
#     style_axes(ax, xlabel=r"$\alpha$", ylabel="Score (Abs.)", title=f"{concept_fixed}  |  TopK={topk_fixed}")
#     ax.set_xticks(sorted(sub['scale'].unique()))
#     for label in ax.get_xticklabels():
#         label.set_fontproperties(prop_t)
#     ax.margins(y=0.2)
#     leg = ax.legend(title="Metric", prop=prop_legend, loc='upper left', bbox_to_anchor=(1.02,1))
#     if leg.get_title():
#         leg.get_title().set_fontweight('bold')
#         leg.get_title().set_fontsize(prop_legend.get_size_in_points())
#     for t in leg.get_texts():
#         t.set_fontproperties(prop_legend)
#         t.set_fontsize(prop_legend.get_size_in_points())
#     plt.tight_layout()
#     fig.savefig(os.path.join(save_dir, f"metrics_vs_alpha_{concept_fixed}_topk{topk_fixed}_scaled.pdf"),
#                 bbox_inches='tight')
#     plt.close(fig)
# def plot_metrics_vs_topk(concept_fixed, alpha_fixed):
#     sub = df[(df['concept'] == concept_fixed) & (df['scale'] == alpha_fixed)].sort_values('topk')
#     fig, ax = plt.subplots(figsize=(12,8))
#     scale_t = 10
#     scale_nt = 20
#     ax.plot(sub['topk'], sub['epr'], marker='o', label="EPR", linewidth=3, color='C0')
#     ax.plot(sub['topk'], sub['target']*scale_t, marker='s', label=f"Target x{scale_t}", linewidth=3, color='C1')
#     ax.plot(sub['topk'], sub['non_target']*scale_nt, marker='^', label=f"Non-target x{scale_nt}", linewidth=3, color='C2')
#     add_stroke(ax)
#     style_axes(ax, xlabel="TopK", ylabel="Score (Abs.)", title=f"{concept_fixed}  |  α={alpha_fixed}")
#     ax.set_xticks(sorted(sub['topk'].unique()))
#     for label in ax.get_xticklabels():
#         label.set_fontproperties(prop_t)
#     ax.margins(y=0.2)
#     leg = ax.legend(title="Metric", prop=prop_legend, loc='upper left', bbox_to_anchor=(1.02,1))
#     if leg.get_title():
#         leg.get_title().set_fontweight('bold')
#         leg.get_title().set_fontsize(prop_legend.get_size_in_points())
#     for t in leg.get_texts():
#         t.set_fontproperties(prop_legend)
#         t.set_fontsize(prop_legend.get_size_in_points())
#     plt.tight_layout()
#     fig.savefig(os.path.join(save_dir, f"metrics_vs_topk_{concept_fixed}_alpha{alpha_fixed}_scaled.pdf"),
#                 bbox_inches='tight')
#     plt.close(fig)

# ==== 用法举例 ====
# 画一张横轴α的
plot_metrics_vs_alpha(concept_fixed="smiling", topk_fixed=1)
# 画一张横轴topk的
plot_metrics_vs_topk(concept_fixed="smiling", alpha_fixed=128)

print("所有 PDF 图表已保存至:", save_dir)
