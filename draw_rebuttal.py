import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker
import seaborn as sns
from glob import glob

# 清除matplotlib字体缓存，确保字体设置生效
try:
    fm.fontManager.__init__()
except:
    pass

# ==================== 配置 ======================
base_dir = "./epr/resnet/sae"
save_dir = "./charts_results_pdf/new_single"
os.makedirs(save_dir, exist_ok=True)
font_path = "/u/wff7ad/.fonts/TIMES.TTF"
prop = fm.FontProperties(fname=font_path, size=126, weight='bold')
prop_legend = fm.FontProperties(fname=font_path, size=86, weight='normal')
prop_t = fm.FontProperties(fname=font_path, size=86, weight='bold')
prop_xlabel = fm.FontProperties(fname=font_path, size=86, weight='normal')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.weight': 'normal',
    'font.size': 48,
    'axes.titlesize': 66,
    'axes.titleweight': 'bold',
    'axes.labelsize': 66,
    'axes.labelweight': 'normal',
    'xtick.labelsize': 46,
    'ytick.labelsize': 46,
    'legend.fontsize': 66,
    'legend.title_fontsize': 46,
    'pdf.fonttype': 42,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'STIX Two Text',
    'mathtext.it': 'STIX Two Text:italic',
    'mathtext.bf': 'STIX Two Text:bold',
    'mathtext.tt': 'STIX Two Text',
    'mathtext.default': 'rm',
})

sns.set_style("ticks", {
    "axes.facecolor": "#f5f5f5",
    "axes.edgecolor": "black",
    "grid.color": "white",
    "grid.linestyle": "-", 
    "axes.grid": True,
    "axes.axisbelow": True
})
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
# 构建DataFrame
df = pd.DataFrame(records)
# 调试：输出df的列名
print('DataFrame columns:', df.columns.tolist())
# ===============================================

def style_axes(ax, xlabel, ylabel, title):
    """统一设置标题、坐标轴标签、刻度、边框等"""
    ax.set_title(title, fontproperties=prop, pad=8, weight='bold')
    ax.set_xlabel(xlabel, fontproperties=prop_xlabel, labelpad=6)
    ax.set_ylabel(ylabel, fontproperties=prop, labelpad=6, weight='bold')

    ax.tick_params(axis='both', which='major',
                   labelsize=56, width=2.5, length=6, color='black')
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(prop_t)

    ax.yaxis.grid(True, color='#b0b0b0', linestyle='--', linewidth=2.0)
    ax.xaxis.grid(True, color='#b0b0b0', linestyle='--', linewidth=2.0)
    ax.set_facecolor('#f5f5f5')


def plot_delta_vs_topk(concept_fixed, alpha_fixed):
    sub = df[(df['concept'] == concept_fixed) & (df['scale'] == alpha_fixed) & (df['topk'] >= 1) & (df['topk'] <= 8)].sort_values('topk')

    # === 创建图像 ===
    fig, ax = plt.subplots(figsize=(22, 8))

    # === 绘图：实线和虚线 ===
    l1, = ax.plot(sub['topk'], sub['target'],
                   color='C1', marker='s', linewidth=3, linestyle='-', label="Target",
                   path_effects=[pe.withStroke(linewidth=5, foreground='black'), pe.Normal()],
                   markeredgewidth=1.8, markeredgecolor='black', markersize=10)

    l2, = ax.plot(sub['topk'], sub['non_target'],
                   color='C2', marker='^', linewidth=3, linestyle='--', label="Non-target",
                   path_effects=[pe.withStroke(linewidth=5, foreground='black'), pe.Normal()],
                   markeredgewidth=1.8, markeredgecolor='black', markersize=10)

    # === 样式 ===
    style_axes(ax, xlabel="TopK", ylabel="Δ", title="")
    
    # Y轴刻度保留1位小数
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # === 坐标 ===
    # TopK轴刻度稀疏化
    all_topks = sorted(sub['topk'].unique())
    if len(all_topks) > 4:
        sparse_topks = [all_topks[i] for i in range(0, len(all_topks), max(1, len(all_topks)//4))]
    else:
        sparse_topks = all_topks
    ax.set_xticks(sparse_topks)
    
    # 纵轴刻度适中
    ax.locator_params(axis='y', nbins=6)

    # === 图例 ===
    lines = [l1, l2]
    labels = [L.get_label() for L in lines]
    leg = ax.legend(lines, labels, prop=prop_legend,
                     loc='upper right', frameon=True, framealpha=0.6,
                     facecolor='#fdfdfd',
                     bbox_to_anchor=(1.0, 1.0),
                     columnspacing=0.3,
                     handletextpad=0.2,
                     handlelength=0.8,
                     borderpad=0.2,
                     markerscale=0.8)
    leg.get_frame().set_linewidth(0.5)
    leg.set_zorder(200)

    # === 保存 ===
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"delta_vs_topk_{concept_fixed}_alpha{alpha_fixed}.pdf"),
                bbox_inches='tight')
    plt.close(fig)


# ==== 使用示例 ====
for c in df['concept'].unique():
    plot_delta_vs_topk(concept_fixed=c, alpha_fixed=128)

print("所有 Delta PDF 图表已保存至:", save_dir)