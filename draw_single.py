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
save_dir = "./charts_results_pdf/single"
os.makedirs(save_dir, exist_ok=True)
font_path = "/u/wff7ad/.fonts/TIMES.TTF"
prop = fm.FontProperties(fname=font_path, size=48, weight='bold')
prop_legend = fm.FontProperties(fname=font_path, size=38, weight='normal')  # 图例也用字体文件
# prop_legend = fm.FontProperties(fname='/u/wff7ad/.fonts/stix/STIXTwoText-BoldItalic.otf', size=24, weight='normal')

prop_t       = fm.FontProperties(fname=font_path, size=36, weight='bold')
prop_xlabel  = fm.FontProperties(fname=font_path, size=48, weight='normal')  # 不加粗的xlabel

# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman'],
#     'font.weight': 'normal',
#     'font.size': 28,
#     'axes.titlesize': 28,
#     'axes.titleweight': 'bold',
#     'axes.labelsize': 28,
#     'axes.labelweight': 'normal',  # 改为 normal，这样 xlabel 不会自动加粗
#     'xtick.labelsize': 26,
#     'ytick.labelsize': 26,
#     'legend.fontsize': 36,
#     'legend.title_fontsize': 36,
#     'pdf.fonttype': 42,
#     'mathtext.fontset': 'stix',     # 让公式使用 STIX（接近 Times New Roman）
#     'mathtext.rm': 'Times New Roman',
#     'mathtext.it': 'Times New Roman:italic',
#     'mathtext.bf': 'Times New Roman:bold',
#     'mathtext.default': 'rm',

# })


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.weight': 'normal',
    'font.size': 28,
    'axes.titlesize': 28,
    'axes.titleweight': 'bold',
    'axes.labelsize': 28,
    'axes.labelweight': 'normal',
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'legend.fontsize': 36,
    'legend.title_fontsize': 36,
    'pdf.fonttype': 42,

    # 数学字体配置    
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'STIX Two Text',       # 普通
    'mathtext.it': 'STIX Two Text:italic',# 斜体
    'mathtext.bf': 'STIX Two Text:bold',  # 粗体
    'mathtext.tt': 'STIX Two Text',
    'mathtext.default': 'rm',
})

sns.set_style("ticks", {
    "axes.facecolor": "#f5f5f5",    # 灰白背景
    "axes.edgecolor": "black",      # 坐标轴黑边
    "grid.color": "white",          # 内部网格白色
    "grid.linestyle": "-", 
    "axes.grid": True,              # 开启网格
    "axes.axisbelow": True          # 网格在线下层
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

    ax.set_facecolor('#f5f5f5')



def plot_metrics_vs_alpha(concept_fixed, topk_fixed):
    sub = df[(df['concept'] == concept_fixed) & (df['topk'] == topk_fixed)].sort_values('scale')

    # === 创建图像 ===
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = ax1.twinx()  # 双 Y 轴

    # === 绘图 ===
    l1, = ax1.plot(sub['scale'], sub['epr'],
                   color='C0', marker='o', linewidth=3, label="EPR",
                   path_effects=[pe.withStroke(linewidth=5, foreground='black'), pe.Normal()],
                   markeredgewidth=1.8, markeredgecolor='black', markersize=10)

    l2, = ax2.plot(sub['scale'], sub['target'],
                   color='C1', marker='s', linewidth=3, label=r"Target",
                   path_effects=[pe.withStroke(linewidth=5, foreground='black'), pe.Normal()],
                   markeredgewidth=1.8, markeredgecolor='black', markersize=10)

    l3, = ax2.plot(sub['scale'], sub['non_target'],
                   color='C2', marker='^', linewidth=3, label=r"Non-Target",
                   path_effects=[pe.withStroke(linewidth=5, foreground='black'), pe.Normal()],
                   markeredgewidth=1.8, markeredgecolor='black', markersize=10)

    # === 图层控制 ===
    ax1.set_zorder(1)          # 主轴在下（有背景和网格）
    ax2.set_zorder(2)          # 副轴在上（透明）
    ax1.set_axisbelow(True)    # 主轴网格在底层
    ax2.set_axisbelow(False)   # 副轴无网格
    
    # 线条层级设置
    l1.set_zorder(10)          # EPR线在主轴上方
    l2.set_zorder(20)          # Target线在副轴，层级更高
    l3.set_zorder(20)          # Non-target线在副轴，层级更高
    
    # 背景设置
    ax1.patch.set_visible(True)    # 主轴有灰色背景
    ax1.set_facecolor('#f5f5f5')   # 主轴灰色背景
    ax2.patch.set_visible(False)   # 副轴透明背景
    ax2.set_facecolor('none')      # 副轴完全透明

    # === 样式 ===
    style_axes(ax1, xlabel="α", ylabel="EPR", title="")
    
    # 副轴样式：无网格，只设置标签
    ax2.set_ylabel("Δ", fontproperties=prop, labelpad=6, weight='bold')
    ax2.yaxis.set_label_coords(1.08, 0.45)  # 调整副轴Y标签位置，再往右移一点
    ax2.tick_params(axis='both', which='major', labelsize=26, width=2.5, length=6, color='black')
    
    # 设置副轴Y轴刻度保留1位小数（主/副轴将统一为1位）
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    
    for t in ax2.get_yticklabels():
        t.set_fontproperties(prop_t)
    # 副轴关闭所有网格
    ax2.grid(False)
    ax2.xaxis.grid(False)
    ax2.yaxis.grid(False)

    # === 坐标 & 网格 ===
    # 横轴刻度更稀疏 - 只显示部分α值
    all_scales = sorted(sub['scale'].unique())
    sparse_scales = [all_scales[i] for i in range(0, len(all_scales), 2)]  # 每隔一个显示
    ax1.set_xticks(sparse_scales)
    for label in ax1.get_xticklabels():
        label.set_fontproperties(prop_t)
    
    # 纵轴刻度适中
    ax1.locator_params(axis='y', nbins=6)  # EPR轴最多6个刻度
    ax2.locator_params(axis='y', nbins=6)  # δ轴最多6个刻度
    # 主轴和副轴刻度格式为一位小数
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        # 主轴和副轴刻度格式为一位小数
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # === 图例 ===
    lines = [l1, l2, l3]
    labels = [L.get_label() for L in lines]
    leg = ax2.legend(lines, labels, prop=prop_legend,
                     loc='upper right', frameon=True, framealpha=0.6,
                     facecolor='#fdfdfd',
                     bbox_to_anchor=(1.02, 1.02),  # 更靠右上角
                     columnspacing=0.3,  # 列间距
                     handletextpad=0.2,  # 图例标记与文本间距
                     handlelength=0.8,   # 图例标记长度
                     borderpad=0.2,      # 图例内边距
                     markerscale=0.8)    # 标记大小
    leg.get_frame().set_linewidth(0.5)  # 和 draw_charts.py 一样的细边框
    leg.set_zorder(200)  # 图例在最上层，比所有线条都高

    # === 保存 ===
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"metrics_vs_alpha_{concept_fixed}_topk{topk_fixed}_dualY.pdf"),
                bbox_inches='tight')
    plt.close(fig)


def plot_metrics_vs_topk(concept_fixed, alpha_fixed):
    sub = df[(df['concept'] == concept_fixed) & (df['scale'] == alpha_fixed)].sort_values('topk')

    # === 创建图像 ===
    fig, ax1 = plt.subplots(figsize=(20, 8))
    ax2 = ax1.twinx()

    # === 绘图 ===
    l1, = ax1.plot(sub['topk'], sub['epr'],
                   color='C0', marker='o', linewidth=3, label="EPR",
                   path_effects=[pe.withStroke(linewidth=5, foreground='black'), pe.Normal()],
                   markeredgewidth=1.8, markeredgecolor='black', markersize=10)

    l2, = ax2.plot(sub['topk'], sub['target'],
                   color='C1', marker='s', linewidth=3, label="Target",
                   path_effects=[pe.withStroke(linewidth=5, foreground='black'), pe.Normal()],
                   markeredgewidth=1.8, markeredgecolor='black', markersize=10)

    l3, = ax2.plot(sub['topk'], sub['non_target'],
                   color='C2', marker='^', linewidth=3, label="Non-target",
                   path_effects=[pe.withStroke(linewidth=5, foreground='black'), pe.Normal()],
                   markeredgewidth=1.8, markeredgecolor='black', markersize=10)

    # === 图层控制 ===
    ax1.set_zorder(1)          # 主轴在下（有背景和网格）
    ax2.set_zorder(2)          # 副轴在上（透明）
    ax1.set_axisbelow(True)    # 主轴网格在底层
    ax2.set_axisbelow(False)   # 副轴无网格
    
    # 线条层级设置
    l1.set_zorder(10)          # EPR线在主轴上方
    l2.set_zorder(20)          # Target线在副轴，层级更高
    l3.set_zorder(20)          # Non-target线在副轴，层级更高
    
    # 背景设置
    ax1.patch.set_visible(True)    # 主轴有灰色背景
    ax1.set_facecolor('#f5f5f5')   # 主轴灰色背景
    ax2.patch.set_visible(False)   # 副轴透明背景
    ax2.set_facecolor('none')      # 副轴完全透明

    # === 样式 ===
    style_axes(ax1, xlabel="TopK", ylabel="EPR", title="")
    
    # 副轴样式：无网格，只设置标签
    ax2.set_ylabel("Δ", fontproperties=prop, labelpad=6, weight='bold')
    ax2.yaxis.set_label_coords(1.08, 0.45)  # 调整副轴Y标签位置，再往右移一点
    ax2.tick_params(axis='both', which='major', labelsize=26, width=2.5, length=6, color='black')
    
    # 设置副轴Y轴刻度保留1位小数（主/副轴将统一为1位）
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    
    for t in ax2.get_yticklabels():
        t.set_fontproperties(prop_t)
    # 副轴关闭所有网格
    ax2.grid(False)
    ax2.xaxis.grid(False)
    ax2.yaxis.grid(False)

    # === 坐标 & 图例 ===
    # TopK轴刻度稀疏化 - 只显示部分值
    all_topks = sorted(sub['topk'].unique())
    if len(all_topks) > 4:
        sparse_topks = [all_topks[i] for i in range(0, len(all_topks), max(1, len(all_topks)//4))]
    else:
        sparse_topks = all_topks
    ax1.set_xticks(sparse_topks)
    for label in ax1.get_xticklabels():
        label.set_fontproperties(prop_t)
    
    # 纵轴刻度适中
    ax1.locator_params(axis='y', nbins=6)  # EPR轴最多6个刻度
    ax2.locator_params(axis='y', nbins=6)  # δ轴最多6个刻度

    lines = [l1, l2, l3]
    labels = [L.get_label() for L in lines]
    leg = ax2.legend(lines, labels, prop=prop_legend,
                     loc='upper right', frameon=True, framealpha=0.6,
                     facecolor='#fdfdfd',
                     bbox_to_anchor=(1.02, 1.02),  # 更靠右上角
                     columnspacing=0.3,  # 列间距
                     handletextpad=0.2,  # 图例标记与文本间距
                     handlelength=0.8,   # 图例标记长度
                     borderpad=0.2,      # 图例内边距
                     markerscale=0.8)    # 标记大小
    leg.get_frame().set_linewidth(0.5)  # 和 draw_charts.py 一样的细边框
    leg.set_zorder(200)  # 图例在最上层，比所有线条都高

    # === 保存 ===
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"metrics_vs_topk_{concept_fixed}_alpha{alpha_fixed}_dualY.pdf"),
                bbox_inches='tight')
    plt.close(fig)


# ==== 用法举例 ====
# 画一张横轴α的
for c in df['concept'].unique():
    plot_metrics_vs_alpha(concept_fixed=c, topk_fixed=1)
# 画一张横轴topk的
    plot_metrics_vs_topk(concept_fixed=c, alpha_fixed=128)

    print("所有 PDF 图表已保存至:", save_dir)
