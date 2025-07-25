import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

fig = plt.figure(figsize=(7, 7))
ax = plt.gca()
ax.set_aspect('equal')
plt.axis('off')

# 向量定义
O = np.array([0, 0])
v_ideal = np.array([2, 0.3])
v_entangled = np.array([1, 2.2])
v_learned = v_ideal + 0.5 * v_entangled

# ideal direction（绿色虚线）
plt.arrow(*O, *v_ideal, color='green', linestyle='--', linewidth=2, length_includes_head=True, head_width=0.12, head_length=0.18)
plt.text(*(v_ideal*1.12), 'ideal semantic direction', color='green', fontsize=13, ha='left', va='center')

# entangled direction（红色虚线）
plt.arrow(*O, *v_entangled, color='red', linestyle='--', linewidth=2, length_includes_head=True, head_width=0.12, head_length=0.18)
plt.text(*(v_entangled*1.06), 'entangled direction', color='red', fontsize=13, ha='left', va='center')

# learned direction（黑色实线）
plt.arrow(*O, *v_learned, color='k', linestyle='-', linewidth=3, length_includes_head=True, head_width=0.15, head_length=0.23)
plt.text(*(v_learned*1.05), 'learned direction', color='k', fontsize=13, ha='left', va='center')

# learned在ideal方向上的投影
proj_ideal_len = np.dot(v_learned, v_ideal) / np.dot(v_ideal, v_ideal)
proj_ideal = proj_ideal_len * v_ideal
proj_entangled = v_learned - proj_ideal

# ideal分量（绿色点虚线）
plt.arrow(*O, *proj_ideal, color='green', linestyle=':', linewidth=2, length_includes_head=True, head_width=0.08, head_length=0.12)
plt.text(*(proj_ideal*1.04), 'ideal comp.', color='green', fontsize=11, ha='left', va='center')

# entangled分量（红色点虚线）
arrow_start = proj_ideal
plt.arrow(*arrow_start, *proj_entangled, color='red', linestyle=':', linewidth=2, length_includes_head=True, head_width=0.08, head_length=0.12)
plt.text(*(arrow_start+proj_entangled*0.55), 'entangled comp.', color='red', fontsize=11, ha='left', va='center')

# 起点
plt.plot([0], [0], 'ko', markersize=7)

plt.xlim(-0.5, max(v_learned[0], v_entangled[0])+1.0)
plt.ylim(-0.5, max(v_learned[1], v_entangled[1])+1.0)

plt.title('Illustration of semantic direction entanglement and decomposition', fontsize=14)
plt.tight_layout()

# 保存为PDF
plt.savefig('semantic_direction_decomposition.pdf', bbox_inches='tight', dpi=300)
plt.close()
