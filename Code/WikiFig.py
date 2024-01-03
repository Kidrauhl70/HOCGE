# import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker

# 假设你有6个不同维度的实验结果，分别存储在这个列表中
# dimensions = ['Dim1', 'Dim2', 'Dim3', 'Dim4', 'Dim5', 'Dim6']


# dimensions
# dimensions = [16,32,64,128,256,512]
# num_node = [0.623383836, 0.64125677813, 0.65471371, 0.684, 0.674849281925, 0.666718158983501]
# num_edges = [0.8199999, 0.86959999, 0.888, 0.892, 0.8768, 0.868]
# ship = [0.75714616, 0.78906741777, 0.813604995, 0.824, 0.8276294495, 0.81890035375]

# alpha
alpha = [1,5,10,15,20]
num_nodes = [163201,43985,9898,4568,4211]
num_edges = [389623,205885,97923,73010,71319]
# ship = [0.812847335, 0.813443528, 0.81354346, 0.824, 0.819615693, 0.82011288]

# 创建 x 轴标签
# x = np.arange(len(dimensions))
x = np.arange(len(alpha))

plt.figure(figsize=(5.1, 4),dpi=200)

# 创建每个数据的柱状图
plt.bar(x - 0.2, num_nodes, label='Number of nodes', width=0.4, color='C9')
plt.bar(x + 0.2, num_edges, label='Number of edges', width=0.4, color='C0')

# 在柱状图上标注数字
for i in range(len(x)):
    plt.text(x[i] - 0.2, num_nodes[i], str(num_nodes[i]), ha='center', va='bottom',fontsize=6)
    plt.text(x[i] + 0.2, num_edges[i], str(num_edges[i]), ha='center', va='bottom',fontsize=6)


# 设置 x 轴标签
plt.xticks(x, alpha)

plt.yscale('log')
# plt.gca().set_yscale('log')

# 设置 y 轴标签
plt.ylabel('Graph Size', fontsize=12)
plt.xlabel('α', fontsize=12)

# 添加图例
plt.legend()

# 显示图表
plt.show()

