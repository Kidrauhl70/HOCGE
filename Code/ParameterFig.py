# import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 假设你有6个不同维度的实验结果，分别存储在这个列表中
# dimensions = ['Dim1', 'Dim2', 'Dim3', 'Dim4', 'Dim5', 'Dim6']


# dimensions
# dimensions = [16,32,64,128,256,512]
# wiki = [0.623383836, 0.64125677813, 0.65471371, 0.684, 0.674849281925, 0.666718158983501]
# air = [0.8199999, 0.86959999, 0.888, 0.892, 0.8768, 0.868]
# ship = [0.75714616, 0.78906741777, 0.813604995, 0.824, 0.8276294495, 0.81890035375]

# walk_length
# walk_length = [20,40,60,80,100,120]
# wiki = [0.629716728, 0.655964909, 0.670736835, 0.684, 0.678217969, 0.674363607]
# air = [0.883428571, 0.892571429, 0.891428571, 0.892, 0.892571429, 0.899428571]
# ship = [0.812847335, 0.813443528, 0.81354346, 0.824, 0.819615693, 0.82011288]

alpha = [1,5,10,15,20]
wiki = [0.645, 0.650, 0.672, 0.684, 0.679]


# # 创建一个DataFrame，方便使用Seaborn进行绘图
# data = pd.DataFrame({
#     'Dimension': dimensions,
#     'Wiki': wiki,
#     'Air': air,
#     'Ship': ship
# })

# # 使用Seaborn绘制折线图
# sns.set(style="white")  # 设置网格样式
# plt.figure(figsize=(4, 3))  # 设置图形大小

# # 绘制折线图
# sns.lineplot(x="Dimension", y="value", hue="variable", data=pd.melt(data, ['Dimension']), marker='o')

# # 设置标题
# plt.title('Effect of Dimensions on Experiment Results')


# # 显示图例
# plt.legend(title='Dataset')

# # 显示图表
# plt.show()

# 创建 x 轴标签
# x = np.arange(len(dimensions))
# x = np.arange(len(walk_length))
x = np.arange(len(alpha))

plt.figure(figsize=(5.1, 4),dpi=200)


# 创建每个数据集的折线图
plt.plot(x, wiki, marker='o', label='Wiki',)
# plt.plot(x, air, marker='o', label='Air')
# plt.plot(x, ship, marker='o', label='Ship')

# 设置 x 轴标签
# plt.xticks(x, dimensions)
# plt.xticks(x, walk_length)
plt.xticks(x, alpha)

# 设置 y 轴标签
plt.ylabel('F1 Score', fontsize=12)
plt.xlabel('α', fontsize=12)
# plt.xlabel('Dimensions', fontsize=12)

# 设置标题
# plt.title('Effect of Dimensions on Experiment Results')

# 添加图例
plt.legend()

# 显示图表
plt.show()
