import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns  # 导入Seaborn库

# 设置Seaborn样式
sns.set(style="whitegrid")

dataset = 'air'

# 读取嵌入向量文件
embedding_file_path = f'/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/Results/{dataset}_embedding.txt'
with open(embedding_file_path, "r") as f:
    next(f)
    lines = f.readlines()

# 解析嵌入向量文件中的数据
node_ids = []
embeddings = []
for line in lines:
    parts = line.strip().split()
    node_ids.append(parts[0])
    embeddings.append([float(x) for x in parts[1:]])

node_ids, embedding_vectors = zip(*embeddings)

# 转换为NumPy数组
embeddings = np.array(embeddings)

# 读取标签信息
label_file_path = f'/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/{dataset}_labels.tsv'
labels_df = pd.read_csv(label_file_path, sep='\t')

# 提取节点ID和标签列
labels = labels_df["ID"]
node_labels = labels_df["label"]

# 初始化t-SNE模型
tsne = TSNE(n_components=2, random_state=1)

# 对嵌入向量进行降维
embedded_vectors = tsne.fit_transform(embeddings)

# 绘制可视化图像
plt.figure(figsize=(10, 8))
unique_labels = np.unique(node_labels)
for label in unique_labels:
    idx = node_labels == label
    plt.scatter(embedded_vectors[idx, 0], embedded_vectors[idx, 1], label=label)

plt.legend()
plt.title("t-SNE Visualization of Node Embeddings with Labels")
# 显示网格
plt.grid(True)

# 使用Seaborn进行美化
sns.despine()

# 显示图像
plt.show()






