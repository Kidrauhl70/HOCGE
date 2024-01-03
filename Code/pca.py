import itertools
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# apply PCA to node embeddings
# and report the variance ratio of PCA
# input file: node embeddings.tsv and labelfile.tsv
# output: visualization of PCA and report the variance ratio

def PCAvis(label_file, embedding_file, seed):
    # 读取标签和嵌入
    labels = pd.read_csv(label_file, sep='\t', header=None)
    embeddings = pd.read_csv(embedding_file, sep=' ', header=None, skiprows=1)
    # embeddings = pd.read_csv(embedding_file, sep='\t', header=None)

    # 按照节点ID进行排序
    labels = labels.sort_values(by=0)
    embeddings = embeddings.sort_values(by=0)

    # 只取标签文件的第二列（类别）
    labels = labels[1]

    # 只取嵌入文件的嵌入向量列
    embeddings = embeddings.iloc[:, 1:]

    # Scale the data
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # 创建 PCA 对象
    pca = PCA(n_components=2, random_state=seed)

    # 对嵌入进行 PCA
    embeddings_pca = pca.fit_transform(embeddings_scaled)

    # 打印方差比率
    print('Explained variance ratio:', pca.explained_variance_ratio_)

    # 创建一个新的 DataFrame，其中包含 PCA 嵌入和标签
    df = pd.DataFrame(data = embeddings_pca, columns = ['principal component 1', 'principal component 2'])
    df = pd.concat([df, labels], axis = 1)

    # 绘制 PCA 嵌入
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    targets = labels.unique()
    # 统计类别数
    num_classes = len(targets)
    # 生成colorplate
    colorplate = plt.cm.get_cmap('gist_rainbow', num_classes)
    for target, color in zip(targets,colorplate(np.linspace(0, 1, num_classes))):
        indicesToKeep = df[1] == target
        ax.scatter(df.loc[indicesToKeep, 'principal component 1']
                   , df.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()


label_file = 'C:/Users/Lenovo/Desktop/MyCode/Visualization/TSV/air_labels_nohead.tsv'
embedding_file = 'C:/Users/Lenovo/Desktop/MyCode/Visualization/air2_hon_deepWalk1.txt'
seeds = [4]
for seed in seeds:
    PCAvis(label_file, embedding_file, seed)