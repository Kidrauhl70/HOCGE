import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap
import seaborn as sns
import itertools

# 设置 Seaborn 样式
sns.set(style="white")

dataset = 'air'

# 读取嵌入向量文件
# embedding_file = f'/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/Results/ship_embedding.txt'
# embedding_file = f'/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/Results/air2_hon_deepWalk1.txt'
label_file = f'/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/{dataset}_labels.tsv'

def tSNE(label_file, counter, seed):
    targeted = True
    targeted = False

    embedding_file = f'/home/guozhi/桌面/HON_Embedding/MyCode/Visualization/air2_hon_deepWalk1.txt'
    # embedding_file = f'/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/Results/{dataset}_embedding.txt'
    embeddings = []
    with open(embedding_file, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            node_id = int(parts[0])
            embedding = np.array(parts[1:], dtype=float)
            embeddings.append((node_id, embedding))
            
    node_ids, embedding_vectors = zip(*embeddings)

    # 将嵌入向量转换为 NumPy 数组
    X = np.array(embedding_vectors)

    # 读取标签文件

    labels_df = pd.read_csv(label_file, sep='\t')

    # 合并嵌入向量和标签数据
    merged_data = pd.DataFrame({'ID': node_ids})
    merged_data = merged_data.merge(labels_df, on='ID')

    # 提取标签和转换为数值
    labels = merged_data['label']
    label_mapping = {label: idx for idx, label in enumerate(labels.unique())}
    merged_data['LabelIndex'] = merged_data['label'].map(label_mapping)
    print(f'{len(label_mapping)} unique labels')

    # 反向标签映射字典
    # reverse_label_mapping = {idx: label for label, idx in label_mapping.items()}

    # 使用 t-SNE 进行降维
    # iter = 2000
    # perplex = 80
    # perplexity(30.0)代表算法的困惑度，较大的困惑度会考虑更多的邻近点 (这个大一点对ship有提升)
    # early_exaggeration(12.0)用于控制在迭代的早期阶段对距离的放大程度。较大的值会导致更显著的聚类，但可能会引入不稳定性
    # learning_rate (200.0)指定学习率，影响嵌入点之间的距离更新速度。较大的学习率可能会导致更快的收敛，但也可能会导致嵌入点之间的错位
    tsne = TSNE(
        n_components=2, 
        random_state=seed
        # n_iter=iter, 
        # perplexity=perplex, 
        # early_exaggeration=12.0, 
        # learning_rate=200.0
                )
    tsne_results = tsne.fit_transform(X)

    if targeted: 
        target_labels = [
            'Great Lakes',  # 左下角下面积
            # 'Black Sea',  # 和Mediteranean Sea相近
            # 'Western Indian Ocean',  # 偏上，点太少
            'Somali/Arabian', # 右上 点不多
            'Mediterranean Sea', # 右边的占大面积的
            'Northern European Seas', # 下面的占大面积的
            # 'Western Coral Triangle',  # 上面的，偏散
            # 'North Brazil Shelf', # 太散
            'Tropical Northwestern Atlantic', # Great Lakes上面 部分有点散
            # 'Warm Temperate Northwest Atlantic', # Great Lakes上面 比上面那个还散
            # 'Cold Temperate Northwest Atlantic', # 连接Great Lakes上方
            'Cold Temperate Northwest Pacific', # 在上部分
            # 'West and South Indian Shelf', # 和Somali/Arabian相近
            # 'Bay of Bengal', # 和Somali/Arabian相近
            # 'Red Sea and Gulf of Aden',  # 和Somali/Arabian相近
            # 'South China Sea', # 在上部分 偏散
            # 'Sunda Shelf' # 和Cold Temperate Northwest Pacific相近 但是cold好一点
        ]             
        # 选择要绘制的特定标签值
        # target_labels = ['West and South Indian Shelf'
        #     ]
        # 随机选择两个标签值
        # target_labels = np.random.choice(list(label_mapping.keys()), size=5, replace=False)

        # print(f'Target labels: {target_labels}')    

    # 获取一个根据标签数量自动调整的颜色映射
        # cmap = sns.color_palette('tab20', n_colors=len(label_mapping))
        cmap = sns.color_palette()        

        # 筛选出符合特定标签值的数据
        filtered_data = merged_data[merged_data['label'].isin(target_labels)]
        filtered_tsne_results = tsne_results[filtered_data.index]

        plt.figure(figsize=(10, 8))
        sc = sns.scatterplot(x=filtered_tsne_results[:, 0], 
                             y=filtered_tsne_results[:, 1], 
                             hue=filtered_data['LabelIndex'], 
                             palette=cmap
                             )

        plt.title(f't-SNE Visualization of Selected Labels')
        plt.legend(title='Label', labels=target_labels,bbox_to_anchor=(1.00, 1), loc='upper left')
        # plt.savefig(f'/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/{dataset}Pics/tsne-{target_labels}.png')
        # plt.legend(title='Label', labels=[reverse_label_mapping[idx] for idx in target_labels])
        plt.show()        
    else:
        # 获取一个根据标签数量自动调整的颜色映射
        cmap = sns.color_palette("tab20", n_colors=len(label_mapping))

        # 绘制可视化结果，使用 Seaborn 来创建图表
        plt.figure(figsize=(5, 4))
        sc = sns.scatterplot(x=tsne_results[:, 0], 
                            y=tsne_results[:, 1], 
                            hue=merged_data['LabelIndex'], 
                            palette=cmap
                            )
        # 去掉图例
        sc.legend_.remove()
        # 不显示坐标轴刻度
        plt.xticks([])
        plt.yticks([])
        
        plt.title(f't-SNE Visualization of Embeddings seed{seed} counter{counter}')
        # plt.legend(title='Label', labels=label_mapping.keys(),bbox_to_anchor=(1.00, 1),loc='upper left')
        # plt.legend(title='Label', labels=[reverse_label_mapping[idx] for idx in label_mapping.values()])
        plt.show()
        # 保存图表至/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/pics
        # plt.savefig(f'/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/{dataset}Pics/iter3000/tsne-counter{counter}-seed{seed}-iter{iter}.png')


# labels = ['South China Sea', 'Sunda Shelf',
#        'Warm Temperate Northwest Pacific', 'West and South Indian Shelf',
#        'Eastern Coral Triangle', 'Java Transitional', 
#        'Cold Temperate Northwest Pacific', 'Western Coral Triangle',
#        'Warm Temperate Southwestern Atlantic', 'Bay of Bengal',
#        'Cold Temperate Northwest Atlantic', 'Temperate Australasia',
#        'Warm Temperate Northeast Pacific',
#        'Tropical Southwestern Atlantic', 'Mediterranean Sea',
#        'Warm Temperate Southeastern Pacific', 'Northern European Seas',
#        'Tropical East Pacific', 'Cold Temperate Northeast Pacific',
#        'Temperate Southern Africa', 'Red Sea and Gulf of Aden',
#        'Tropical Northwestern Atlantic',
#        'Warm Temperate Northwest Atlantic', 'North Brazil Shelf',
#        'Lusitanian', 'Tropical Southwestern Pacific', 'Sahul Shelf',
#        'Andaman', 'Western Indian Ocean', 'Eastern Indo-Pacific',
#        'Gulf of Guinea', 'Black Sea', 'Magellanic', 'Arctic',
#        'Great Lakes', 'West African Transition', 'Volga']

# seeds = [1, 4, 7,8,12,16,32]
# counters = [1,2,3,4,5,6,7,8,9,10]
seeds = [8]
counters = [1]


# for target_labels in labels:
#     tSNE(label_file, counter=counters, seed=seeds, target_labels=target_labels)


for seed, counter in itertools.product(seeds, counters):
    tSNE(label_file, counter=counter, seed=seed)
