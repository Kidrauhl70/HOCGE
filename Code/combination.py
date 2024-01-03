import numpy as np
from sklearn.linear_model import LogisticRegression
from classify import *

fon_filename = '/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/Results/wiki1_fon_node2vec.txt'
hon_filename = '/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/Results/wiki2_hon_node2vec.txt'
label_file = '/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/wiki_labels.csv'

# 尝试把两个embedding进行组合，结果效果不好
def Combination(fon_filename, hon_filename, alpha=1):
    # 读取文件，将embedding转成字典
     fon_embedding = {}
     with open (fon_filename, 'r') as f:
          next(f)
          for line in f:
               values = line.split()
               node = values[0].split('|')[0]
               emb = np.asarray(values[1:], dtype='float32')
               fon_embedding[node] = emb
     
     hon_embedding = {}
     with open (hon_filename, 'r') as f:
          next(f)
          for line in f:
               values = line.split()
               node = values[0]
               emb = np.asarray(values[1:], dtype='float32')
               hon_embedding[node] = emb

     # 对于key相同的节点，将embedding进行平均
     combine_emb = {}
     for key in fon_embedding:
          if key in hon_embedding:
               # 判断两个embedding之间的距离，如果距离太大，就不进行组合
               distance = np.linalg.norm(fon_embedding[key] - hon_embedding[key])
               if distance < 3:
                    # combine_emb[int(key)] = alpha * fon_embedding[key] + (1 - alpha) * hon_embedding[key]
                    # combine为较大的embedding
                    combine_emb[int(key)] = fon_embedding[key] if np.linalg.norm(fon_embedding[key]) < np.linalg.norm(hon_embedding[key]) else hon_embedding[key]
               else:
                    combine_emb[int(key)] = hon_embedding[key]
     
     return combine_emb


combine_emb = Combination(fon_filename, hon_filename)

p = 0.7

X, Y = read_node_label(label_file)
print(f"Training classifier using {p*100:.2f}% nodes...")
clf = Classifier(vectors=combine_emb, clf=LogisticRegression())
# clf = Classifier(vectors=vectors, clf=LogisticRegression())
clf.split_train_evaluate(X, Y, p, seed=0)
