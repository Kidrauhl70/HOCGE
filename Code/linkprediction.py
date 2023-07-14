import numpy as np
import random
from sklearn.metrics import precision_recall_curve, auc

def LinkPrediction(model, edges, test_size=0.2):
    nodes = model.vectors.keys()
    
    # 从边列表中随机选择一部分作为测试集
    random.shuffle(edges)
    test_edges = edges[:int(len(edges) * test_size)]
    train_edges = edges[int(len(edges) * test_size):]

    # 计算训练集和测试集中的正样本和负样本
    train_pos = set(train_edges)
    train_neg = set()
    while len(train_neg) < len(train_pos):
        i, j = random.sample(nodes, 2)
        if i != j and (i, j) not in train_pos and (i, j) not in train_neg:
            train_neg.add((i, j))
    test_pos = set(test_edges)
    test_neg = set()
    while len(test_neg) < len(test_pos):
        i, j = random.sample(nodes, 2)
        if i != j and (i, j) not in train_pos and (i, j) not in train_neg and (i, j) not in test_pos and (i, j) not in test_neg:
            test_neg.add((i, j))

    # 计算训练集和测试集中的特征向量
    train_features = np.array([np.multiply(model.vectors[str(i)], model.vectors[str(j)]) for i, j in train_pos.union(train_neg)])
    train_labels = np.array([1] * len(train_pos) + [0] * len(train_neg))
    test_features = np.array([np.multiply(model.vectors[str(i)], model.vectors[str(j)]) for i, j in test_pos.union(test_neg)])
    test_labels = np.array([1] * len(test_pos) + [0] * len(test_neg))

    # 训练逻辑回归模型
    clf = LogisticRegression()
    clf.fit(train_features, train_labels)

    # 预测测试集中的边
    y_pred = clf.predict_proba(test_features)[:, 1]

    # 计算精度和召回率
    precision, recall, _ = precision_recall_curve(test_labels, y_pred)
    auc_score = auc(recall, precision)

    return auc_score
    