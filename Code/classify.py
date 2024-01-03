from __future__ import print_function
import numpy 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from time import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]  # 每个节点的标签个数（为了多标签分类问题）
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        # averages = ["micro", "macro", "samples", "weighted"]
        averages = ["micro"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        # print('Results, using embeddings of dimensionality', len(self.embeddings[X[0]]))
        # print('-------------------')
        print(results)
        return results
        # print('-------------------')

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        # try:
        #     X_ = numpy.asarray([self.embeddings[x] for x in X])
        # except KeyError as e:
        #     print(f"KeyError: {e} for node {x}")
        #     x = e.args[0]
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=None):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]
        # 检查是否有重复的节点
        assert len(pd.unique(X_train)) + len(pd.unique(X_test)) == len(X), "X_train and X_test are not disjoint"
        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)


    # def split_train_evaluate(self, X, Y, k=5, seed=None):  # k折交叉验证
    #     state = numpy.random.get_state()

    #     # 将X和Y转换为numpy数组，方便后续处理
    #     X = numpy.array(X)
    #     Y = numpy.array(Y)

    #     # 使用StratifiedKFold进行分层5折交叉验证
    #     skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    #     accuracy_scores = []  # 用于保存每一折的准确率
    #     score = []
    #     i = 0
    #     for train_indices, test_indices in skf.split(X, Y):
    #         X_train, X_test = X[train_indices], X[test_indices]
    #         Y_train, Y_test = Y[train_indices], Y[test_indices]

    #         # 检查是否有重复的节点
    #         assert len(pd.unique(X_train)) + len(pd.unique(X_test)) == len(X), "X_train and X_test are not disjoint"

    #         # 训练模型
    #         self.train(X_train.tolist(), Y_train.tolist(), Y.tolist())

    #         # 评估模型
    #         accuracy = self.evaluate(X_test.tolist(), Y_test.tolist())
    #         score.append(accuracy['micro'])
    #         i += 1
    #         accuracy_scores.append(accuracy)

    #     numpy.random.set_state(state)
    #     mean_accuracy = numpy.mean(score)
    #     print('mean_accuracy:', mean_accuracy)
    #     return mean_accuracy


def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size+1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num
    return vectors


def read_node_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        if filename.split('.')[-1] == 'csv':
            vec = l.strip().split(',')
        else:
            vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y
