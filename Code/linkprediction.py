import numpy as np
import pandas as pd
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from sklearn.model_selection import train_test_split
from graph import *
import time
from findedges import FindEdges
from getembeddings import GetEmbeddings
from args import parse_args
from filedirect import FileDirect
from utils import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import itertools

def LinkPrediction(result_filename, dataset,number_of_walks, walk_lengths_train, walk_lengths_test, args):
    dataset_filename_hon, dataset_filename, _ = FileDirect(args)

    # dataset_filename = '/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/air1.txt'
    # dataset_filename_hon = '/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/air2.txt'
    edges_info = pd.read_csv(dataset_filename, sep=' ', header=None, names=['source', 'target', 'weight'])
    graph = StellarGraph(edges=edges_info, is_directed=True)

    # print(graph.info())

    edge_splitter_test = EdgeSplitter(graph)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
    # reduced graph graph_test with the sampled links removed:
    graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(p=0.1, method="global")


    # print(graph_test.info())

    # Do the same process to compute a training subset from within the test graph
    edge_splitter_train = EdgeSplitter(graph_test, graph)
    graph_train, examples, labels = edge_splitter_train.train_test_split(p=0.1, method="global")
    (
        examples_train,
        examples_model_selection,
        labels_train,
        labels_model_selection,
    ) = train_test_split(examples, labels, train_size=0.75, test_size=0.25)

    # print(graph_train.info())

    matched_edges_filename_train = FindEdges(graph_train, dataset_filename_hon, True)


    pd.DataFrame(
        [
            (
                "Training Set",
                len(examples_train),
                "Train Graph",
                "Test Graph",
                "Train the Link Classifier",
            ),
            (
                "Model Selection",
                len(examples_model_selection),
                "Train Graph",
                "Test Graph",
                "Select the best Link Classifier model",
            ),
            (
                "Test set",
                len(examples_test),
                "Test Graph",
                "Full Graph",
                "Evaluate the best Link Classifier",
            ),
        ],
        columns=("Split", "Number of Examples", "Hidden from", "Picked from", "Use"),
    ).set_index("Split")


    # 在这加入改一下 初步思路：根据分割后的edge去选高阶的edge
    # 利用graph_train的边输入主函数，直接得到embedding_dict
    # 比对air1 air2 将air1中graph train的边对应air2 将对应边输入主函数，得到embedding_dict
    # embedding_dict = GetEmbeddings(args=parse_args(), num_neg_samples=1, num_epochs=10, counter=-80)
    args.input = matched_edges_filename_train
    embedding_dict = GetEmbeddings(args=args, 
                                num_neg_samples=1, num_epochs=10,
                                counter=-10, # counter=-10 代表训练集
                                walk_length=walk_lengths_train, window_size=5, 
                                pos_prob=0.3, neg_prob=0.3, 
                                number_walks=number_of_walks)

    # 这里根据嵌入向量维度改变
    default_embedding = np.zeros(args.representation_size)

    def get_embedding(node_id):
        try:
            return embedding_dict[str(node_id)]
        except KeyError:
            # 使用默认向量，如随机初始化或全零向量
            return default_embedding

    embedding_train = get_embedding

    def run_link_prediction(binary_operator):
        clf = train_link_prediction_model(examples_train, labels_train, embedding_train, binary_operator)
        test_score_roc, test_score_auprc, test_score_f1 = evaluate_link_prediction_model(
            clf,
            examples_model_selection,
            labels_model_selection,
            embedding_train,
            binary_operator,
        )

        return {
            "classifier": clf,
            "binary_operator": binary_operator,
            # "score": score,
            "score_roc": test_score_roc,
            "score_auprc": test_score_auprc,
            "score_f1": test_score_f1,
        }

    binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]

    results = [run_link_prediction(op) for op in binary_operators]
    # best_result = max(results, key=lambda result: result["score"])

    # print(f"Best result from '{best_result['binary_operator'].__name__}'")

    # print(pd.DataFrame(
    #     [(result["binary_operator"].__name__, result["score"]) for result in results],
    #     # columns=("name", "ROC AUC score"),
    #     columns=("name", "AUPRC score"),
    # ).set_index("name"))
    print(pd.DataFrame(
        [(result["binary_operator"].__name__, result["score_roc"],result["score_auprc"],result["score_f1"]) for result in results],
        columns=("name", "ROC AUC score","AUPRC score","F1 score"),
        # columns=("name", "AUPRC score"),
    ).set_index("name"))

    matched_edges_filename_test = FindEdges(graph_test, dataset_filename_hon, False)
    # embedding_dict = GetEmbeddings(args=parse_args(), num_neg_samples=1, num_epochs=10, counter=0)
    args.input = matched_edges_filename_test
    embedding_dict = GetEmbeddings(args=args, 
                                num_neg_samples=1, num_epochs=10,
                                counter=0, # counter=-10 代表训练集
                                walk_length=walk_lengths_test, window_size=5, 
                                pos_prob=0.3, neg_prob=0.3, 
                                number_walks=number_of_walks)


    embedding_test = get_embedding

    for result in results:
        best_result = result    
        test_score_roc, test_score_auprc, test_score_f1 = evaluate_link_prediction_model(
            best_result["classifier"],
            examples_test,
            labels_test,
            embedding_test,
            best_result["binary_operator"],
        )

        print(
            # f"ROC AUC score on test set using '{best_result['binary_operator'].__name__}': {test_score}"
            f"ROC AUPRC score on test set using '{best_result['binary_operator'].__name__}': {test_score_roc, test_score_auprc, test_score_f1}"
        )

        # # 先构造一个包含所有 score 的字符串
        # scores_str = ''
        # # for score in 'binary_operator':
        # scores_str += f"'{best_result['binary_operator'].__name__}': {test_score_roc, test_score_auprc, test_score_f1}\n"

        # # 然后一次性写入文件
        # with open(result_filename, 'a') as f:
        #     f.write(scores_str)
        #     f.write('---------------------------------------------\n')
        #     f.write(str(number_of_walks) + str(walk_lengths_train) + str(walk_lengths_test))

        with open(result_filename, 'a') as f:
            # f.write('---------------------------------------------\n')
            f.write(f"'{best_result['binary_operator'].__name__}': {test_score_roc, test_score_auprc, test_score_f1}\n")
            # f.write(str(number_of_walks) + str(walk_lengths_train) + str(walk_lengths_test))
            # f.write('---------------------------------------------\n')

        # with open(result_filename, 'a') as f:
        #     # f.write('---------------------------------------------\n')
        #     # f.write(f"'{best_result['binary_operator'].__name__}': {test_score_roc, test_score_auprc, test_score_f1}\n")
        #     f.write(str(number_of_walks) + str(walk_lengths_train) + str(walk_lengths_test))
        #     f.write('---------------------------------------------\n')
    
    # Calculate edge features for test data
    # link_features = link_examples_to_features(
    #     examples_test, embedding_test, best_result["binary_operator"]
    # )

    # Learn a projection from 128 dimensions to 2
    # pca = PCA(n_components=2)
    # X_transformed = pca.fit_transform(link_features)

    # # plot the 2-dimensional points
    # plt.figure(figsize=(16, 12))
    # plt.scatter(
    #     X_transformed[:, 0],
    #     X_transformed[:, 1],
    #     c=np.where(labels_test == 1, "b", "r"),
    #     alpha=0.5,
    # )
    # plt.show

dataset = 'wikijump'
result_filename = f'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/Results/LinkPrediction_{dataset}.txt'
# for i in range(4):
    # t1 = time.time()
seed = 8
# random.seed(seed)  
np.random.seed(seed)
number_of_walks_choices = [15]
walk_lengths_train_choices = [50]
walk_lengths_test_choices = [10]
# number_of_walks_choices = [40]
# walk_lengths_train_choices = [10]
# walk_lengths_test_choices = [10]
# for window_size, length, num_walk, sample in itertools.product(window_sizes, walk_length, num_walks, num_negative_samples):
for number_of_walks, walk_lengths_train, walk_lengths_test in itertools.product(number_of_walks_choices, walk_lengths_train_choices, walk_lengths_test_choices):
    for i in range(1):
        t1 = time.time()
        LinkPrediction(result_filename, dataset, number_of_walks, walk_lengths_train, walk_lengths_test, args=parse_args())
        print(number_of_walks, walk_lengths_train, walk_lengths_test)

        with open(result_filename, 'a') as f:
            # f.write('---------------------------------------------\n')
            # f.write(f"'{best_result['binary_operator'].__name__}': {test_score_roc, test_score_auprc, test_score_f1}\n")
            f.write(str(number_of_walks) + str(walk_lengths_train) + str(walk_lengths_test))
            f.write('\n---------------------------------------------\n')
        
        t2 = time.time()
    # t2 = time.time()
        print(f'time cost: {time.strftime("%H:%M:%S", time.gmtime(t2 - t1))}')

# python linkprediction.py --method deepWalk --input wikijump --directed --weighted --hon --task 1
