import numpy as np
import networkx as nx
from collections import defaultdict

# dim = 64

# embedding_filename = "/home/guozhi/桌面/HON_Embedding/OpenNE-master/data/network-Disorder-2000-lines-embeddings.txt"
# edgelist_filename = "/home/guozhi/桌面/HON_Embedding/OpenNE-master/data/network-Disorder-2000-lines-manual.txt"

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def aggregate_embeddings(embeddings, softmax_scores, dim, weight_dict):
    embedding_dict = {}
    for key in softmax_scores.keys():
        embedding_dict[key] = np.zeros(dim)
        i = 0  # 定义索引 i
        for sub_key, sub_value in weight_dict[key].items():
            embedding_dict[key] += np.array(embeddings[sub_key]) * softmax_scores[key][i]
            i += 1  # 在每次迭代中增加索引
    return embedding_dict



def edge_weight(edgelist_filename):
    weight_dict = {}
    with open(edgelist_filename, 'r') as f:
        for line in f:
            if edgelist_filename.split('.')[-1] == 'csv':
                row = line.split(',')
            else:
                row = line.split(' ')            
            key = int(row[0].split('|')[0]) # 将节点转换为int类型
            sub_key = row[0]
            sub_value = float(row[2]) # 将权重转换为float类型
            if key in weight_dict.keys():
                if sub_key in weight_dict[key]: # 如果已经存在该高阶节点，则将权重相加
                    weight_dict[key][sub_key] += sub_value
                else:
                    weight_dict[key][sub_key] = sub_value
            else:
                weight_dict[key] = {sub_key: sub_value}

            # 也考虑节点作为边终点的情况
            key = int(row[1].split('|')[0]) # 将节点转换为int类型
            sub_key = row[1]
            if key in weight_dict.keys():
                if sub_key in weight_dict[key]: # 如果已经存在该高阶节点，则将权重相加
                    weight_dict[key][sub_key] += sub_value
                else:
                    weight_dict[key][sub_key] = sub_value
            else:
                weight_dict[key] = {sub_key: sub_value}
    return weight_dict


def Aggregate(dim, embeddings, graph, output_filename, type):
    
    print("Aggregating...")

    hon_embedding_dict = embeddings

    # 利用graph得到每个节点的权重 {节点：{变阶节点：权重}} 
    # pred中key为B，value为A，表示A指向B，weight为A指向B的边的权重；succ相反
    node_weight_dict_pred = {}
    for key, value in graph._pred.items():
        # 计算value中所有weight的和
        total_weight = sum(item['weight'] for item in value.values())
        # 将结果存入node_weight_dict中
        node_weight_dict_pred[key] = total_weight

    node_weight_dict_succ = {}
    for key, value in graph._succ.items():
        # 计算value中所有weight的和
        total_weight = sum(item['weight'] for item in value.values())
        # 将结果存入node_weight_dict中
        node_weight_dict_succ[key] = total_weight

    # node_weight_dict 为{节点：权重}，将pred和succ中的相同节点的权重相加
    node_weight_dict = {}
    for key in node_weight_dict_pred.keys():
        if key in node_weight_dict_succ.keys():
            node_weight_dict[key] = node_weight_dict_pred[key] + node_weight_dict_succ[key]
        else:
            node_weight_dict[key] = node_weight_dict_pred[key]
    
    # 前置节点相同的变阶节点列入同一个字典，{节点：{变阶节点：权重}} 
    weight_dict = {}
    for node in node_weight_dict.keys():
        key = int(node.split('|')[0])
        # 降低低阶节点权重
        if node_weight_dict[node] > 1000: # 这里降低最大权重
            node_weight_dict[node]/=200
        if key in weight_dict.keys():
            weight_dict[key][node] = node_weight_dict[node]
        else:
            weight_dict[key] = {node: node_weight_dict[node]}

    # 将weight_dict的每个权重都设置为1 （测试avg）
    if type:
        print('avg mode')
        for key in weight_dict.keys():
            for sub_key in weight_dict[key].keys():
                weight_dict[key][sub_key] = 1

    # 用于确认weight_dict的key是否包含所有节点
    node_size = 0
    for key in weight_dict.keys():
        node_size += len(weight_dict[key])
    print(f'node_size: {node_size}')

    # # 用于调试
    # print('done')
    # --------------------------这里是softmax 计算归一化----------------------------
    # for key in weight_dict.keys():
    #     sub_key_values = list(weight_dict[key].values())
    #     normalized_weights = softmax(sub_key_values)
    #     for i, sub_key in enumerate(weight_dict[key].keys()):
    #         weight_dict[key][sub_key] = normalized_weights[i]
    #     # Check the sum of weights is approximately 1
    #     if abs(1 - sum(weight_dict[key].values())) > 1e-8:
    #         print(f"Error: Node {key} 归一化后的权重和不为1")
    # ----------------------------------------------------------------------------------------

    # ----------------------------------若不用softmax------------------------------------------------------
    # # 为同个节点的不同阶节点的权重进行归一化 输出：weight_dict
    weight_sum_dict = {}
    for key in weight_dict.keys():
        weight_sum_dict[key] = sum(weight_dict[key].values())
        for sub_key in weight_dict[key].keys():
            weight_dict[key][sub_key] /= weight_sum_dict[key]
            # check the sum of weight is 1
        if abs(1 - sum(weight_dict[key].values())) > 1e-8:
            print(f"Error：Node{key}归一化后的权重和不为1")
    # ----------------------------------------------------------------------------------------

    # 将每个节点的嵌入向量与权重相乘
    embedding_dict = {}
    for key in weight_dict.keys():
        embedding_dict[key] = np.zeros(dim)
        for sub_key in weight_dict[key].keys():
            embedding_dict[key] += np.array(hon_embedding_dict[sub_key]) * weight_dict[key][sub_key]


    # args.methods.vectors = embedding_dict
    print("Saving embeddings...")
    if output_filename == 'none':
        # 将embedding_dict的键转为str类型
        embedding_dict = {str(key): value for key, value in embedding_dict.items()}
        return embedding_dict
    else:
        # 将最终的embedding_dict写入文件
        fout = open(output_filename, 'w')
        node_num = len(embedding_dict)
        fout.write("{} {}\n".format(node_num, dim))
        for node, vec in embedding_dict.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()

        # 将embedding_dict的键转为str类型
        embedding_dict = {str(key): value for key, value in embedding_dict.items()}

        return embedding_dict

    # 输出前2行：于比较aggregate前后的embedding
    # with open(output_filename, 'r') as f:
    #     print("\nAfter aggregation:\n")
    #     for i in range(2):
    #         line = f.readline()
    #         print(line)

    # 用于调试
    # print('\nAggregation done')

# Aggregate(dim, embedding_filename, edgelist_filename)

def AggregateAvg(embeddings, graph):
    # 将前置节点（即‘|’前元素相同）相同的变阶节点的embedding平均聚合为一个节点
    print("Aggregating...")

    embedding_dict = defaultdict(float)
    counter = defaultdict(int)

    for node in graph.nodes():
        key = int(node.split('|')[0])
        embedding_dict[key] += embeddings[node]
        counter[key] += 1

    for key in embedding_dict.keys():
        embedding_dict[key] /= counter[key]
    
    return embedding_dict



