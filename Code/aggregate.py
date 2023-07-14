import numpy as np
import networkx as nx

# dim = 64

# embedding_filename = "/home/guozhi/桌面/HON_Embedding/OpenNE-master/data/network-Disorder-2000-lines-embeddings.txt"
# edgelist_filename = "/home/guozhi/桌面/HON_Embedding/OpenNE-master/data/network-Disorder-2000-lines-manual.txt"

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



def node_distance(G, low_order_node, high_order_node):
    # 计算低阶节点与高阶节点之间的最短路径长度
    shortest_path_length = nx.shortest_path_length(G, source=low_order_node, target=high_order_node)
    # 将最短路径长度转换为权重
    weight = 1 / shortest_path_length
    # 计算低阶节点对于所有高阶节点的权重
    total_weight = sum([1 / nx.shortest_path_length(G, source=low_order_node, target=x) for x in G.nodes() if x != low_order_node])
    # 计算该低阶节点对于该高阶节点的权重
    node_weight = weight / total_weight
    return node_weight

def Aggregate(dim, embedding_filename, edgelist_filename, output_filename):
    
    print("Aggregating...")

    # 得到每个节点的embedding {节点：嵌入向量}
    hon_embedding_dict = {}
    i = 0
    with open(embedding_filename, 'r') as f:
        next(f)  # 跳过第一行
        for line in f:
            # 输出前2行：用于比较aggregate前后的embedding
            # if i < 2:
            #     print(line)
            #     i += 1  
            row = line.split()
            key = row[0]
            values = [float(x) for x in row[1:]]
            hon_embedding_dict[key] = values

    # 得到每个节点的权重 {节点：{变阶节点：权重}}
    weight_dict = edge_weight(edgelist_filename)

    # 将weight_dict的每个权重都设置为1 （测试avg）
    for key in weight_dict.keys():
        for sub_key in weight_dict[key].keys():
            weight_dict[key][sub_key] = 1


    # 用于确认weight_dict的key是否包含所有节点
    node_size = 0
    for key in weight_dict.keys():
        node_size += len(weight_dict[key])
    print(f'node_size: {node_size}')

    # 为同个节点的不同阶节点的权重进行归一化 输出：weight_dict
    weight_sum_dict = {}
    for key in weight_dict.keys():
        weight_sum_dict[key] = sum(weight_dict[key].values())
        for sub_key in weight_dict[key].keys():
            weight_dict[key][sub_key] /= weight_sum_dict[key]
            # check the sum of weight is 1
        if abs(1 - sum(weight_dict[key].values())) > 1e-8:
            print(f"Error：Node{key}归一化后的权重和不为1")

    # # 用于调试
    # print('done')

    # 将每个节点的嵌入向量与权重相乘
    embedding_dict = {}
    for key in weight_dict.keys():
        embedding_dict[key] = np.zeros(dim)
        for sub_key in weight_dict[key].keys():
            embedding_dict[key] += np.array(hon_embedding_dict[sub_key]) * weight_dict[key][sub_key]

    # args.methods.vectors = embedding_dict
    print("Saving embeddings...")

    # 将最终的embedding_dict写入文件
    fout = open(output_filename, 'w')
    node_num = len(embedding_dict)
    fout.write("{} {}\n".format(node_num, dim))
    for node, vec in embedding_dict.items():
        fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
    fout.close()

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
