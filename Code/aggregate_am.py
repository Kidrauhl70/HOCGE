import numpy as np
import networkx as nx
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Dense

# dim = 64

# embedding_filename = "/home/guozhi/桌面/HON_Embedding/OpenNE-master/data/network-Disorder-2000-lines-embeddings.txt"
# edgelist_filename = "/home/guozhi/桌面/HON_Embedding/OpenNE-master/data/network-Disorder-2000-lines-manual.txt"

# 定义自注意力层
class SelfAttentionLayer(Layer):
    def __init__(self, dim, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.W_q = self.add_weight("W_q", shape=(self.dim, self.dim), initializer="random_normal", trainable=True)
        self.W_k = self.add_weight("W_k", shape=(self.dim, self.dim), initializer="random_normal", trainable=True)
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, x):
        q = tf.matmul(x, self.W_q)
        k = tf.matmul(x, self.W_k)

        # 计算注意力权重
        attention_weights = tf.matmul(q, k, transpose_b=True)

        # 使用 softmax 归一化注意力权重
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        # 使用注意力权重对嵌入进行加权聚合
        output = tf.matmul(attention_weights, x)
        return output

def Aggregate_am(dim, embeddings, graph, output_filename, type):
    
    print("Aggregating with self-attention...")

    hon_embedding_dict = embeddings

    # 设置字典 {一阶节点：对应的变阶节点}
    node_dict = defaultdict(list)
    for node in graph.nodes():
        key = int(node.split('|')[0])
        node_dict[key].append(node)

    # 创建自注意力层
    self_attention_layer = SelfAttentionLayer(dim)

    # 如果该节点只以一阶形式存在，则该节点的最终嵌入向量为该节点的嵌入向量
    embedding_dict = {}
    for key in node_dict.keys():
        if len(node_dict[key]) == 1:
            embedding_dict[key] = hon_embedding_dict[str(key)]
        else:
            embeddings_to_agg = [hon_embedding_dict[higher_node] for higher_node in node_dict[key]]
            aggregated_embedding = self_attention_layer(tf.convert_to_tensor(embeddings_to_agg, dtype=tf.float32))
            # 成一个随机数，范围在(0,len(embeddings)-1)
            # random_index = np.random.randint(0, len(embeddings_to_agg))
            # embedding_dict[key] = aggregated_embedding.numpy()[random_index]
            # embedding_dict[key] = aggregated_embedding.numpy()[-1]
            embedding_dict[key] = aggregated_embedding.numpy()[0]

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




