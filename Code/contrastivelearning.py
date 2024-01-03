import numpy as np
import tensorflow as tf
# import random
import multiprocessing
# from concurrent.futures import ThreadPoolExecutor


# 定义对比学习模型
class ContrastiveModel(tf.keras.Model):
     def __init__(self, num_nodes, embedding_dim):
          super(ContrastiveModel, self).__init__()
          # self.embedding = tf.keras.layers.Embedding(num_nodes, embedding_dim)
          self.embedding = tf.Variable(tf.random.normal([num_nodes, embedding_dim]))

     def call(self, node_pairs):
        # 获取节点嵌入向量
     #    embeddings = self.embedding(node_pairs)
          # embeddings = tf.nn.embedding_lookup(self.embedding, node_pairs)
          embeddings = tf.nn.embedding_lookup(self.embedding, node_pairs)
          return embeddings

# 定义对比学习损失函数
def InfoNCELoss(positive_pairs, negative_pairs, model):
     # 正例对和负例对的节点ID
     pos_node1, pos_node2 = zip(*positive_pairs)
     neg_node1, neg_node2 = zip(*negative_pairs)

     # 正例对和负例对的嵌入向量
     pos_embed1 = model(tf.convert_to_tensor(pos_node1))
     pos_embed2 = model(tf.convert_to_tensor(pos_node2))
     neg_embed1 = model(tf.convert_to_tensor(neg_node1))
     neg_embed2 = model(tf.convert_to_tensor(neg_node2))

     # 计算正例对和负例对的内积
     pos_sim = tf.reduce_sum(pos_embed1 * pos_embed2, axis=1)
     neg_sim = tf.reduce_sum(neg_embed1 * neg_embed2, axis=1)

     # 构造对比学习损失（InfoNCE损失函数）
     logits = tf.concat([pos_sim, neg_sim], axis=0)
     # labels = tf.concat([tf.ones_like(pos_sim), tf.zeros_like(neg_sim)], axis=0)

     labels = tf.concat([tf.ones_like(pos_sim, dtype=tf.int32), tf.zeros_like(neg_sim, dtype=tf.int32)], axis=0)
     
     loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # the original one

     # labels = tf.concat([tf.ones_like(pos_sim, dtype=tf.int32), -tf.ones_like(neg_sim, dtype=tf.int32)], axis=0)
     # loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

     loss = loss_fn(labels, logits)

     return loss

# 定义构造负例对的辅助函数，用于分配任务
def construct_negative_pairs_helper(graph, pos_pairs, nodes, edgelist, num_neg_samples):
     args_list = [(graph, pos_pair, nodes, edgelist, num_neg_samples) for pos_pair in pos_pairs]
     return args_list

def construct_negative_pairs(args):
     graph, pos_pair, nodes, edgelist, num_neg_samples = args
     node1 = pos_pair[0]
     neg_pair = []
     for i in range(num_neg_samples):
          # neg_node选择范围为node1邻居外的节点
          nodes_selected = list(set(nodes) - set(graph.neighbors(node1)))  
          neg_node = np.random.choice(nodes_selected)
          while (neg_node, node1) in edgelist:
               neg_node = np.random.choice(list(nodes_selected))
          neg_pair.append((node1, neg_node))
     return neg_pair

def parallel_construct_negative_pairs(graph, pos_pairs, nodes, edgelist, num_neg_samples):
     args_list = construct_negative_pairs_helper(graph, pos_pairs, nodes, edgelist, num_neg_samples)
     
     # 创建一个进程池，设置进程池的大小为 CPU 的核心数
     num_processes = multiprocessing.cpu_count()
     pool = multiprocessing.Pool(processes=num_processes)

     # 使用 map 方法将任务分发给进程池中的多个进程并行处理
     results = pool.map(construct_negative_pairs, args_list)

     # 关闭进程池，阻止新的任务提交
     pool.close()

     # 等待所有进程完成
     pool.join()
     
     all_neg_pairs = []
     for result in results:
        all_neg_pairs.extend(result)
     return all_neg_pairs

def ContrastiveLearning(num_neg_samples,num_epochs, pos_prob, neg_prob, walks, embedding, dim, graph):
     
     print("Contrastive Learning...")

     pos_pairs = []
     neg_pairs = []

     nodes = list(graph.nodes())
     # print(f"number of nodes: {len(nodes)}")
     # 高阶节点为“|”分割的节点,仅有高阶节点参与对比学习
     # ho_nodes = [node for node in nodes if "|"  in node]
     num_nodes = len(nodes)
     # print(f"number of high-order nodes: {num_nodes}")
     edgelist = list(graph.edges())

     # 构建节点ID到整数索引的映射字典
     id_to_index = {node_id: index for index, node_id in enumerate(nodes)}

     # 对高阶节点构造正例对
     for walk in walks:
          # 当前节点是序列的第一个节点，无法与前一个节点构成正例对
          for i, node in enumerate(walk[1:], start=1):
               # 构造正例对，当前节点与前一个节点构成正例
               if "|" in node:
                    pos_pair = (walk[i], walk[i-1])
                    pos_pairs.append(pos_pair)
     pos_pairs = [pos_pair for pos_pair in pos_pairs if np.random.random() > pos_prob]
     pos_pairs = list(set(pos_pairs))

     # # 构造正例对
     # for walk in walks:
     #      # 当前节点是序列的第一个节点，无法与前一个节点构成正例对
     #      for i, node in enumerate(walk[1:], start=1):
     #           # 构造正例对，当前节点与前一个节点构成正例
     #           pos_pair = (walk[i], walk[i-1])
     #           pos_pairs.append(pos_pair)
     # # 概率抽样pos_pairs
     # pos_pairs = [pos_pair for pos_pair in pos_pairs if np.random.random() > pos_prob]
     # pos_pairs = list(set(pos_pairs))

     # 构造负例对
     # 多线程构造负例对
     neg_pairs = parallel_construct_negative_pairs(graph, pos_pairs, nodes, edgelist, num_neg_samples)
     
     # 概率抽样neg_pairs
     neg_pairs = [neg_pair for neg_pair in neg_pairs if np.random.random() > neg_prob]

     # check negative pairs 是否有重复的，若重复则删除
     neg_pairs = list(set(neg_pairs))

     # print("Done with positive and negative pairs...")

     # 将正例对和负例对的节点ID转换为整数索引
     pos_pairs = [(id_to_index[node1], id_to_index[node2]) for node1, node2 in pos_pairs]
     neg_pairs = [(id_to_index[node1], id_to_index[node2]) for node1, node2 in neg_pairs]

     # 将embedding的key转换为整数索引
     embedding = {id_to_index[node_id]: embedding[node_id] for node_id in embedding}

     # print("Done with converting node id to index...")

     model = ContrastiveModel(num_nodes, dim)

     model.embedding.assign(np.array(list(embedding.values())))

     optimizer = tf.keras.optimizers.Adam()

     for epoch in range(num_epochs):         

          with tf.GradientTape() as tape:
               loss = InfoNCELoss(pos_pairs, neg_pairs, model)

          gradients = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, model.trainable_variables))

          print(f"Epoch {epoch+1}/{num_epochs}: loss = {loss:.4f}")

     # node_embeddings = model(tf.convert_to_tensor(list(embedding.keys())))

     # 在构造节点嵌入时，使用整数索引代替节点ID
     node_embeddings = model(tf.convert_to_tensor(list(range(num_nodes))))

     # 构建节点索引到原始节点ID的逆映射
     index_to_id = {index: node_id for node_id, index in id_to_index.items()}

     # 将整数索引转换回原始的节点ID
     node_embeddings_ID = {index_to_id[index]: embedding for index, embedding in enumerate(node_embeddings)}

     # value转成array
     node_embeddings_ID = {node_id: np.array(embedding) for node_id, embedding in node_embeddings_ID.items()}
     
     # print(f"number of negative samples: {num_neg_samples}\nnum_epochs: {num_epochs}")
     print(f"pos_prob: {pos_prob}\nneg_prob: {neg_prob}")

     return node_embeddings_ID

