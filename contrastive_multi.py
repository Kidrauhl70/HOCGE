import numpy as np
import tensorflow as tf
import multiprocessing
import networkx as nx


class ContrastiveModel(tf.keras.Model):
    def __init__(self, num_nodes, embedding_dim):
        super(ContrastiveModel, self).__init__()
        self.embedding = tf.Variable(tf.random.normal([num_nodes, embedding_dim]))

    def call(self, node_pairs):
        return tf.nn.embedding_lookup(self.embedding, node_pairs)


def InfoNCELoss(positive_pairs, negative_pairs, model):
    pos_node1, pos_node2 = zip(*positive_pairs)
    neg_node1, neg_node2 = zip(*negative_pairs)

    pos_embed1 = model(tf.convert_to_tensor(pos_node1))
    pos_embed2 = model(tf.convert_to_tensor(pos_node2))
    neg_embed1 = model(tf.convert_to_tensor(neg_node1))
    neg_embed2 = model(tf.convert_to_tensor(neg_node2))

    pos_sim = tf.reduce_sum(pos_embed1 * pos_embed2, axis=1)
    neg_sim = tf.reduce_sum(neg_embed1 * neg_embed2, axis=1)

    logits = tf.concat([pos_sim, neg_sim], axis=0)
    labels = tf.concat([
        tf.ones_like(pos_sim, dtype=tf.int32),
        tf.zeros_like(neg_sim, dtype=tf.int32)
    ], axis=0)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return loss_fn(labels, logits)


def construct_negative_pairs(args):
    graph, pos_pair, nodes, edgelist, num_neg_samples = args
    node1 = pos_pair[0]
    neg_pairs = []
    candidates = list(set(nodes) - set(graph.neighbors(node1)))
    for _ in range(num_neg_samples):
        neg_node = np.random.choice(candidates)
        while (neg_node, node1) in edgelist:
            neg_node = np.random.choice(candidates)
        neg_pairs.append((node1, neg_node))
    return neg_pairs


def parallel_construct_negative_pairs(graph, pos_pairs, nodes, edgelist, num_neg_samples):
    args_list = [(graph, pair, nodes, edgelist, num_neg_samples) for pair in pos_pairs]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(construct_negative_pairs, args_list)
    all_neg_pairs = [pair for sublist in results for pair in sublist]
    return all_neg_pairs


def ContrastiveLearningMul(num_neg_samples, num_epochs, pos_prob, neg_prob,
                           walks, embedding, dim, graph_hon, graph_fon):

    print("Starting contrastive learning...")

    nodes = list(graph_hon.nodes())
    num_nodes = len(nodes)
    edgelist = list(graph_hon.edges())
    id_to_index = {node_id: idx for idx, node_id in enumerate(nodes)}

    # Positive pairs from HON
    pos_pair1 = []
    for walk in walks:
        for i, node in enumerate(walk[1:], start=1):
            if '|' in node:
                pos_pair1.append((walk[i], walk[i - 1]))
    pos_pair1 = list(set(pos_pair1))
    print(f"Positive pairs (HON): {len(pos_pair1)}")

    neg_pair1 = parallel_construct_negative_pairs(graph_hon, pos_pair1, nodes, edgelist, num_neg_samples)
    neg_pair1 = [pair for pair in neg_pair1 if np.random.random() > neg_prob]
    neg_pair1 = list(set(neg_pair1))

    # Positive pairs from FON (based on co-pred/succ of same lower-order node)
    ho_nodes_in_fo = list(set([node.split('|')[0] for node in nodes if '|' in node]))
    print(f"First-order nodes: {len(ho_nodes_in_fo)}")

    pos_pair2_pred = []
    pos_pair2_succ = []

    for node in ho_nodes_in_fo:
        node = str(node)
        try:
            preds = [(p, graph_fon[p][node]['weight']) for p in graph_fon.predecessors(node)]
            preds.sort(key=lambda x: x[1], reverse=True)
            for i in range(0, int(len(preds) * 0.3), 2):
                if i + 1 < len(preds):
                    pos_pair2_pred.append((preds[i][0], preds[i + 1][0]))
        except KeyError:
            continue

        try:
            succs = [(s, graph_fon[s][node]['weight']) for s in graph_fon.successors(node)]
            succs.sort(key=lambda x: x[1], reverse=True)
            for i in range(0, int(len(succs) * 0.5), 2):
                if i + 1 < len(succs):
                    pos_pair2_succ.append((succs[i][0], succs[i + 1][0]))
        except KeyError:
            continue

    pos_pair2 = list(set(pos_pair2_pred + pos_pair2_succ))
    pos_pair2 = [pair for pair in pos_pair2 if np.random.random() > pos_prob]
    print(f"Positive pairs (FON): {len(pos_pair2)}")

    neg_pair2 = parallel_construct_negative_pairs(graph_fon, pos_pair2, nodes, edgelist, num_neg_samples)
    neg_pair2 = [pair for pair in neg_pair2 if np.random.random() > neg_prob]
    neg_pair2 = list(set(neg_pair2))

    # Convert to integer indices
    pos_pair1 = [(id_to_index[a], id_to_index[b]) for a, b in pos_pair1]
    neg_pair1 = [(id_to_index[a], id_to_index[b]) for a, b in neg_pair1]
    pos_pair2 = [(id_to_index[a], id_to_index[b]) for a, b in pos_pair2]
    neg_pair2 = [(id_to_index[a], id_to_index[b]) for a, b in neg_pair2]
    embedding = {id_to_index[k]: v for k, v in embedding.items()}

    model = ContrastiveModel(num_nodes, dim)
    model.embedding.assign(np.array([embedding[i] for i in range(num_nodes)]))
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            loss1 = InfoNCELoss(pos_pair1, neg_pair1, model)
            loss2 = InfoNCELoss(pos_pair2, neg_pair2, model)
            loss = loss1 + 0.5 * loss2
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"Epoch {epoch+1}/{num_epochs}: loss = {loss:.4f}")

    node_embeddings = model(tf.convert_to_tensor(list(range(num_nodes))))
    index_to_id = {idx: node_id for node_id, idx in id_to_index.items()}
    node_embeddings = {index_to_id[i]: np.array(vec) for i, vec in enumerate(node_embeddings)}

    print(f"pos_prob: {pos_prob}, neg_prob: {neg_prob}")
    return node_embeddings
