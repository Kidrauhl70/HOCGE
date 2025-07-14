import numpy as np
import networkx as nx
from collections import defaultdict


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def Aggregate(dim, embeddings, graph, output_filename, use_avg=False):
    print("Aggregating embeddings...")

    # Compute combined edge weights from predecessors and successors
    node_weight_pred = {node: sum(attr['weight'] for attr in preds.values())
                        for node, preds in graph._pred.items()}
    node_weight_succ = {node: sum(attr['weight'] for attr in succs.values())
                        for node, succs in graph._succ.items()}

    node_total_weight = {}
    for node in node_weight_pred:
        node_total_weight[node] = node_weight_pred[node] + node_weight_succ.get(node, 0)

    # Build weight dict: {lower_node: {higher_node: weight}}
    weight_dict = {}
    for high_node in node_total_weight:
        low_node = int(high_node.split('|')[0])

        weight = node_total_weight[high_node]

        if low_node not in weight_dict:
            weight_dict[low_node] = {}
        weight_dict[low_node][high_node] = 1 if use_avg else weight

    print(f"Total grouped nodes: {sum(len(v) for v in weight_dict.values())}")

    # Normalize weights for each group
    for node in weight_dict:
        total = sum(weight_dict[node].values())
        for sub_node in weight_dict[node]:
            weight_dict[node][sub_node] /= total
        if abs(1 - sum(weight_dict[node].values())) > 1e-6:
            print(f"Warning: Weights for node {node} not normalized.")

    # Aggregate embeddings
    aggregated = {}
    for node in weight_dict:
        aggregated[node] = np.zeros(dim)
        for sub_node, weight in weight_dict[node].items():
            aggregated[node] += embeddings[sub_node] * weight

    print("Saving embeddings...")
    if output_filename:
        with open(output_filename, 'w') as f:
            f.write(f"{len(aggregated)} {dim}\n")
            for node, vec in aggregated.items():
                vec_str = ' '.join(map(str, vec))
                f.write(f"{node} {vec_str}\n")
    return {str(k): v for k, v in aggregated.items()}


