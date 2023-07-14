import numpy as np

def PathPrediction(embeddings, test_path_length, dim):
    path_filename = "/home/guozhi/桌面/HON_Embedding/hon-master/tests/Disorder-2000-lines.csv"
    
    # dim = 64
    # test_path_length = 3

    data = np.loadtxt(path_filename, delimiter=" ", dtype=int)
    # 去掉第一列
    path_array = data[:, 1:]
    rows, cols = path_array.shape  

    test_path = np.zeros((rows,test_path_length), dtype=int)
    train_last_node = np.zeros(rows, dtype=int)
    last_node_embedding = np.zeros((rows, dim), dtype=float)  
    
    for i, path in enumerate(path_array):
        test_path[i] = path[-test_path_length:]
        train_last_node[i] = path[-(1 + test_path_length)]

    num_of_correct = 0
    # 搜索与train_last_node的embedding最相似的节点(除了train_last_node本身)
    for i in range(rows):
        similarity = {}    
        last_node_embedding = embeddings[train_last_node[i]]
        for key in embeddings.keys():
            if key != train_last_node[i]:
                similarity[key] = np.dot(embeddings[key], last_node_embedding) / (np.linalg.norm(embeddings[key]) * np.linalg.norm(last_node_embedding))
        similarity = sorted(similarity.items(), key=lambda item:item[1], reverse=True)
        next_node = similarity[0][0]

        # print(f"predicted next node: {next_node}")
        # print(f"true next node:{test_path[i][0]}")
        if next_node == test_path[i][0]:
            num_of_correct = num_of_correct + 1
    print(f"path prediction accuracy: {num_of_correct / rows}")


    # return next_node
