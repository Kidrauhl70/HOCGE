
# 将str类型的轨迹数据文件转化成int类型的轨迹数据文件
def DataPreprocessing(mapping_filename, trajectory_filename, converted_trajectory_filename):
    # 读取编号-节点(str)映射文件
     node_mapping_dict = {}
     with open (mapping_filename, 'r') as f:
          for line in f:
               row = line.strip().split(',')
               node_mapping_dict[int(row[0])] = row[1]
     
     with open(trajectory_filename, 'r') as fin, open(converted_trajectory_filename, 'w') as fout:
        for line in fin:
            nodes = line.strip().split(',')
            node_ids = [str(list(node_mapping_dict.keys())[list(node_mapping_dict .values()).index(node)]) for node in nodes]
            fout.write(' '.join(node_ids) + '\n')

mapping_filename = '/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/wiki_mapping.csv'
trajectory_filename = '/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/wikipedia_clickstreams.ngram'
converted_trajectory_filename = '/home/guozhi/桌面/HON_Embedding/MyCode/data/dat_dge/wikipedia_clickstreams_converted.txt'

DataPreprocessing(mapping_filename, trajectory_filename, converted_trajectory_filename)