def FindEdges(graph_split, filename_hon, type):
     hon_edges = set()

     # 读取高阶节点边
     with open(filename_hon, 'r') as f:
          for line in f:
               edge = line.strip().split()
               hon_edges.add((edge[0], edge[1], edge[2]))

     # 将 graph_split.edges 转换为字典以便更快地进行查找
     graph_split_dict = {edge: True for edge in graph_split.edges()}

     # 查找匹配的高阶节点边
     matched_edges = set()
     for hon_edge in hon_edges:
          if (int(hon_edge[0].split('|')[0]), int(hon_edge[1].split('|')[0])) in graph_split_dict:
               matched_edges.add(hon_edge)

     # 判断训练集还是测试集，True为训练集，False为测试集
     if type:
          matched_edges_filename = filename_hon.split('.')[0] + '_matched_train.txt'
     else:
          matched_edges_filename = filename_hon.split('.')[0] + '_matched_test.txt'

     # 将matched_edges写入txt文件，用空格分隔    
     with open(matched_edges_filename, 'w') as f:
          for edge in matched_edges:
               f.write(edge[0] + ' ' + edge[1] + ' ' + edge[2] + '\n')

     return matched_edges_filename