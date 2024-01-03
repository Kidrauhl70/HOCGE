from __future__ import print_function
import time
from gensim.models import Word2Vec
from walker import *
from contrastivelearning import ContrastiveLearning


class Node2vec(object):

    def __init__(self, graph, path_length, num_paths, dim, p=1.0, q=1.0, dw=False, **kwargs):

        kwargs["workers"] = kwargs.get("workers", 8) # 并行的线程数 要在配置里更改
        if dw:
            kwargs["hs"] = 1
            p = 1.0  # only for node2vec
            q = 1.0  # only for node2vec

        self.graph = graph
        if dw:
            self.walker = BasicWalker(graph, workers=8)  # deepWalk
        else:
            self.walker = Walker(
                graph, p=p, q=q, workers=kwargs["workers"])
            print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()

        sentences = self.walker.simulate_walks(
            num_walks=num_paths, walk_length=path_length)
        
        flag = False
        
        # 使用列表解析来过滤掉长度小于2的句子
        filtered_sentences = [sentence for sentence in sentences if len(sentence) >= 1]
                
        # 将sentences中元素的类型从int转换为str
        if type(filtered_sentences[0][0]) == int:
            flag = True
            for i in range(len(filtered_sentences)):
                filtered_sentences[i] = [str(item) for item in filtered_sentences[i]]

        sentences = filtered_sentences

        del filtered_sentences
        
        kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = kwargs.get("vector_size", dim)  # 若要改成更高级的环境，size需要改成vector_size
        kwargs["sg"] = 1

        # self.walks = sentences

        self.size = kwargs["vector_size"]  # 若要改成更高级的环境，size需要改成vector_size
        print("Learning representation...")
        word2vec = Word2Vec(**kwargs)
        self.vectors = {}
        if flag:
            for word in graph.nodes():
                try:
                    self.vectors[word] = word2vec.wv[str(word)]
                # 如果遇到key error，则index为str型
                except KeyError:
                    self.vectors[word] = word2vec.wv[word]
                
        else:
            for word in graph.nodes():
                self.vectors[word] = word2vec.wv[word]             
        del word2vec

        # self.vectors_original = self.vectors
        self.walks_path = sentences

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
