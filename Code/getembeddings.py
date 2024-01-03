from __future__ import print_function
import numpy as np
import os
import ast
from sklearn.linear_model import LogisticRegression
from graph import *
from node2vec import *
from classify import Classifier, read_node_label
from line import *
from tadw import *
from lle import *
from hope import *
from lap import *
from gf import *
from sdne import *
from grarep import GraRep
from aggregate import Aggregate
from contrastivelearning import ContrastiveLearning
from walker import *
from filedirect import FileDirect
from aggregate_am import Aggregate_am
from aggregate_am_rand import Aggregate_am_rand
# from visualization import Visualization

def GetEmbeddings(args, num_neg_samples, num_epochs, counter, walk_length, window_size, pos_prob, neg_prob, number_walks):
# def GetEmbeddings(args, num_neg_samples, num_epochs, counter):

     g = Graph()
     dataset = args.input
     args.input, fon_filename, args.label_file = FileDirect(args)
     if args.task == 1:
          if counter == -10:
               print("Training for link prediction...")
               # args.input = f'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/{dataset}2_matched_train.txt'  
          else:
               print("Testing for link prediction...")
               # args.input = f'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/{dataset}2_matched_test.txt'  

     if not args.output:
          # output路径为input路径下的Results文件夹 #这里要改
          args.output = os.path.join(os.path.dirname(args.input), 'Results/HOCGE')

     print("Reading...")

     if args.graph_format == 'edgelist':
          g.read_edgelist(filename=args.input, weighted=args.weighted,
                         directed=args.directed)
     elif args.graph_format == 'adjlist':
          g.read_adjlist(filename=args.input)

     if args.method == 'node2vec':
          # model = Node2vec(graph=g.G, path_length=args.walk_length,
          #                          num_paths=args.number_walks, dim=args.representation_size,
          #                          workers=args.workers, p=args.p, q=args.q, window=args.window_size)
          model = Node2vec(graph=g.G, path_length=walk_length,
                                    num_paths=number_walks, dim=args.representation_size,
                                    workers=args.workers,  p=args.p, q=args.q, window=window_size, dw=True)
          
     # elif args.method == 'line':
     #      if args.label_file and not args.no_auto_save:
     #          model = LINE(g, epoch=args.epochs, rep_size=args.representation_size, order=args.order,
     #                            label_file=args.label_file, clf_ratio=args.clf_ratio)
     #      else:
     #          model = LINE(g, epoch=args.epochs,
     #                            rep_size=args.representation_size, order=args.order)
     elif args.method == 'deepWalk': # 这里
          # model = Node2vec(graph=g.G, path_length=args.walk_length,
          #                          num_paths=args.number_walks, dim=args.representation_size,
          #                          workers=args.workers, window=args.window_size, dw=True)
          model = Node2vec(graph=g.G, path_length=walk_length,
                                    num_paths=number_walks, dim=args.representation_size,
                                    workers=args.workers, window=window_size, dw=True)
     # elif args.method == 'tadw':
     #      assert args.label_file != ''
     #      assert args.feature_file != ''
     #      g.read_node_label(args.label_file)
     #      g.read_node_features(args.feature_file)
     #      model = TADW(
     #          graph=g, dim=args.representation_size, lamb=args.lamb)
     # elif args.method == 'grarep':
     #     model = GraRep(graph=g, Kstep=args.kstep, dim=args.representation_size)
     # elif args.method == 'lle':
     #     model = LLE(graph=g, d=args.representation_size)
     # elif args.method == 'hope':
     #     model = HOPE(graph=g, d=args.representation_size)
     # elif args.method == 'sdne':
     #     encoder_layer_list = ast.literal_eval(args.encoder_list)
     #     model = SDNE(g, encoder_layer_list=encoder_layer_list,
     #                       alpha=args.alpha, beta=args.beta, nu1=args.nu1, nu2=args.nu2,
     #                       batch_size=args.bs, epoch=args.epochs, learning_rate=args.lr)
     # elif args.method == 'lap':
     #     model = LaplacianEigenmaps(g, rep_size=args.representation_size)
     # elif args.method == 'gf':
     #     model = GraphFactorization(g, rep_size=args.representation_size,
     #                                   epoch=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay)

     # 对比学习（在聚合前）
     # num_neg_samples = 1  # 每个正例对应的负例数
     # num_epochs = 30  # 对比学习的迭代次数
     # pos_prob = 0.3
     # neg_prob = 0.3
     model.vectors = ContrastiveLearning(num_neg_samples=num_neg_samples, num_epochs=num_epochs ,
                                        pos_prob=pos_prob, neg_prob=neg_prob,
                                        walks=model.walks_path, embedding=model.vectors, 
                                        dim=args.representation_size, graph=g.G)
     

     # 若是高阶网络，先保存中间结果，再聚合，输出两个文件
     if args.hon:
          # model.vectors = ContrastiveLearning(num_neg_samples=num_neg_samples, num_epochs=num_epochs ,
          #                               pos_prob=pos_prob, neg_prob=neg_prob,
          #                               walks=model.walks_path, embedding=model.vectors, 
          #                               dim=args.representation_size, graph=g.G)
          print("Saving processing embeddings...")
          # processing_embedding_filename  = args.output + '/' + (args.input.split('/')[-1]).split('.')[0] + f'_processing_{args.method}.txt'
          processing_embedding_filename  = f'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/Results/HOCGE/processing_{args.method}.txt'
          model.save_embeddings(processing_embedding_filename)
          # counter = 0
          output_embedding_filename = f'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/Results/HOCGE/processing_hon_{args.method}{counter}.txt'
          # output_embedding_filename = args.output + '/' + (args.input.split('/')[-1]).split('.')[0]  + f'_hon_{args.method}{counter}.txt'

          if args.avg:
               embedding_dict = Aggregate(dim=args.representation_size, embeddings=model.vectors, graph = g.G,
                    output_filename=output_embedding_filename, type=True)
          else:
               embedding_dict = Aggregate(dim=args.representation_size, embeddings=model.vectors, graph = g.G,
                    output_filename=output_embedding_filename, type=False)
               # embedding_dict1 = Aggregate(dim=args.representation_size, embeddings=model.vectors, graph = g.G,
               #      output_filename=output_embedding_filename, type=False)
               # embedding_dict2 = Aggregate_am(dim=args.representation_size, embeddings=model.vectors, graph = g.G,
               #      output_filename=output_embedding_filename, type=False)
               # embedding_dict3 = Aggregate_am_rand(dim=args.representation_size, embeddings=model.vectors, graph = g.G,
               #      output_filename=output_embedding_filename, type=False)

     # 若不是高阶网络，直接保存
     else:
          print("Saving embeddings...")
          # output_embedding_filename = args.output + '/' + (args.input.split('/')[-1]).split('.')[0]  + f'_fon_{args.method}{counter}.txt'
          output_embedding_filename = f'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/Results/processing_hon_{args.method}{counter}.txt'
          model.save_embeddings(output_embedding_filename)
          # embedding_dict将model.vectors转化为字典,key为int型
          embedding_dict = {}
          for key, value in model.vectors.items():
               embedding_dict[key.split('|')[0]] = value


     if args.label_file and args.task == 0:
          X, Y = read_node_label(args.label_file)
          print(f"Training classifier using {args.clf_ratio*100:.2f}% nodes...")

          clf = Classifier(vectors=embedding_dict, clf=LogisticRegression())  # 这个是最初的
          
          classification_result = clf.split_train_evaluate(X, Y, args.clf_ratio, seed=0)
          # # classification_result = clf.split_train_evaluate(X, Y, k=5, seed=0)  # k折交叉验证
          # clf1 = Classifier(vectors=embedding_dict1, clf=LogisticRegression())  
          # clf2 = Classifier(vectors=embedding_dict2, clf=LogisticRegression())  
          # clf3 = Classifier(vectors=embedding_dict3, clf=LogisticRegression())  
          
          # classification_result1 = clf1.split_train_evaluate(X, Y, args.clf_ratio, seed=0)
          # classification_result2 = clf2.split_train_evaluate(X, Y, args.clf_ratio, seed=0)
          # classification_result3 = clf3.split_train_evaluate(X, Y, args.clf_ratio, seed=0)

     if args.task == 0:
          return classification_result
          # return classification_result1, classification_result2, classification_result3
     else:
          return embedding_dict
