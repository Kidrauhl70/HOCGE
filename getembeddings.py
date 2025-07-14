from __future__ import print_function
import numpy as np
import os
import networkx as nx
from sklearn.linear_model import LogisticRegression

from graph import Graph
from node2vec import Node2vec
from classify import Classifier, read_node_label
from aggregate import Aggregate
from contrastive_multi import ContrastiveLearningMul
from filedirect import FileDirect

def GetEmbeddings(args, num_neg_samples, num_epochs, counter, walk_length, window_size, pos_prob, neg_prob, number_walks):
    g = Graph()
    g_fon = Graph()

    dataset = args.input
    args.input, fon_filename, args.label_file = FileDirect(counter, args)

    if not args.output:
        args.output = os.path.join(os.path.dirname(args.input), 'Results', 'HOCGE')

    print("Reading graph...")

    if args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input, weighted=args.weighted, directed=args.directed)
        g_fon.read_edgelist(filename=fon_filename, weighted=args.weighted, directed=args.directed)
    elif args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)

    model = Node2vec(
        graph=g.G,
        path_length=walk_length,
        num_paths=number_walks,
        dim=args.representation_size,
        workers=args.workers,
        window=window_size,
        dw=True
    )

    print(nx.info(g.G), nx.info(g_fon.G))

    model.vectors = ContrastiveLearningMul(
        num_neg_samples=num_neg_samples,
        num_epochs=num_epochs,
        pos_prob=pos_prob,
        neg_prob=neg_prob,
        walks=model.walks_path,
        embedding=model.vectors,
        dim=args.representation_size,
        graph_hon=g.G,
        graph_fon=g_fon.G
    )

    if args.hon:
        filename_prefix = os.path.splitext(os.path.basename(args.input))[0]
        output_file = os.path.join(args.output, f"{filename_prefix}_hon_{args.method}{counter}.txt")

        if args.avg:
            embedding_dict = Aggregate(
                dim=args.representation_size,
                embeddings=model.vectors,
                graph=g.G,
                output_filename=output_file,
                type=True
            )
        else:
            embedding_dict = Aggregate(
                dim=args.representation_size,
                embeddings=model.vectors,
                graph=g.G,
                output_filename=None,
                type=False
            )
    else:
        print("Saving embeddings...")
        filename_prefix = os.path.splitext(os.path.basename(args.input))[0]
        output_file = os.path.join(args.output, f"{filename_prefix}_fon_{args.method}{counter}.txt")
        model.save_embeddings(output_file)
        embedding_dict = {key.split('|')[0]: value for key, value in model.vectors.items()}

    if args.label_file and args.task == 0:
        X, Y = read_node_label(args.label_file)
        print(f"Training classifier using {args.clf_ratio * 100:.2f}% of nodes...")
        clf = Classifier(vectors=embedding_dict, clf=LogisticRegression())
        return clf.split_train_evaluate(X, Y, args.clf_ratio, seed=0)

    return embedding_dict
