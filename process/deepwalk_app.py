import time

from ge import DeepWalk
import networkx as nx
import argparse
import pickle


def parse_args():
    """
    Parses the node2vec arguments.
    """
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--input", type=str, default='../data/app_w.edgelist')
    parser.add_argument("--output", type=str, default='../output/emb')

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    G = nx.read_edgelist(args.input, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    t2 = time.time()
    print("read_edgelist cost time:",t2-t1)

    model = DeepWalk(G, walk_length=10, num_walks=80, workers=4)
    t3 = time.time()
    print("init DeepWalk cost time:", t3 - t2)

    t4 = time.time()
    model.train(window_size=5, iter=3)
    t5 = time.time()
    print("train cost time:", t5 - t4)

    embeddings = model.get_embeddings()
    t6 = time.time()
    print("get embeddings,length :%d, cost time:%d"%(len(embeddings), t6 - t5))

    with open(args.output, 'wb') as f:
        pickle.dump(embeddings, f)
    print(embeddings)

    print("work done, cost time", time.time()-t6)

