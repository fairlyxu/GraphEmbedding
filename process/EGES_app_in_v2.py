import sys
import os
sys.path.append("..")
from ge import DeepWalk
import networkx as nx
import pickle
import traceback
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import argparse
from utils.util import graph_context_batch_iter
from utils.data_process import get_graph_data
from sklearn.preprocessing import LabelEncoder
from utils.features import FeaturesProcess
from models import EGES

def parse_args():
    """
    Parses the node2vec arguments.
    """
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_sampled", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--input", type=str, default='../data/g_sample_eges.txt')
    parser.add_argument("--output", type=str, default='../output/EGES.embed')
    parser.add_argument("--item_feat", type=str, default='../data/item_v2.dat')
    parser.add_argument("--features_label_file", type=str, default='../output/features_label_v2.pkl')
    parser.add_argument("--model_out_dir", type=str, default='./checkpoints_v2/EGES')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_out_dir = args.model_out_dir
    item_feat_name = ['d_st_2_did', 'category2', 'category3', 'from_type','gp_score', 'lan', 'is_pay', 'is_aha_combined','is_offer']

    len_item = 100000
    feature_lens = [len_item, 3, 100, 10, 10,10, 3, 3, 4]
    t1 = time.time()
    # item 特征
    item_df = pd.read_csv(args.item_feat, sep='\t', names = item_feat_name)
    values = {'is_offer': 0}
    item_df.fillna(value=values, inplace=True)
    item_df['is_offer'] = item_df['is_offer'].astype(np.int)
    item_df = item_df[item_feat_name]
    item_df = item_df.infer_objects()

    fp = FeaturesProcess(args.features_label_file)

    # id2index
    for feat in item_feat_name:
        item_df[feat] = item_df[feat].map(lambda x: fp.fit_transform(feat, x))

    item_df = item_df.sort_values(by=['d_st_2_did'], ascending=True)


    data = pd.read_csv(args.input,delimiter='\t', header= None,names=['ii1','ii2','w'])#.iloc[0:1000]
    data = data.infer_objects()
    data['ii1'] = data['ii1'].astype(np.int)
    data['ii2'] = data['ii2'].astype(np.int)

    print("data:", data['ii1'].dtypes,data['ii2'].dtypes)
    try:
        for feat in ['ii1','ii2']:
            data[feat] = data[feat].map(lambda x: fp.transform('d_st_2_did', x))
            data[feat] = data[feat].astype(np.int)
    except:
        traceback.print_exc()

    fp.save_feature_file()

    node_file = '../data/tmp_app_e.edgelist'
    data.to_csv(node_file, sep='\t', index=False,header=0)

    print("sample data:",data.head())

    G = nx.read_edgelist(node_file, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    t2 = time.time()
    print("read_edgelist cost time:",t2-t1)
    model = DeepWalk(G, walk_length=10, num_walks=80, workers=4)
    all_pairs = get_graph_data(model.sentences)
    print("graph: ", all_pairs.shape)
    print("init DeepWalk cost time:", time.time() - t2)

    print("feature_lens:", feature_lens)

    EGES = EGES(len_item, len(item_feat_name), feature_lens, n_sampled=args.n_sampled, embedding_dim=args.embedding_dim, lr=args.lr)
    # init model
    print('init model...')
    start_time = time.time()
    init = tf.global_variables_initializer()
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    sess = tf.Session(config=config_tf)
    sess.run(init)
    saver = tf.compat.v1.train.Saver()

    if os.path.exists(model_out_dir) :
        saver.restore(sess, model_out_dir)
    end_time = time.time()
    print('time consumed for init: %.2f' % (end_time - start_time))
    print_every_k_iterations = 100
    loss = 0
    iteration = 0
    start = time.time()
    max_iter =  len(all_pairs)//args.batch_size*args.epochs
    for iter in range(max_iter):
        iteration += 1
        batch_features, batch_labels = next(graph_context_batch_iter(all_pairs, args.batch_size, item_df.values, len(item_feat_name)))

        feed_dict = {input_col: batch_features[:, i] for i, input_col in enumerate(EGES.inputs[:-1])}
        feed_dict[EGES.inputs[-1]] = batch_labels
        _, train_loss = sess.run([EGES.train_op, EGES.cost], feed_dict=feed_dict)

        loss += train_loss

        if iteration % print_every_k_iterations == 0:
            end = time.time()
            e = iteration*args.batch_size//len(all_pairs)
            print("Epoch {}/{}".format(e, args.epochs),
                  "Iteration: {}".format(iteration),
                  "Avg. Training loss: {:.4f}".format(loss / print_every_k_iterations),
                  "{:.4f} sec/batch".format((end - start) / print_every_k_iterations))
            loss = 0
            start = time.time()

    print('optimization finished...')

    saver.save(sess, model_out_dir)
    feed_dict_test = {}

    # predict and save
    try:
        feed_dict_test = {input_col: list(item_df.values[:, i]) for i, input_col in enumerate(EGES.inputs[:-1])}
        feed_dict_test[EGES.inputs[-1]] = np.zeros((len(item_df), 1), dtype=np.int32)
        embedding_result = sess.run(EGES.merge_emb, feed_dict=feed_dict_test)
        print("embedding_result:", embedding_result.shape)

        print('saving embedding result...')
        res_emb_dict = {}

        for i in range(len(embedding_result)):
            iid = fp.inverse_transform('d_st_2_did',i)
            res_emb_dict[iid] = embedding_result[i].tolist()

        with open(args.output, 'wb') as f:
            pickle.dump(res_emb_dict, f)

    except:
        traceback.print_exc()






