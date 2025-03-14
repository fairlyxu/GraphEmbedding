
import numpy as np



def get_graph_context_all_pairs(walks, window_size):
    all_pairs = []
    for k in range(len(walks)):
        for i in range(len(walks[k])):
            for j in range(i - window_size, i + window_size + 1):
                if i == j or j < 0 or j >= len(walks[k]):
                    continue
                else:
                    all_pairs.append([walks[k][i], walks[k][j]])
    return np.array(all_pairs, dtype=np.int32)

def get_graph_data(session_reproduce,window_size=5):
    session_reproduce = list(filter(lambda x: len(x) > 2, session_reproduce))

    # get pair
    all_pairs = get_graph_context_all_pairs(session_reproduce,  window_size)


    return all_pairs
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='manual to this script')
#     parser.add_argument("--data_path", type=str, default='./data/')
#     parser.add_argument("--p", type=float, default=0.25)
#     parser.add_argument("--q", type=float, default=2)
#     parser.add_argument("--num_walks", type=int, default=10)
#     parser.add_argument("--walk_length", type=int, default=10)
#     parser.add_argument("--window_size", type=int, default=5)
#     args = parser.parse_known_args()[0]
#
#     action_data = pd.read_csv(args.data_path + 'action_head.csv', parse_dates=['action_time']).drop('module_id', axis=1).dropna().iloc[0:10]
#     all_skus = action_data['sku_id'].unique()
#     all_skus = pd.DataFrame({'sku_id': list(all_skus)})
#     sku_lbe = LabelEncoder()
#     all_skus['sku_id'] = sku_lbe.fit_transform(all_skus['sku_id'])
#     action_data['sku_id'] = sku_lbe.transform(action_data['sku_id'])
#
#     print('make session list\n')
#     start_time = time.time()
#     session_list = get_session(action_data, use_type=[1, 2, 3, 5])
#     session_list_all = []
#     for item_list in session_list:
#         for session in item_list:
#             if len(session) > 1:
#                 session_list_all.append(session)
#
#     print('make session list done, time cost {0}'.format(str(time.time() - start_time)))
#
#     # session2graph
#     node_pair = dict()
#     for session in session_list_all:
#         for i in range(1, len(session)):
#             if (session[i - 1], session[i]) not in node_pair.keys():
#                 node_pair[(session[i - 1], session[i])] = 1
#             else:
#                 node_pair[(session[i - 1], session[i])] += 1
#
#     in_node_list = list(map(lambda x: x[0], list(node_pair.keys())))
#     out_node_list = list(map(lambda x: x[1], list(node_pair.keys())))
#     weight_list = list(node_pair.values())
#     graph_df = pd.DataFrame({'in_node': in_node_list, 'out_node': out_node_list, 'weight': weight_list})
#     graph_df.to_csv('./data_cache/graph.csv', sep=' ', index=False, header=False)
#
#     G = nx.read_edgelist('./data_cache/graph.csv', create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
#     walker = RandomWalker(G, p=args.p, q=args.q)
#     print("Preprocess transition probs...")
#     walker.preprocess_transition_probs()
#
#     session_reproduce = walker.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length, workers=4,
#                                               verbose=1)
#     session_reproduce = list(filter(lambda x: len(x) > 2, session_reproduce))
#
#     # add side info
#     product_data = pd.read_csv(args.data_path + 'jdata_product.csv').drop('market_time', axis=1).dropna()
#
#     all_skus['sku_id'] = sku_lbe.inverse_transform(all_skus['sku_id'])
#     print("sku nums: " + str(all_skus.count()))
#     sku_side_info = pd.merge(all_skus, product_data, on='sku_id', how='left').fillna(0)
#
#     # id2index
#     for feat in sku_side_info.columns:
#         if feat != 'sku_id':
#             lbe = LabelEncoder()
#             sku_side_info[feat] = lbe.fit_transform(sku_side_info[feat])
#         else:
#             sku_side_info[feat] = sku_lbe.transform(sku_side_info[feat])
#
#     sku_side_info = sku_side_info.sort_values(by=['sku_id'], ascending=True)
#     sku_side_info.to_csv('./data_cache/sku_side_info.csv', index=False, header=False, sep='\t')
#     print(sku_side_info.head(5))
#
#     # get pair
#     all_pairs = get_graph_context_all_pairs(session_reproduce, args.window_size)
#     np.savetxt('./data_cache/all_pairs', X=all_pairs, fmt="%d", delimiter=" ")