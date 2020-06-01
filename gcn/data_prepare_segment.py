import numpy as np
import pickle as pkl

from collections import defaultdict
from gcn.utils import parse_index_file

import random as rd
import scipy.sparse as sp
import json


def get_ndarray(items, dim=9):

    if len(items) == 1 and items != "-1":
        v = np.zeros(dim)
        v[items] = 1
        return v
    elif len(items) == 1 and items == "-1":
        return None
    else:
        map(float, items)
        v = np.array(items)
        return v


# supports emb file and tag file
def trans_input_file_to_ndarray(input):

    support_suffix = ["emb", "embedding", "embeddings", "tag"]

    if input.rsplit('.', 1)[-1] not in support_suffix:
        raise BaseException("Only support emb and tag file.")

    output_ndarray_dic = {}

    with open(input) as f:
        for line in f:
            node_others = line.strip().split(' ')
            node_id = node_others[0]
            others = node_others[1:]
            oth_array = get_ndarray(others, dim=9)
            if len(oth_array) > 0:
                output_ndarray_dic[node_id] = oth_array

    return output_ndarray_dic


def load_edgelist(file_, idx_dict, undirected=True):
    G = defaultdict(list)
    with open(file_) as f:
        for l in f:
            x, y = l.strip().split()[:2]
            x = idx_dict[x]
            y = idx_dict[y]
            G[x].append(y)
            if undirected:
                G[y].append(x)

    for k in G.keys():
        G[k] = list(sorted(set(G[k])))

    for x in G:
        if x in G[x]:
            G[x].remove(x)

    return G


def split_labeled_instance(all_labeled_samples_path, train_size, valid_size):

    with open(all_labeled_samples_path, "rb") as f:
        all_labeled_samples = pkl.load(f)

    rd.shuffle(all_labeled_samples)
    rd.shuffle(all_labeled_samples)
    rd.shuffle(all_labeled_samples)

    size = len(all_labeled_samples)

    x = all_labeled_samples[:train_size]

    valid = all_labeled_samples[train_size: train_size + valid_size]

    test = all_labeled_samples[train_size + valid_size:]

    print(len(test))

    with open("sanfrancisco/ind.sanfrancisco.x.index", "w+") as f:
        for item in x:
            f.write(str(item) + '\n')

    with open("sanfrancisco/ind.sanfrancisco.testx.index", "w+") as f:
        for item in test:
            f.write(str(item) + '\n')

    with open("sanfrancisco/ind.sanfrancisco.validx.index", "w+") as f:
        for item in valid:
            f.write(str(item) + '\n')

    return len(test)


def get_x_y_file(input_idx_file, seg2tag, seg2emb, idx2seg, network, output=["x", "y"]):

    x = []

    y = []

    red = set()

    with open(input_idx_file) as f:
        for l in f:
            seg_id = idx2seg[int(l.strip())]
            if int(seg_id) not in network:
                continue
            if seg_id not in seg2emb:
                red.add(int(l.strip()))
                continue
            features = seg2emb[seg_id]
            # label = node2tag[node_id]
            label = [1, 0] if seg_id in seg2tag else [0, 1]

            new_features = []
            for i in range(len(features)):
                new_features.append(float(features[i]))

            new_label = label

            x.append(new_features)
            y.append(np.array(new_label))

    X = sp.csr_matrix(x)

    Y = np.array(y)

    with open("sanfrancisco/ind.sanfrancisco." + output[0], "wb") as f:
        pkl.dump(X, f)

    with open("sanfrancisco/ind.sanfrancisco." + output[1], "wb") as f:
        pkl.dump(Y, f)

    return red


# the samples that have not label
def get_other_x_y_file(idx_paths, seg2emb, seg2idx, network):

    other_idx = []

    x = []

    y = []

    idx_had = set()
    for path in idx_paths:
        with open(path, "r") as f:
            for l in f:
                idx_had.add(int(l.strip()))

    for seg_id in seg2idx:
        if int(seg_id) not in network:
            continue
        idx = seg2idx[seg_id]
        if idx in idx_had:
            continue
        else:
            idx_had.add(idx)
            other_idx.append(idx)
            features = seg2emb[seg_id]
            label = np.zeros(2)

            new_features = []
            for i in range(len(features)):
                new_features.append(float(features[i]))

            new_label = label
            x.append(new_features)
            y.append(np.array(new_label))

    X = sp.csr_matrix(x)

    Y = np.array(y)

    with open("sanfrancisco/ind.sanfrancisco.otherx", "wb") as f:
        pkl.dump(X, f)

    with open("sanfrancisco/ind.sanfrancisco.othery", "wb") as f:
        pkl.dump(Y, f)

    with open("sanfrancisco/ind.sanfrancisco.otherx.index", "w") as f:
        for i in other_idx:
            f.write(str(i) + '\n')

    network_idxs = set()

    for idx in network:
        network_idxs.add(idx)

    red = network_idxs - idx_had

    return red


def get_all_x_y_file():

    names = ['x', 'y', 'validx', 'validy', 'otherx', 'othery']
    objects = []
    for i in range(len(names)):
        with open("sanfrancisco/ind.sanfrancisco.{}".format(names[i]), 'rb') as f:
            objects.append(pkl.load(f))

    x, y, validx, validy, otherx, othery = tuple(objects)

    X = sp.vstack((x, validx, otherx, otherx[-1:]))

    Y = np.vstack((y, validy, othery, othery[-1:]))

    with open("sanfrancisco/ind.sanfrancisco.allx", "wb") as f:
        pkl.dump(X, f)

    with open("sanfrancisco/ind.sanfrancisco.ally", "wb") as f:
        pkl.dump(Y, f)


def generate_global_idx(seg2emb, idx2seg, output_path, network):

    idx_paths = ["sanfrancisco/ind.sanfrancisco.x.index",
                 "sanfrancisco/ind.sanfrancisco.validx.index",
                 "sanfrancisco/ind.sanfrancisco.otherx.index",
                 "sanfrancisco/ind.sanfrancisco.testx.index"]

    test_idx_reorder = parse_index_file("sanfrancisco/ind.{}.test.index".format("sanfrancisco"))
    test_idx_range = np.sort(test_idx_reorder)

    feature_idx = []
    for i in range(len(idx_paths)):
        with open(idx_paths[i]) as f:
            for l in f:
                f_idx = int(l.strip())
                if idx2seg[f_idx] not in seg2emb:
                    continue
                feature_idx.append(int(l.strip()))

    feature_idx.append(0)
    print(len(feature_idx))

    features = np.array(feature_idx)

    features[test_idx_reorder] = features[test_idx_range]

    with open(output_path, "wb") as f:
        pkl.dump(features, f)


def remove_redundant_node(road_network, redundant_idx, graph_file_path):

    for idx in redundant_idx:
        del road_network[idx]

    for k in road_network:
        nodes = road_network[k]
        road_network[k] = list(set(nodes) - redundant_idx)

    with open(graph_file_path, "wb") as graph_file:
        pkl.dump(road_network, graph_file)


def generate_network_graph(network_file_path, graph_file_path, node_idx_dict):

    graph = {}

    with open(network_file_path) as f:
        for line in f:
            ids = line.strip().split(' ')
            start = node_idx_dict[ids[0]]
            end = node_idx_dict[ids[1]]

            if start not in graph:
                graph[start] = []
            graph[start].append(end)

            if end not in graph:
                graph[end] = []
            graph[end].append(start)

    with open(graph_file_path + '.json', 'w+') as f:
        f.write(json.dumps(graph))

    with open(graph_file_path, 'wb') as f:
        pkl.dump(graph, f)

    seg_count = 0
    for key in graph:
        seg_count += 1
    print(seg_count)


def gen_all_labeled_pkl(seg2tag, seg2idx, all_labeled_samples_path, network):

    all_labeled_samples_idx = []

    count_labeled = 0
    count_unlabeled = 0
    for key in network:
        key = str(key)
        if key in seg2tag and seg2tag[key] == 'Ave':
            all_labeled_samples_idx.append(seg2idx[key])
            count_labeled += 1
        elif count_unlabeled < count_labeled:
            all_labeled_samples_idx.append(seg2idx[key])
            count_unlabeled += 1

    print(len(all_labeled_samples_idx))

    with open(all_labeled_samples_path, 'wb') as f:
        pkl.dump(all_labeled_samples_idx, f)


def gen_seg_idx_pkl_path(seg2emb, seg2idx_pkl_path, idx2seg_pkl_path):

    seg2idx = {}
    idx2seg = {}

    count = 0
    for key in seg2emb:
        seg2idx[key] = count
        idx2seg[count] = key
        count += 1

    with open(seg2idx_pkl_path, 'wb') as f:
        pkl.dump(seg2idx, f)

    with open(idx2seg_pkl_path, 'wb') as f:
        pkl.dump(idx2seg, f)


def gen_test_index_file(samples_num, output_path):

    test_idxs = list(range(75517 - samples_num, 75517))
    rd.shuffle(test_idxs)

    with open(output_path, 'w+') as f:
        for idx in test_idxs:
            f.write(str(idx) + '\n')


if __name__ == '__main__':

    with open("sanfrancisco/sf_seg_idx_dict.pkl", "rb") as f:
        seg_idx_dict = pkl.load(f)

    with open("sanfrancisco/sf_idx_seg_dict.pkl", "rb") as f:
        idx_seg_dict = pkl.load(f)

    with open("sanfrancisco/osm_data/sf_segments_tiger_nametype.json") as f:
        seg_tag_dict = json.loads(f.readline())

    seg_emb_dict = trans_input_file_to_ndarray('sanfrancisco/embeddings/sanfrancisco_raw_feature_segment.embeddings')

    graph_file_path = "sanfrancisco/ind.sanfrancisco.graph"
    with open(graph_file_path, "rb") as f:
        network = pkl.load(f)

    x_index = ['sanfrancisco/ind.sanfrancisco.x.index', 'x', 'y']
    test_x_index = ['sanfrancisco/ind.sanfrancisco.testx.index', 'tx', 'ty']
    valid_x_index = ['sanfrancisco/ind.sanfrancisco.validx.index', 'validx', 'validy']
    indexes_to_gen = [x_index, test_x_index, valid_x_index]

    # step1: generate x,testx,validx,y,testy,validy file
    # for index in indexes_to_gen:
    #     red_idx = get_x_y_file(index[0], seg_tag_dict, seg_emb_dict, idx_seg_dict, network, output=index[1:])
    #     if len(red_idx) > 0:
    #         print('red_idx:', len(red_idx))
    #         # remove_redundant_node(network, red_idx, graph_file_path)

    # step2: generate otherx, othery file
    # gcn_emb_idx_path = 'sanfrancisco/embeddings/sf_gcn_raw_feature_none_16d_traffic.embedding.idx.pkl'
    # idx_paths = [x_index[0], test_x_index[0], valid_x_index[0]]
    # red_idx = get_other_x_y_file(idx_paths, seg_emb_dict, seg_idx_dict, network)
    # if len(red_idx) > 0:
    #     print('red_idx:', len(red_idx))
    #     # remove_redundant_node(network, red_idx, graph_file_path)
    # generate_global_idx(seg_emb_dict, idx_seg_dict, gcn_emb_idx_path, network)

    # step3: generate allx, ally files that use in train
    get_all_x_y_file()

    ########################################
    # under is not main process
    ########################################

    # step0-0: generate the init indexes file to x, testx, validx
    # seg_idx_pkl_path = 'sanfrancisco/sf_seg_idx_dict.pkl'
    # idx_seg_pkl_path = 'sanfrancisco/sf_idx_seg_dict.pkl'
    # gen_seg_idx_pkl_path(seg_emb_dict, seg_idx_pkl_path, idx_seg_pkl_path)

    # step0-1: generate the road network graph
    # generate_network_graph('sanfrancisco/osm_data/sanfrancisco_segment.network',
    #                        'sanfrancisco/ind.sanfrancisco.graph',
    #                        seg_idx_dict)

    # step0-2: generate idx pkl file of the all labeled samples that use to train/test/valid
    # all_labeled_pkl_path = 'sanfrancisco/ind.sanfrancisco.all.labeled.pkl'
    # gen_all_labeled_pkl(seg_tag_dict, seg_idx_dict, all_labeled_pkl_path, network)

    # step0-3: generate idx file of the all labeled samples that use to test
    # all_labeled_pkl_path = 'sanfrancisco/ind.sanfrancisco.all.labeled.pkl'
    # samples_size = split_labeled_instance(all_labeled_pkl_path, 18000, 2000)
    # test_index_file_path = 'sanfrancisco/ind.sanfrancisco.test.index'
    # gen_test_index_file(samples_size, test_index_file_path)

    print("1")







