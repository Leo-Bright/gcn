import numpy as np
import pickle as pkl

from collections import defaultdict
from gcn.utils import parse_index_file

import random as rd
import scipy.sparse as sp


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

    if input.rsplit('.',1 )[-1] not in support_suffix:
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


def split_labeled_instance(all_labeled_pkl_path):

    with open(all_labeled_pkl_path, "rb") as f:
        all_labeled_samples = pkl.load(f)

    rd.shuffle(all_labeled_samples)
    rd.shuffle(all_labeled_samples)
    rd.shuffle(all_labeled_samples)

    x = all_labeled_samples[:-1500]

    test = all_labeled_samples[-1500:-500]

    valid = all_labeled_samples[-500:]

    with open("sanfrancisco/ind.sanfrancisco.x.index", "w+") as f:
        for item in x:
            f.write(str(item) + '\n')

    with open("sanfrancisco/ind.sanfrancisco.testx.index", "w+") as f:
        for item in test:
            f.write(str(item) + '\n')

    with open("sanfrancisco/ind.sanfrancisco.validx.index", "w+") as f:
        for item in valid:
            f.write(str(item) + '\n')


def get_x_y_file(input_idx_file, node2tag, node2emb, idx2node, node2idx, output=["x", "y"]):

    x = []

    y = []

    red = set()

    with open(input_idx_file) as f:
        for l in f:
            node_id = idx2node[int(l.strip())]
            if node_id not in node2emb:
                red.add(int(l.strip()))
                continue
            features = node2emb[node_id]
            label = node2tag[node_id]

            new_features = []
            for i in range(len(features)):
                new_features.append(float(features[i]))

            new_label = []
            for i in range(len(label)):
                new_label.append(int(label[i]))

            x.append(new_features)
            y.append(np.array(new_label))

    X = sp.csr_matrix(x)

    Y = np.array(y)

    with open("sanfrancisco/ind.sanfrancisco." + output[0], "wb") as f:
        pkl.dump(X, f)

    with open("sanfrancisco/ind.sanfrancisco." + output[1], "wb") as f:
        pkl.dump(Y, f)

    return red


def get_other_x_y_file(node2emb, node2idx, network):

    other_idx = []

    x = []

    y = []

    idx_paths = ["sanfrancisco/ind.sanfrancisco.x.index",
                 "sanfrancisco/ind.sanfrancisco.testx.index",
                 "sanfrancisco/ind.sanfrancisco.validx.index"]

    idx_had = set()
    for path in idx_paths:
        with open(path, "r") as f:
            for l in f:
                idx_had.add(int(l.strip()))

    for node_id in node2emb:
        idx = node2idx[node_id]
        if idx in idx_had:
            continue
        else:
            idx_had.add(idx)
            other_idx.append(idx)
            features = node2emb[node_id]
            label = np.zeros(9)

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

    X = sp.vstack((x, validx, otherx))

    Y = np.vstack((y, validy, othery))

    with open("sanfrancisco/ind.sanfrancisco.allx", "wb") as f:
        pkl.dump(X, f)

    with open("sanfrancisco/ind.sanfrancisco.ally", "wb") as f:
        pkl.dump(Y, f)


def generate_global_idx(node2emb, idx2node):

    idx_paths = ["sanfrancisco/ind.sanfrancisco.x.index",
                 "sanfrancisco/ind.sanfrancisco.validx.index",
                 "sanfrancisco/ind.sanfrancisco.otherx.index",
                 "sanfrancisco/ind.sanfrancisco.testx.index"]

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format("sanfrancisco"))
    test_idx_range = np.sort(test_idx_reorder)

    feature_idx = []
    for i in range(len(idx_paths)):
        with open(idx_paths[i]) as f:
            for l in f:
                f_idx = int(l.strip())
                if idx2node[f_idx] not in node2emb:
                    continue
                feature_idx.append(int(l.strip()))

    print(len(feature_idx))

    features = np.array(feature_idx)

    features[test_idx_reorder] = features[test_idx_range]

    with open("sanfrancisco/gcn_128dim_embedding.idx", "wb") as f:
        pkl.dump(features, f)


def remove_redundant_node(road_network, redundant_idx):

    for idx in redundant_idx:
        del road_network[idx]

    for k in road_network:
        nodes = road_network[k]
        road_network[k] = list(set(nodes) - redundant_idx)


if __name__ == '__main__':

    with open("sanfrancisco/sf_node_idx_dict.pkl", "rb") as f:
        node_idx_dict = pkl.load(f)

    with open("sanfrancisco/sf_idx_node_dict.pkl", "rb") as f:
        idx_node_dict = pkl.load(f)

    with open("sanfrancisco/sanfrancisco_nodes_with_all_tag.pkl", "rb") as f:
        node_tag_dict = pkl.load(f)

    with open("sanfrancisco/sf_shortest_distance_dim128_isrn2vec_node.pkl", "rb") as f:
        node_emb_dict = pkl.load(f)

    with open("sanfrancisco/ind.sanfrancisco.graph", "rb") as f:
        network = pkl.load(f)

    with open("sanfrancisco/ind.sanfrancisco.graph", "rb") as f:
        network = pkl.load(f)

    with open("sanfrancisco/ind.sanfrancisco.allx", "rb") as f:
        allx = pkl.load(f)

    with open("sanfrancisco/ind.sanfrancisco.tx", "rb") as f:
        tx = pkl.load(f)

    # red_idx = get_x_y_file("sanfrancisco/ind.sanfrancisco.valid.index", node_tag_dict,
    #                        node_emb_dict, idx_node_dict,node_idx_dict, output=["validx", "validy"])

    # red_idx = get_other_x_y_file(node_emb_dict, node_idx_dict, network)

    # get_all_x_y_file()

    # test_idxs = list(range(57257, 57257 + 997))
    #
    # rd.shuffle(test_idxs)

    # with open("sanfrancisco/ind.sanfrancisco.test.index", 'w') as f:
    #     for idx in test_idxs:
    #         f.write(str(idx) + '\n')

    # remove_redundant_node(network, red_idx)

    # with open("sanfrancisco/ind.sanfrancisco.graph", "wb") as f:
    #     pkl.dump(network, f)

    # print(red_idx)

    generate_global_idx(node_emb_dict, idx_node_dict)

    print("1")







