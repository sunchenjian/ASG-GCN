from scipy.spatial import distance
import numpy as np
from sklearn.model_selection import StratifiedKFold
import sys

def load_data(label_dir, feature_dir):

    labels = np.load(label_dir, allow_pickle=True)

    y = labels
    y = y.astype(np.long)
    raw_features = np.load(feature_dir, allow_pickle=True)
    raw_features = raw_features.astype(float)

    return raw_features, y

def data_split(raw_features, y, n_folds):
    skf1 = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=111)
    skf2 = StratifiedKFold(n_splits=n_folds, shuffle=True)
    train_index, val_index, test_index = [], [], []
    for tr_ind, te_ind in skf1.split(raw_features, y):
        test_index.append(te_ind)
        for tr_tmp, val_tmp in skf2.split(raw_features[tr_ind], y[tr_ind]):
            train_index.append(tr_ind[tr_tmp])
            val_index.append(tr_ind[val_tmp])
            break
    return train_index, val_index, test_index

def get_static_affinity_adj(features):
    distv = distance.pdist(features, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))
    adj = feature_sim

    return adj

def get_adj(features, mood_att):
    n = features.shape[0]
    num_edge = n*(1+n)//2 - n
    ftr_dim = features.shape[1]
    edgenet_input = np.zeros([num_edge, 2 * ftr_dim], dtype=np.float32)
    edge_index = np.zeros([2, num_edge], dtype=np.int64)
    aff_score = np.zeros(num_edge, dtype=np.float32)
    mood_score = np.zeros(num_edge, dtype=np.float32)
    aff_adj = get_static_affinity_adj(features)
    flatten_ind = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[:, flatten_ind] = [i,j]
            edgenet_input[flatten_ind] = np.concatenate((features[i], features[j]))
            aff_score[flatten_ind] = aff_adj[i, j]
            mood_score[flatten_ind] = mood_att[i, j]
            flatten_ind += 1

    assert flatten_ind == num_edge

    keep_ind = np.where(aff_score > 0.95)[0]
    edge_index = edge_index[:, keep_ind]
    edgenet_input = edgenet_input[keep_ind]
    mood_score = mood_score[keep_ind]

    return edge_index, edgenet_input, mood_score

def get_A(n, p , s):
    edge_index = np.zeros([2, (n * (p + s) - p * (p + 1) // 2 - s * (s + 1) // 2)], dtype=np.int64)
    fla_ind = 0
    for i in range(n):
        for j in range(max(0, i - p), min(i + s + 1, n)):
            if i == j:
                continue
            edge_index[:, fla_ind] = [i, j]
            fla_ind += 1
    return edge_index


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass