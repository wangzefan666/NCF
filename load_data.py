import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data
import config


def load_all():
    """ We load all the two file here to save time in each epoch. """
    train_data = pd.read_csv(config.train_rating, sep='\t', header=None, names=['user', 'item'],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1
    train_data = train_data.values.tolist()  # [[user, item] * n_samples]

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)  # initialize with shape
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []  # [[user, item] * (1 + n_ng_samples) * n_users]
    with open(config.test_negative, 'r') as f:
        for l in f.readlines():
            arr = l.strip().split('\t')
            if len(arr) > 1:
                uid = eval(arr[0])[0]
                test_data.append([uid, eval(arr[0])[1]])  # (uid, pos_item)
                for i in arr[1:]:
                    test_data.append([uid, int(i)])
    return train_data, test_data, user_num, item_num, train_mat


class NCFData(data.Dataset):
    """
    Note that the labels are only useful when training, we thus
    add them in the ng_sample() function.
    """
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
        super(NCFData, self).__init__()
        self.num_item = num_item
        self.train_num_ng = num_ng
        self.num_features = len(features)
        self.is_training = is_training
        self.train_mat = train_mat
        self.features = features
        self.labels = [0 for _ in range(self.num_features)]
        self.train_features, self.train_labels = None, None

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        features_ng = []
        for x in self.features:
            u = x[0]
            for t in range(self.train_num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:  # train_mat: sp.dok_matrix
                    j = np.random.randint(self.num_item)
                features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features))]
        labels_ng = [0 for _ in range(len(features_ng))]

        # concat train data and labels, and they will be shuffled at the beginning of each epoch
        self.train_features = self.features + features_ng
        self.train_labels = labels_ps + labels_ng

    def __len__(self):
        return (self.train_num_ng + 1) * self.num_features

    def __getitem__(self, idx):
        features = self.train_features if self.is_training else self.features
        labels = self.train_labels if self.is_training else self.labels
        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label
