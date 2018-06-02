from torch.utils.data import Dataset
import numpy as np
import torch
import math
import os
import glob
from sklearn.decomposition import FastICA, PCA


class MyDataset(Dataset):
    def __init__(self, dir_path='../MADSYN/data/eeg', num_files=860, seq_len=256, stride=0.5,
                 max_freq=80, num_channels=1, per_user=True, mode=None,  # it could be None, pca or ica
                 use_abs=False, model_dataset_depth_offset=2):  # we start with 4x4 resolution instead of 1x1
        self.model_depth = 0
        self.alpha = 1.0
        self.model_dataset_depth_offset = model_dataset_depth_offset
        self.dir_path = dir_path
        self.all_files = glob.glob(os.path.join(dir_path, '*_1.txt'))[:num_files]
        self.seq_len = seq_len
        self.stride = int(seq_len * stride)
        self.num_channels = num_channels
        self.use_abs = use_abs
        sizes = []
        for i in range(num_files):
            with open(self.all_files[i]) as f:
                all_data_len = len(list(map(float, f.read().split()))) // (80 // max_freq)
                sizes.append(max(int(np.ceil((all_data_len - self.seq_len + 1) / self.stride)), 0))
        self.sizes = sizes
        self.data_pointers = [(i, j) for i in range(num_files) for j in range(self.sizes[i])]
        num_points = [(self.sizes[i] - 1) * self.stride + self.seq_len for i in range(num_files)]
        self.datas = [np.zeros((num_channels, num_points[i]), dtype=np.float32) for i in range(num_files)]
        for i in range(num_files):
            for j in range(num_channels):
                with open('{}_{}.txt'.format(self.all_files[i][:-6], j + 1)) as f:
                    tmp = np.array(list(map(float, f.read().split())), dtype=np.float32)[::(80 // max_freq)][
                          :num_points[i]]
                    self.datas[i][j, :] = tmp
            if per_user:
                self.datas[i] = self.normalize(self.datas[i])
        if not per_user:
            self.normalize_all(num_files)
        if mode:
            self.transformer = PCA(n_components=num_channels) if mode == 'pca' else FastICA(n_components=num_channels)
            self.transformer.fit(np.concatenate([d.T for d in self.datas], axis=0))
            for i in range(num_files):
                self.datas[i] = self.transformer.transform(self.datas[i].T).T
            self.all_max, self.all_min = self.normalize_all(num_files)
        else:
            self.transformer = None
        self.max_dataset_depth = self.infer_max_dataset_depth(self.load_file(0))
        self.min_dataset_depth = self.model_dataset_depth_offset
        self.description = {
            'len': len(self),
            'shape': self.shape,
            'depth_range': (self.min_dataset_depth, self.max_dataset_depth)
        }

    def inverse_transform(self, x):
        if self.transformer is None:
            return x
        y = self.denormalize(x)
        return np.stack([self.transformer.inverse_transform(y[i].T).T for i in range(len(y))])

    def denormalize(self, x):
        if self.use_abs:
            return x * max(abs(self.all_max), abs(self.all_min))
        return (x + 1.0) / 2.0 * (self.all_max - self.all_min) + self.all_min

    def normalize_all(self, num_files):
        all_max = max([self.datas[i].max() for i in range(num_files)])
        all_min = min([self.datas[i].min() for i in range(num_files)])
        for i in range(num_files):
            self.datas[i] = self.normalize(self.datas[i], all_max, all_min)
        return all_max, all_min

    def normalize(self, arr, arr_max=None, arr_min=None):
        if arr_max is None:
            arr_max = arr.max()
        if arr_min is None:
            arr_min = arr.min()
        if self.use_abs:
            arr_max = max(abs(arr_max), abs(arr_min))
            return arr / arr_max
        return ((arr - arr_min) / (arr_max - arr_min)) * 2.0 - 1.0

    @property
    def data(self):
        return self.datas

    @property
    def shape(self):
        return (len(self),) + self.load_file(0).shape

    def __len__(self):
        return len(self.data_pointers)

    def get_datapoint_version(self, datapoint, datapoint_depth, target_depth):
        if datapoint_depth == target_depth:
            return datapoint
        return self.create_datapoint_from_depth(datapoint, datapoint_depth, target_depth)

    def create_datapoint_from_depth(self, datapoint, datapoint_depth, target_depth):
        datapoint = datapoint.astype(np.float32)
        depthdiff = (datapoint_depth - target_depth)
        return datapoint[:, ::(2 ** depthdiff)]

    def load_file(self, item):
        i, k = self.data_pointers[item]
        res = self.datas[i][:, k * self.stride:k * self.stride + self.seq_len]
        return res

    @staticmethod
    def infer_max_dataset_depth(datapoint):
        return int(math.log(datapoint.shape[-1], 2))

    def __getitem__(self, item):
        datapoint = self.load_file(item)
        datapoint = self.get_datapoint_version(datapoint, self.max_dataset_depth,
                                               self.model_depth + self.model_dataset_depth_offset)
        datapoint = self.alpha_fade(datapoint)
        return torch.from_numpy(datapoint.astype('float32'))

    def alpha_fade(self, datapoint):
        if self.alpha == 1:
            return datapoint
        c, t = datapoint.shape
        t = datapoint.reshape(c, t // 2, 2).mean(axis=2).repeat(2, 1)
        return datapoint + (t - datapoint) * (1 - self.alpha)
