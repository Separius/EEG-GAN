import os
import math
import glob
import torch
import numpy as np
from tqdm import trange
from scipy.io import loadmat
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, dir_path='./data/tuh2', seq_len=512, stride=0.25, num_channels=5,
                 per_user=True, dataset_freq=80,
                 model_dataset_depth_offset=2):  # start from progression_scale^2 instead of progression_scale^0
        self.model_depth = 0
        self.alpha = 1.0
        self.model_dataset_depth_offset = model_dataset_depth_offset
        self.dir_path = dir_path
        self.all_files = glob.glob(os.path.join(dir_path, '*_1.txt'))
        num_files = len(self.all_files)
        self.seq_len = seq_len
        self.progression_scale = 2
        self.stride = int(seq_len * stride)
        self.num_channels = num_channels
        self.max_freq = dataset_freq
        self.per_user = per_user
        self.dataset_freq = dataset_freq
        self.max_dataset_depth = int(math.log(self.seq_len, self.progression_scale))
        self.min_dataset_depth = self.model_dataset_depth_offset
        sizes = []
        for i in range(num_files):
            with open(self.all_files[i]) as f:
                all_data_len = len(list(map(float, f.read().split())))
                sizes.append(max(int(np.ceil((all_data_len - seq_len + 1) / self.stride)), 0))
        self.sizes = sizes
        self.data_pointers = [(i, j) for i in range(num_files) for j in range(self.sizes[i])]
        num_points = [((self.sizes[i] - 1) * self.stride + seq_len) if self.sizes[i] > 0 else 1 for i in
                      range(num_files)]
        self.datas = [np.zeros((num_channels, num_points[i]), dtype=np.float32) for i in range(num_files)]
        for i in trange(num_files):
            for j in range(num_channels):
                with open('{}_{}.txt'.format(self.all_files[i][:-6], j + 1)) as f:
                    tmp = np.array(list(map(float, f.read().split())), dtype=np.float32)[:num_points[i]]
                    self.datas[i][j, :] = tmp
            if per_user and self.sizes[i] > 0:
                self.datas[i] = self.normalize(self.datas[i])
        if not per_user:
            self.normalize_all(num_files)
        self.description = {
            'len': len(self),
            'shape': self.shape,
            'depth_range': (self.min_dataset_depth, self.max_dataset_depth)
        }

    def normalize_all(self, num_files):
        all_max = max([self.datas[i].max() for i in range(num_files)])
        all_min = min([self.datas[i].min() for i in range(num_files)])
        for i in range(num_files):
            self.datas[i] = self.normalize(self.datas[i], all_max, all_min)

    def normalize(self, arr, arr_max=None, arr_min=None):
        if arr_max is None:
            arr_max = arr.max()
        if arr_min is None:
            arr_min = arr.min()
        return ((arr - arr_min) / (arr_max - arr_min)) * 2.0 - 1.0

    @property
    def data(self):
        return self.datas

    @property
    def shape(self):
        return len(self), self.num_channels, self.seq_len

    def __len__(self):
        return len(self.data_pointers)

    def get_datapoint_version(self, datapoint, datapoint_depth, target_depth):
        if datapoint_depth == target_depth:
            return datapoint
        return self.create_datapoint_from_depth(datapoint, datapoint_depth, target_depth)

    def create_datapoint_from_depth(self, datapoint, datapoint_depth, target_depth):
        datapoint = datapoint.astype(np.float32)
        depthdiff = (datapoint_depth - target_depth)
        return datapoint[:, ::(self.progression_scale ** depthdiff)]

    def load_file(self, item):
        i, k = self.data_pointers[item]
        res = self.datas[i][:, k * self.stride:k * self.stride + self.seq_len]
        return res

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
        t = datapoint.reshape(c, t // self.progression_scale, self.progression_scale).mean(axis=2).repeat(
            self.progression_scale, axis=1)
        return datapoint + (t - datapoint) * (1 - self.alpha)
