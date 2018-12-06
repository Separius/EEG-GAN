import os
import math
import glob
import torch
import numpy as np
from tqdm import trange
from utils import load_pkl, save_pkl
from torch.utils.data import Dataset

DATASET_VERSION = 1


class EEGDataset(Dataset):
    # TODO conditional DB
    def __init__(self, given_data, dir_path: str = './data/tuh1', seq_len: int = 512, stride: float = 0.25,
                 num_channels: int = 5, per_user_normalization: bool = True, dataset_freq: int = 80,
                 progression_scale: int = 2, num_files: int = 12518, per_channel_normalization: bool = False,
                 model_dataset_depth_offset: int = 2):  # start from progression_scale^2 instead of progression_scale^0
        super().__init__()
        self.model_depth = 0
        self.alpha = 1.0
        self.model_dataset_depth_offset = model_dataset_depth_offset
        self.dir_path = dir_path
        self.seq_len = seq_len
        self.progression_scale = progression_scale
        self.stride = int(seq_len * stride)
        self.num_channels = num_channels
        self.max_freq = dataset_freq
        self.per_user_normalization = per_user_normalization
        self.per_channel_normalization = per_channel_normalization
        self.dataset_freq = dataset_freq
        self.max_dataset_depth = int(math.log(self.seq_len, self.progression_scale))
        self.min_dataset_depth = self.model_dataset_depth_offset
        self.all_files = glob.glob(os.path.join(dir_path, '*_1.txt'))
        num_files = min(len(self.all_files), num_files)
        if given_data is not None:
            self.sizes = given_data[0]
            self.data_pointers = given_data[1]
            self.datas = [given_data[2]['arr_{}'.format(i)] for i in trange(len(given_data[2]))]
            return
        sizes = []
        num_points = []
        self.datas = []
        for i in trange(num_files):
            is_ok = True
            for j in range(num_channels):
                with open('{}_{}.txt'.format(self.all_files[i][:-6], j + 1)) as f:
                    tmp = list(map(float, f.read().split()))
                    if j == 0:
                        size = int(np.ceil((len(tmp) - seq_len + 1) / self.stride))
                        if size <= 0:
                            is_ok = False
                            break
                        sizes.append(size)
                        num_points.append((sizes[-1] - 1) * self.stride + seq_len)
                        self.datas.append(np.zeros((num_channels, num_points[-1]), dtype=np.float32))
                    tmp = np.array(tmp, dtype=np.float32)[:num_points[-1]]
                    self.datas[-1][j, :] = tmp
            if is_ok and per_user_normalization:
                self.datas[-1], is_ok = self.normalize(self.datas[-1], self.per_channel_normalization)
                if not is_ok:
                    del sizes[-1]
                    del num_points[-1]
                    del self.datas[-1]
        self.sizes = sizes
        self.data_pointers = [(i, j) for i, s in enumerate(self.sizes) for j in range(s)]
        if not per_user_normalization:
            self.normalize_all()

    @classmethod
    def from_config(cls, dir_path: str, seq_len: int, stride: float, num_channels: int, per_user_normalization: bool,
                    dataset_freq: int, progression_scale: int, model_dataset_depth_offset: int, num_files: int,
                    per_channel_normalization: bool):
        target_location = os.path.join(dir_path,
                                       '{}l_{}c_{}p_{}o_{}m_{}v.npz'.format(seq_len, num_channels, progression_scale,
                                                                            model_dataset_depth_offset,
                                                                            per_user_normalization * 2 + per_channel_normalization * 1,
                                                                            DATASET_VERSION))
        if os.path.exists(target_location):
            print('loading dataset from file')
            given_data = np.load(target_location)
            given_data = [load_pkl(target_location + '_sizes.pkl'), load_pkl(target_location + '_pointers.pkl'),
                          given_data]
        else:
            print('creating dataset from scratch')
            given_data = None
        dataset = cls(given_data, dir_path, seq_len, stride, num_channels, per_user_normalization,
                      dataset_freq, progression_scale, num_files, per_channel_normalization, model_dataset_depth_offset)
        if given_data is None:
            np.savez_compressed(target_location, *dataset.datas)
            save_pkl(target_location + '_sizes.pkl', dataset.sizes)
            save_pkl(target_location + '_pointers.pkl', dataset.data_pointers)
        return dataset

    def normalize_all(self):
        num_files = len(self.datas)
        all_max = np.max(
            np.array([self.datas[i].max(axis=1 if self.per_channel_normalization else None) for i in range(num_files)]),
            axis=0)
        all_min = np.min(
            np.array([self.datas[i].min(axis=1 if self.per_channel_normalization else None) for i in range(num_files)]),
            axis=0)
        is_ok = True
        for i in range(num_files):
            self.datas[i], is_ok = self.normalize(self.datas[i], self.per_channel_normalization, all_max, all_min)
        if not is_ok:
            raise ValueError('data is constant!')

    @staticmethod
    def normalize(arr, per_channel, arr_max=None, arr_min=None):
        if arr_max is None:
            arr_max = arr.max(axis=1 if per_channel else None)
        if arr_min is None:
            arr_min = arr.min(axis=1 if per_channel else None)
        is_ok = arr_max != arr_min
        if per_channel:
            is_ok = is_ok.all()
        return ((arr - arr_min) / ((arr_max - arr_min) if is_ok else 1.0)) * 2.0 - 1.0, is_ok

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
        depth_diff = (datapoint_depth - target_depth)
        return datapoint[:, ::(self.progression_scale ** depth_diff)]

    def load_file(self, item):
        i, k = self.data_pointers[item]
        res = self.datas[i][:, k * self.stride:k * self.stride + self.seq_len]
        return res

    def __getitem__(self, item):
        datapoint = self.load_file(item)
        datapoint = self.get_datapoint_version(datapoint, self.max_dataset_depth,
                                               self.model_depth + self.model_dataset_depth_offset)
        datapoint = self.alpha_fade(datapoint)
        return torch.from_numpy(datapoint.astype(np.float32))

    def alpha_fade(self, datapoint):
        if self.alpha == 1:
            return datapoint
        c, t = datapoint.shape
        t = datapoint.reshape(c, t // self.progression_scale, self.progression_scale).mean(axis=2).repeat(
            self.progression_scale, axis=1)
        return datapoint + (t - datapoint) * (1 - self.alpha)
