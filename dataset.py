import os
import math
import glob
import torch
import numpy as np
from random import shuffle
from tqdm import tqdm, trange
from torch.utils.data import Dataset
from utils import load_pkl, save_pkl, EPSILON, random_onehot

DATASET_VERSION = 2


class EEGDataset(Dataset):
    def __init__(self, train_files, norms, given_data, validation_ratio: float = 0.1, dir_path: str = './data/tuh1',
                 seq_len: int = 512, stride: float = 0.25, num_channels: int = 5, per_user_normalization: bool = True,
                 progression_scale: int = 2, per_channel_normalization: bool = False,
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
        self.per_user_normalization = per_user_normalization
        self.per_channel_normalization = per_channel_normalization
        self.max_dataset_depth = int(math.log(self.seq_len, self.progression_scale))
        self.min_dataset_depth = self.model_dataset_depth_offset
        self.y, self.class_options = self.read_meta_info(os.path.join(dir_path, 'meta.info'))
        self.norms = norms
        if given_data is not None:
            self.sizes = given_data[0]['sizes']
            self.files = given_data[0]['files']
            self.norms = given_data[0]['norms']
            self.data_pointers = given_data[0]['pointers']
            self.datas = [given_data[1]['arr_{}'.format(i)] for i in trange(len(given_data[1].keys()))]
            if self.y is not None:
                self.y = [self.y[i] for i in self.files]
            return
        all_files = glob.glob(os.path.join(dir_path, '*_1.txt'))
        files = len(all_files)
        files = [i for i in range(files)]
        if train_files is None:
            shuffle(files)
            files = files[:int(len(all_files) * (1.0 - validation_ratio))]
        else:
            files = list(set(files) - set(train_files))
        if self.y is not None:
            self.y = [self.y[i] for i in files]
        self.files = files
        sizes = []
        num_points = []
        self.datas = []
        for i in tqdm(files):
            is_ok = True
            for j in range(num_channels):
                with open('{}_{}.txt'.format(all_files[i][:-6], j + 1)) as f:
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

    def generate_class_condition(self, batch_size, one_hot_probability=0.8):
        if self.y is None:
            return None
        res = np.zeros((batch_size, len(self.y[0])), dtype=np.float32)
        start_index = 0
        for num_options in self.class_options:
            normalized = np.random.uniform(EPSILON, 1, (batch_size, num_options))
            normalized /= normalized.sum(axis=1, keepdims=True)
            res[:, start_index:start_index + num_options] = np.where(
                np.random.uniform(0, 1, (batch_size, num_options)) > one_hot_probability, normalized,
                random_onehot(num_options, batch_size))
            start_index += num_options
        return torch.from_numpy(res)

    @staticmethod
    def read_meta_info(file_name):
        if not os.path.exists(file_name):
            return None, None
        # example:
        # 3 2 2
        # 0.33 0.33 0.34 1 0 0 1
        attributes = []
        num_values = None
        first_line = True
        with open(file_name) as f:
            for line in f:
                if first_line:
                    num_values = [int(p) for p in line.split()]
                    first_line = False
                    continue
                attributes.append(np.array([float(p) for p in line.split()], dtype=np.float32))
        return np.stack(attributes, axis=0), num_values

    @classmethod
    def from_config(cls, validation_ratio: float, dir_path: str, seq_len: int, stride: float, num_channels: int,
                    per_user_normalization: bool, progression_scale: int, model_dataset_depth_offset: int,
                    per_channel_normalization: bool):
        mode = per_user_normalization * 2 + per_channel_normalization * 1
        train_files = None
        train_norms = None
        datasets = [None, None]
        for index, split in enumerate(('train', 'val')):
            target_location = os.path.join(dir_path,
                                           '{}%_{}l_{}c_{}p_{}o_{}m_{}v_{}.npz'.format(validation_ratio, seq_len,
                                                                                       num_channels, progression_scale,
                                                                                       model_dataset_depth_offset, mode,
                                                                                       DATASET_VERSION, split))
            if os.path.exists(target_location):
                print('loading {} dataset from file'.format(split))
                given_data = (load_pkl(target_location + '.pkl'), np.load(target_location))
            else:
                print('creating {} dataset from scratch'.format(split))
                given_data = None
            dataset = cls(train_files, train_norms, given_data, validation_ratio, dir_path, seq_len, stride,
                          num_channels, per_user_normalization, progression_scale, per_channel_normalization,
                          model_dataset_depth_offset)
            if train_files is None:
                train_files = dataset.files
                train_norms = dataset.norms
            if given_data is None:
                np.savez_compressed(target_location, *dataset.datas)
                save_pkl(target_location + '.pkl', {'sizes': dataset.sizes, 'pointers': dataset.data_pointers,
                                                    'norms': dataset.norms, 'files': dataset.files})
            datasets[index] = dataset
        return datasets[0], datasets[1]

    def normalize_all(self):
        num_files = len(self.datas)
        if self.norms is None:
            all_max = np.max(
                np.array([data.max(axis=1 if self.per_channel_normalization else None) for data in self.datas]), axis=0)
            all_min = np.min(
                np.array([data.min(axis=1 if self.per_channel_normalization else None) for data in self.datas]), axis=0)
            self.norms = (all_max, all_min)
        else:
            all_max, all_min = self.norms
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
        x = torch.from_numpy(datapoint.astype(np.float32))
        if self.y is None:
            return {'x': x}
        return {'x': x, 'y': torch.from_numpy(self.y[self.data_pointers[item][0]])}

    def alpha_fade(self, datapoint):
        if self.alpha == 1:
            return datapoint
        c, t = datapoint.shape
        t = datapoint.reshape(c, t // self.progression_scale, self.progression_scale).mean(axis=2).repeat(
            self.progression_scale, axis=1)
        return datapoint + (t - datapoint) * (1 - self.alpha)
