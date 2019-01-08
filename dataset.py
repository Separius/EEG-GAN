import os
import math
import glob
import torch
import numpy as np
from random import shuffle
from scipy.io import loadmat
from tqdm import tqdm, trange
from torch.utils.data import Dataset
from utils import load_pkl, save_pkl

DATASET_VERSION = 4


# 5(1/6),15(0.5),30(1),60(2),120(4), 240(8),480(16),720(24),1800(60),3000(100),6000(200) # 30 seconds
class EEGDataset(Dataset):
    def __init__(self, train_files, norms, given_data, validation_ratio: float = 0.1, dir_path: str = './data/tuh1',
                 seq_len: int = 1024, stride: float = 0.25, num_channels: int = 5, per_user_normalization: bool = True,
                 progression_scale: int = 2, per_channel_normalization: bool = False, no_condition: bool = True,
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
        self.y, file_ids = self.read_meta_info(os.path.join(dir_path, 'meta.npy'))
        self.norms = norms
        self.no_condition = no_condition
        if given_data is not None:
            self.sizes = given_data[0]['sizes']
            self.files = given_data[0]['files']
            self.norms = given_data[0]['norms']
            self.data_pointers = given_data[0]['pointers']
            self.datas = [given_data[1]['arr_{}'.format(i)] for i in trange(len(given_data[1].keys()))]
            if self.y is not None:
                self.y = self.y[self.files]
            return
        if file_ids is None:
            all_files = glob.glob(os.path.join(dir_path, '*_1.txt'))
            is_matlab = len(all_files) == 0
            if is_matlab:
                all_files = glob.glob(os.path.join(dir_path, '*.mat'))
                num_channels = 17
        else:
            all_files = [os.path.join(dir_path, '{}_1.txt'.format(f)) for f in file_ids]
            is_matlab = len(all_files) > 0 and not os.path.exists(all_files[0])
            if is_matlab:
                all_files = [os.path.join(dir_path, '{}.mat'.format(f)) for f in file_ids]
                num_channels = 17
        files = len(all_files)
        files = [i for i in range(files)]
        if train_files is None:
            shuffle(files)
            files = files[:int(len(all_files) * (1.0 - validation_ratio))]
        else:
            files = list(set(files) - set(train_files))
        if self.y is not None:
            self.y = self.y[files]
        self.files = files
        sizes = []
        num_points = []
        self.datas = []
        for i in tqdm(files):
            is_ok = True
            if is_matlab:
                tmp = loadmat(all_files[i])
                size = int(np.ceil((tmp.shape[1] - seq_len + 1) / self.stride))
                if size <= 0:
                    is_ok = False
                else:
                    sizes.append(size)
                    num_points.append((sizes[-1] - 1) * self.stride + seq_len)
                    self.datas.append(tmp[:num_channels, :num_points[-1]])
            else:
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

    @staticmethod
    def read_meta_info(file_name):
        if not os.path.exists(file_name):
            return None, None
        y = np.load(file_name)
        file_ids = y[:, 0]
        y = y[:, 1:].astype(np.float32)
        y[:, 0] = (y[:, 0] - 10.0) / 90.0
        return y, file_ids

    def generate_class_condition(self, batch_size):
        if self.y is None or self.no_condition:
            return None
        res = np.zeros((batch_size, self.y.shape[1]), dtype=np.float32)
        res[:, 0] = np.random.rand(batch_size)
        res[:, 1:] = np.random.rand(batch_size, self.y.shape[1] - 1) > 0.5  # TODO
        return torch.from_numpy(res)

    @classmethod
    def from_config(cls, validation_ratio: float, dir_path: str, seq_len: int, stride: float, num_channels: int,
                    per_user_normalization: bool, progression_scale: int, model_dataset_depth_offset: int,
                    per_channel_normalization: bool, no_condition: bool):
        mode = per_user_normalization * 2 + per_channel_normalization * 1
        train_files = None
        train_norms = None
        datasets = [None, None]
        for index, split in enumerate(('train', 'val')):
            target_location = os.path.join(dir_path,
                                           '{}%_{}l_{}c_{}p_{}o_{}m_{}s_{}v_{}.npz'.format(validation_ratio, seq_len,
                                                                                           num_channels,
                                                                                           progression_scale,
                                                                                           model_dataset_depth_offset,
                                                                                           mode,
                                                                                           stride, DATASET_VERSION,
                                                                                           split))
            given_data = None
            if os.path.exists(target_location):
                print('loading {} dataset from file'.format(split))
                if split == 'val' and validation_ratio == 0.0:
                    print('creating {} dataset from scratch'.format(split))
                else:
                    given_data = (load_pkl(target_location + '.pkl'), np.load(target_location))
            else:
                print('creating {} dataset from scratch'.format(split))
            dataset = cls(train_files, train_norms, given_data, validation_ratio, dir_path, seq_len, stride,
                          num_channels, per_user_normalization, progression_scale, per_channel_normalization,
                          no_condition, model_dataset_depth_offset)
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
        if self.y is None or self.no_condition:
            return {'x': x}
        return {'x': x, 'y': torch.from_numpy(self.y[self.data_pointers[item][0]])}

    def alpha_fade(self, datapoint):
        if self.alpha == 1:
            return datapoint
        c, t = datapoint.shape
        t = datapoint.reshape(c, t // self.progression_scale, self.progression_scale).mean(axis=2).repeat(
            self.progression_scale, axis=1)
        return datapoint + (t - datapoint) * (1 - self.alpha)
