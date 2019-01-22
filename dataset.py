import glob
import os
from collections import OrderedDict
from random import shuffle
from typing import List, Optional, Callable

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from tqdm import tqdm, trange

from utils import load_pkl, save_pkl, downsample_signal, upsample_signal, resample_signal

DATASET_VERSION = 6


# real_freq: 0.0625(4 samples) -> 0.125 -> 0.25 -> 0.5 -> 2 -> 4 -> 8 -> 12 -> 16 -> 20 -> 30 -> 50 -> 100(6400 samples)
# real_freq: 0.0625(4 samples) -> 0.5(32 samples) -> 2 -> 4 -> 8 -> 12 -> 16 -> 20 -> 30 -> 50 -> 100(6400 samples)
# min_layer_real_freq: 0.5(32 samples) -> 4 -> 8 -> 12 -> 30 -> 50 -> 100(6400 samples)
# ideal_real_freq: (0.5 -> 2)( -> 4)( -> 8)( -> 12)( -> 16)( -> 20)( -> 30)( -> 50)( -> 100)
class EEGDataset(Dataset):
    # for 200(sampling)
    # progression_scale_up = [4, 2, 2, 3, 4, 5, 3, 5, 2]
    # progression_scale_down = [1, 1, 1, 2, 3, 4, 2, 3, 1]

    # for 60(sampling)
    progression_scale_up = [4, 2, 2, 3, 4, 5, 3]
    progression_scale_down = [1, 1, 1, 2, 3, 4, 2]

    # global_cond and local_cond will get a pytorch tensor of shape (num_channels, current_seq_len)
    # and outputs (ch or (ch, new_seq_len based on current_seq_len or just a constant length))
    def __init__(self, train_files, norms, given_data, validation_ratio: float = 0.1, dir_path: str = './data/tuh1',
                 data_sampling_freq: float = 80.0, start_sampling_freq: float = 1.0, end_sampling_freq: float = 60.0,
                 start_seq_len: int = 32, stride: float = 0.25, num_channels: int = 5,
                 per_user_normalization: bool = True, per_channel_normalization: bool = False,
                 global_cond: Optional[List[Callable]] = None, temporal_cond: Optional[List[Callable]] = None):
        super().__init__()
        self.model_depth = 0
        self.alpha = 1.0
        self.dir_path = dir_path
        seq_len = start_seq_len * end_sampling_freq / start_sampling_freq
        assert seq_len == int(seq_len), 'seq_len must be an int'
        seq_len = int(seq_len)
        self.seq_len = seq_len
        self.start_seq_len = start_seq_len
        self.stride = int(seq_len * stride)
        self.per_user_normalization = per_user_normalization
        self.per_channel_normalization = per_channel_normalization
        self.max_dataset_depth = len(self.progression_scale_up)
        self.norms = norms
        self.global_cond = global_cond
        self.temporal_cond = temporal_cond
        self.num_channels = num_channels
        if given_data is not None:
            self.sizes = given_data[0]['sizes']
            self.files = given_data[0]['files']
            self.norms = given_data[0]['norms']
            self.data_pointers = given_data[0]['pointers']
            self.datas = [given_data[1]['arr_{}'.format(i)] for i in trange(len(given_data[1].keys()))]
            return
        all_files = glob.glob(os.path.join(dir_path, '*_1.txt'))
        is_matlab = len(all_files) == 0
        if is_matlab:
            all_files = glob.glob(os.path.join(dir_path, '*.mat'))
        files = len(all_files)
        files = [i for i in range(files)]
        if train_files is None:
            shuffle(files)
            files = files[:int(len(all_files) * (1.0 - validation_ratio))]
        else:
            files = list(set(files) - set(train_files))
        self.files = files
        sizes = []
        num_points = []
        self.datas = []
        for i in tqdm(files):
            is_ok = True
            if is_matlab:
                tmp = loadmat(all_files[i])['eeg_signal']
                tmp = resample_signal(tmp, data_sampling_freq, end_sampling_freq)
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
                        tmp = np.array(tmp, dtype=np.float32)
                        tmp = resample_signal(tmp, data_sampling_freq, end_sampling_freq)
                        if j == 0:
                            size = int(np.ceil((len(tmp) - seq_len + 1) / self.stride))
                            if size <= 0:
                                is_ok = False
                                break
                            sizes.append(size)
                            num_points.append((sizes[-1] - 1) * self.stride + seq_len)
                            self.datas.append(np.zeros((num_channels, num_points[-1]), dtype=np.float32))
                        tmp = tmp[:num_points[-1]]
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
    def from_config(cls, validation_ratio: float, dir_path: str,
                    data_sampling_freq: float, start_sampling_freq: float, end_sampling_freq: float,
                    start_seq_len: int, stride: float, num_channels: int,
                    per_user_normalization: bool, per_channel_normalization: bool,
                    global_cond: Optional[List[Callable]], temporal_cond: Optional[List[Callable]]):
        mode = per_user_normalization * 2 + per_channel_normalization * 1
        train_files = None
        train_norms = None
        datasets = [None, None]
        for index, split in enumerate(('train', 'val')):
            target_location = os.path.join(dir_path, '{}%_{}c_{}m_{}s_{}v_{}ss_{}es_{}l_{}.npz'.format(validation_ratio,
                                                                                                       num_channels,
                                                                                                       mode, stride,
                                                                                                       DATASET_VERSION,
                                                                                                       start_sampling_freq,
                                                                                                       end_sampling_freq,
                                                                                                       start_seq_len,
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
            dataset = cls(train_files, train_norms, given_data, validation_ratio, dir_path, data_sampling_freq,
                          start_sampling_freq, end_sampling_freq, start_seq_len, stride, num_channels,
                          per_user_normalization, per_channel_normalization, global_cond, temporal_cond)
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

    def load_file(self, item):
        i, k = self.data_pointers[item]
        res = self.datas[i][:, k * self.stride:k * self.stride + self.seq_len]
        return res

    def resample_data(self, data, index, forward=True):
        up_scale = self.progression_scale_up[index]
        down_scale = self.progression_scale_down[index]
        t = upsample_signal(data, up_scale if forward else down_scale)
        return downsample_signal(t, down_scale if forward else up_scale)

    def __getitem__(self, item):
        with torch.no_grad():
            datapoint = torch.from_numpy(self.load_file(item).astype(np.float32)).unsqueeze(0)
            target_depth = self.model_depth
            if self.max_dataset_depth != target_depth:
                datapoint = self.create_datapoint_from_depth(datapoint, target_depth)
        x = self.alpha_fade(datapoint).squeeze(0)
        res = OrderedDict()
        res['x'] = x
        if self.global_cond is not None:
            for i, global_cond_function in enumerate(self.global_cond):
                res['global_{}'.format(i)] = global_cond_function(x)
        if self.temporal_cond is not None:
            for i, temporal_cond_function in enumerate(self.temporal_cond):
                res['temporal_{}'.format(i)] = temporal_cond_function(x)
        return res

    def create_datapoint_from_depth(self, datapoint, target_depth):
        depth_diff = (self.max_dataset_depth - target_depth)
        for index in reversed(list(range(len(self.progression_scale_up)))[-depth_diff:]):
            datapoint = self.resample_data(datapoint, index, False)
        return datapoint

    def alpha_fade(self, datapoint):
        if self.alpha == 1:
            return datapoint
        t = self.resample_data(datapoint, self.model_depth, False)
        t = self.resample_data(t, self.model_depth, True)
        return datapoint + (t - datapoint) * (1 - self.alpha)


if __name__ == '__main__':
    a = EEGDataset(None, None, None, dir_path='./data/prepared_sample')
    # TODO global and local conds
