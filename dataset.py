import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from utils import load_pkl, save_pkl, resample_signal, cudize, random_latents

DATASET_VERSION = 7

'''
!pip install imageio
!pip install gdown
#val=0, stride=1(1.5), num_channels=17, start=1
!gdown --id 1Lp5e9vHPeFaVyccT48H0wbGSblrC81Gu
!gdown --id 17QIjg0lkvedcmmpo1YP79OeTpt7TuYNI
'''


# our channels are CHNL = { 'F3', 'F4', 'O1', 'O2', 'CZ' }
# for super res, get F3, O1, CZ and generate F4 and O2
# bio_sampling_freq: 1 -> 4 -> 8 -> 16 -> 24 -> 32 -> 40 -> 60
class EEGDataset(Dataset):
    # for 60(sampling), starting from 1 hz(sampling) [32 samples at the beginning]
    progression_scale_up = [4, 2, 2, 3, 4, 5, 3]
    progression_scale_down = [1, 1, 1, 2, 3, 4, 2]

    # for 60(sampling), starting from 0.25 hz(sampling) [8 samples at the beginning] (also set start_depth)
    # progression_scale_up = [2, 2] + progression_scale_up
    # progression_scale_down = [1, 1] + progression_scale_down

    picked_channels = [3, 5, 9, 15, 16]

    def __init__(self, given_data, dir_path='./data/prepared_eegs_mat_th5/', data_sampling_freq=220,
                 start_sampling_freq=1, end_sampling_freq=60, start_seq_len=32, num_channels=17, return_long=False):
        super().__init__()
        self.model_depth = len(self.progression_scale_up)
        self.alpha = 1.0
        self.dir_path = dir_path
        self.end_sampling_freq = end_sampling_freq
        seq_len = start_seq_len * end_sampling_freq / start_sampling_freq * 1.5
        assert seq_len == int(seq_len), 'seq_len must be an int'
        seq_len = int(seq_len)
        self.seq_len = seq_len
        self.initial_kernel_size = start_seq_len
        self.stride = seq_len
        self.max_dataset_depth = len(self.progression_scale_up)
        self.num_channels = num_channels if self.picked_channels is None else len(self.picked_channels)
        self.return_long = return_long
        if given_data is not None:
            self.seq_len = int(start_seq_len * end_sampling_freq / start_sampling_freq)
            self.data_pointers = given_data[0]
            self.datas = given_data[1]
            return
        all_files = glob.glob(os.path.join(dir_path, '*_1.txt'))
        is_matlab = len(all_files) == 0
        if is_matlab:
            all_files = glob.glob(os.path.join(dir_path, '*.mat'))
        files = [i for i in range(len(all_files))]
        sizes = []
        num_points = []
        self.datas = []
        for i in tqdm(files):
            is_ok = True
            if is_matlab:
                try:
                    tmp = loadmat(all_files[i])['eeg_signal']
                    tmp = resample_signal(tmp, data_sampling_freq, end_sampling_freq, False)
                    size = int(np.ceil((tmp.shape[1] - seq_len + 1) / self.stride))
                except:
                    size = 0
                if size <= 0:
                    is_ok = False
                else:
                    sizes.append(size)
                    num_points.append((sizes[-1] - 1) * self.stride + seq_len)
                    if self.picked_channels is None:
                        self.datas.append(tmp[:num_channels, :num_points[-1]])
                    else:
                        self.datas.append(tmp[self.picked_channels, :num_points[-1]])
            else:
                for_range = range(num_channels) if self.picked_channels is None else self.picked_channels
                for kk, j in enumerate(for_range):
                    with open('{}_{}.txt'.format(all_files[i][:-6], j + 1)) as f:
                        tmp = list(map(float, f.read().split()))
                        tmp = np.array(tmp, dtype=np.float32)
                        tmp = resample_signal(tmp, data_sampling_freq, end_sampling_freq, False)
                        if kk == 0:
                            size = int(np.ceil((len(tmp) - seq_len + 1) / self.stride))
                            if size <= 0:
                                is_ok = False
                                break
                            sizes.append(size)
                            num_points.append((sizes[-1] - 1) * self.stride + seq_len)
                            self.datas.append(np.zeros((num_channels, num_points[-1]), dtype=np.float32))
                        tmp = tmp[:num_points[-1]]
                        self.datas[-1][j, :] = tmp
            if is_ok:
                self.datas[-1], is_ok = self.normalize(self.datas[-1])
                if not is_ok:
                    del sizes[-1]
                    del num_points[-1]
                    del self.datas[-1]
        self.data_pointers = [(i, j) for i, s in enumerate(sizes) for j in range(s)]

    @classmethod
    def from_config(cls, validation_ratio, validation_seed, dir_path, data_sampling_freq,
                    start_sampling_freq, end_sampling_freq, start_seq_len, num_channels, return_long):
        assert end_sampling_freq <= data_sampling_freq
        target_location = os.path.join(dir_path, '{}c_{}v_{}ss_{}es_{}l.npz'.format(num_channels, DATASET_VERSION,
                                                                                    start_sampling_freq,
                                                                                    end_sampling_freq, start_seq_len))
        if os.path.exists(target_location):
            print('loading dataset from file: {}'.format(target_location))
            given_data = np.load(target_location)
            given_data = [load_pkl(target_location + '.pkl'), [given_data['arr_{}'.format(i)]
                                                               for i in range(len(given_data.keys()))]]
        else:
            print('creating dataset from scratch')
            dataset = cls(None, dir_path, data_sampling_freq, start_sampling_freq,
                          end_sampling_freq, start_seq_len, num_channels, return_long)
            np.savez_compressed(target_location, *dataset.datas)
            save_pkl(target_location + '.pkl', dataset.data_pointers)
            given_data = [dataset.data_pointers, dataset.datas]
        return_datasets = []
        for i in range(2):
            return_datasets.append(cls(given_data, dir_path, data_sampling_freq, start_sampling_freq,
                                       end_sampling_freq, start_seq_len, num_channels, return_long))
        data_pointers = [x for x in return_datasets[0].data_pointers]
        np.random.seed(validation_seed)
        np.random.shuffle(data_pointers)
        return_datasets[0].data_pointers = data_pointers[:int((1 - validation_ratio) * len(data_pointers))]
        return_datasets[1].data_pointers = data_pointers[int((1 - validation_ratio) * len(data_pointers)):]
        return return_datasets[0], return_datasets[1]

    @staticmethod
    def normalize(arr):
        arr_max = arr.max()
        arr_min = arr.min()
        is_ok = arr_max != arr_min
        return ((arr - arr_min) / ((arr_max - arr_min) if is_ok else 1.0)) * 2.0 - 1.0, is_ok

    @property
    def shape(self):
        return len(self), self.num_channels, self.seq_len

    def __len__(self):
        return len(self.data_pointers)

    def load_file(self, item):
        i, k = self.data_pointers[item]
        if self.return_long:
            return self.datas[i][:, k * self.stride:(k + 1) * self.stride]
        else:
            rand_shift = np.random.randint(self.stride - self.seq_len)
            return self.datas[i][:, k * self.stride + rand_shift:k * self.stride + rand_shift + self.seq_len]

    def resample_data(self, data, index, forward=True, alpha_fade=False):
        up_scale = self.progression_scale_up[index - (1 if alpha_fade else 0)]
        down_scale = self.progression_scale_down[index - (1 if alpha_fade else 0)]
        if forward:
            return resample_signal(data, down_scale, up_scale, True)
        return resample_signal(data, up_scale, down_scale, True)

    def __getitem__(self, item):
        with torch.no_grad():
            datapoint = torch.from_numpy(self.load_file(item).astype(np.float32)).unsqueeze(0)
            target_depth = self.model_depth
            if self.max_dataset_depth != target_depth:
                datapoint = self.create_datapoint_from_depth(datapoint, target_depth)
        return {'x': self.alpha_fade(datapoint).squeeze(0)}

    def create_datapoint_from_depth(self, datapoint, target_depth):
        depth_diff = (self.max_dataset_depth - target_depth)
        for index in reversed(list(range(len(self.progression_scale_up)))[-depth_diff:]):
            datapoint = self.resample_data(datapoint, index, False)
        return datapoint

    def alpha_fade(self, datapoint):
        if self.alpha == 1:
            return datapoint
        t = self.resample_data(datapoint, self.model_depth, False, alpha_fade=True)
        t = self.resample_data(t, self.model_depth, True, alpha_fade=True)
        return datapoint + (t - datapoint) * (1 - self.alpha)


def get_collate_real(max_sampling_freq, max_len):
    def collate_real(batch):
        return cudize(default_collate(batch))

    return collate_real


def get_collate_fake(latent_size, z_distribution, collate_real):
    def collate_fake(batch):
        batch = collate_real(batch)  # extract condition(features)
        batch['z'] = random_latents(batch['x'].size(0), latent_size, z_distribution)
        return batch

    return collate_fake
