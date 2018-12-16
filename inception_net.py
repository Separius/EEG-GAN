import glob
import torch
import pyedflib
import numpy as np
import torch.nn as nn
from tqdm import tqdm

REF_CHANNELS = {'EEG FP1-REF', 'EEG FP2-REF'}
dataset = []
for f_name in tqdm(glob.glob('data/normal/v2.0.0/**/*.edf', recursive=True)):
    f = pyedflib.EdfReader(f_name)
    labels = {label: i for i, label in enumerate(f.getSignalLabels()) if label.lower().startswith('eeg')}
    if len(REF_CHANNELS - set(labels.keys())) == 0:
        is_ok = True
        for channel in REF_CHANNELS:
            if f.samplefrequency(labels[channel]) != 250:
                is_ok = False
                break
        if is_ok:
            dataset.append(np.stack([f.readSignal(labels[channel]) for channel in REF_CHANNELS], axis=0))


class InceptionModule(nn.Module):
    def __init__(self, input_channels, dropout):
        super().__init__()
        self.p1 = nn.Sequential(nn.Conv1d(input_channels, input_channels, kernel_size=3, padding=1, stride=2),
                                nn.BatchNorm1d(input_channels), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.p2 = nn.Sequential(nn.Conv1d(input_channels, input_channels, kernel_size=5, padding=2, stride=2),
                                nn.BatchNorm1d(input_channels), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.p3 = nn.Sequential(nn.Conv1d(input_channels, input_channels, kernel_size=9, padding=4, stride=2),
                                nn.BatchNorm1d(input_channels), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.aggregate = nn.Sequential(nn.Conv1d(3 * input_channels, input_channels, kernel_size=1, padding=0),
                                       nn.BatchNorm1d(input_channels), nn.ReLU(inplace=True))
        self.residual = nn.AvgPool1d(kernel_size=2)

    def forward(self, x):
        return self.residual(x) + self.aggregate(torch.cat([self.p1(x), self.p2(x), self.p3(x)], dim=1))


# TODO train it => implement Inception and FID plugins
class ChronoNet(nn.Module):
    num_block_map = {2 ** (5 + i): i + 1 for i in range(9)}

    def __init__(self, num_channels, seq_len, target_classes=2, network_channels=64, dropout=0.2):
        super().__init__()
        self.num_classes = target_classes
        network = [InceptionModule(network_channels, dropout) for i in range(self.num_block_map[seq_len])]
        self.network = nn.Sequential(nn.Conv1d(num_channels, network_channels, kernel_size=1), *network,
                                     nn.AdaptiveAvgPool1d(1))
        self.linear = nn.Linear(network_channels, target_classes)

    def forward(self, x):
        h = self.network(x).squeeze()
        return self.linear(h), h
