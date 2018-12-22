import torch
import torch.nn as nn


def get_conv(input_channels, kernel_size, dropout):
    return nn.Sequential(
        nn.Conv1d(input_channels, input_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=2),
        nn.BatchNorm1d(input_channels), nn.ReLU(inplace=True), nn.Dropout(dropout))


class InceptionModule(nn.Module):
    def __init__(self, input_channels, dropout):
        super().__init__()
        self.p1 = get_conv(input_channels, 3, dropout)
        self.p2 = get_conv(input_channels, 5, dropout)
        self.p3 = get_conv(input_channels, 9, dropout)
        self.aggregate = nn.Sequential(nn.Conv1d(3 * input_channels, input_channels, kernel_size=1, padding=0),
                                       nn.BatchNorm1d(input_channels), nn.ReLU(inplace=True))
        self.residual = nn.AvgPool1d(kernel_size=2)

    def forward(self, x):
        return self.residual(x) + self.aggregate(torch.cat([self.p1(x), self.p2(x), self.p3(x)], dim=1))


# TODO isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/
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
