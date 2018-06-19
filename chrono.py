import torch
import torch.nn as nn
from utils import cnn2rnn, rnn2cnn


class InceptionModule(nn.Module):
    def __init__(self, input_channels, num_channels):
        super(InceptionModule, self).__init__()
        self.c = nn.ModuleList([nn.Conv1d(input_channels, num_channels, kernel_size=2 ** (i+1), stride=2, padding=3 if i == 2 else i) for i in range(3)])
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(torch.cat([m(x) for m in self.c], dim=1))


class ChronoNet(nn.Module):
    def __init__(self, input_channels, num_inceptions_modules, num_channels, num_gru_layers, num_classes,
                 temporal_pooling=False):
        super(ChronoNet, self).__init__()
        self.inceptions = nn.Sequential(
            *[InceptionModule(input_channels if i == 0 else (3 * num_channels), num_channels) for i in
              range(num_inceptions_modules)])
        self.grus = nn.ModuleList(
            [nn.GRU(num_channels * (3 if i == 0 else i), num_channels) for i in range(num_gru_layers)])
        self.temporal_pooling = temporal_pooling
        if temporal_pooling:
            self.fc = nn.Conv1d(num_channels, num_classes, kernel_size=1)
        else:
            self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        h = cnn2rnn(self.inceptions(x))
        for i, gru in enumerate(self.grus):
            h, _ = gru(h if i == 0 else t)
            if i == 0:
                t = h
            elif i != len(self.grus) - 1:
                t = torch.cat((t, h), dim=2)
        if self.temporal_pooling:
            return self.fc(rnn2cnn(h)).mean(dim=-1)
        return self.fc(h[-1, :, :])


if __name__ == '__main__':
    a = ChronoNet(22, 3, 32, 4, 2, False)
    b = torch.randn(32, 22, 15000)
    print(b.shape, a(b).shape)
