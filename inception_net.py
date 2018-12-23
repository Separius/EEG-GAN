import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange, tqdm
from dataset import EEGDataset
from torch.utils.data import DataLoader
from utils import parse_config, cudize, random_onehot, trainable_params

default_params = {
    'config_file': None,
    'cpu_deterministic': True,
    'num_data_workers': 1,
    'random_seed': 1373,
    'cuda_device': 0,
    'minibatch_size': 32,
    'num_epochs': 20,
    'save_location': './results/inception.pth',
    'tiny_sizes': False
}


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


class ChronoNet(nn.Module):
    num_block_map = {2 ** (5 + i): i + 1 for i in range(9)}

    def __init__(self, num_channels, seq_len, target_classes, network_channels=64, dropout=0.2):
        super().__init__()
        self.num_classes = target_classes
        network = [InceptionModule(network_channels, dropout) for i in range(self.num_block_map[seq_len])]
        self.network = nn.Sequential(nn.Conv1d(num_channels, network_channels, kernel_size=1), *network,
                                     nn.AdaptiveAvgPool1d(1))
        self.linear = nn.Linear(network_channels, target_classes)

    def forward(self, x):
        h = self.network(x).squeeze()
        return self.linear(h), h


if __name__ == '__main__':
    params = parse_config(default_params, [EEGDataset, ChronoNet, Adam])
    train_dataset, val_dataset = EEGDataset.from_config(**params['EEGDataset'])
    depth = train_dataset.max_dataset_depth - train_dataset.model_dataset_depth_offset
    train_dataset.model_depth = val_dataset.model_depth = depth
    train_dataset.alpha = val_dataset.alpha = 1.0
    if params['tiny_sizes']:
        train_dataset.class_options = [2]
        train_dataset.y = random_onehot(2, len(train_dataset))
        val_dataset.class_options = [2]
        val_dataset.y = random_onehot(2, len(val_dataset))

    train_dataloader = DataLoader(train_dataset, params['minibatch_size'], shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, params['minibatch_size'], shuffle=False, drop_last=False)

    network = cudize(ChronoNet(train_dataset.num_channels, train_dataset.seq_len, train_dataset.class_options[0],
                               **params['ChronoNet']))
    network.train()
    optimizer = Adam(trainable_params(network), **params['Adam'])
    loss_function = nn.CrossEntropyLoss()
    best_loss = None

    for i in trange(params['num_epochs']):
        network.train()
        for x in tqdm(train_dataloader):
            y_pred, _ = network(cudize(x['x']))
            y = torch.argmax(x['y'], dim=1)
            loss = loss_function(y_pred, cudize(y))
            network.zero_grad()
            loss.backward()
            optimizer.step()
        network.eval()
        total_loss = 0
        for i, x in enumerate(tqdm(val_dataloader)):
            y_pred, _ = network(cudize(x['x']))
            y = torch.argmax(x['y'], dim=1)
            loss = loss_function(y_pred, cudize(y))
            total_loss += loss.item()
        new_loss = total_loss / i
        if best_loss is None or new_loss < best_loss:
            torch.save(network, params['save_location'])
            best_loss = new_loss
