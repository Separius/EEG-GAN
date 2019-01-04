import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import EEGDataset
from torch.utils.data import DataLoader
from utils import parse_config, cudize, trainable_params, mkdir, num_params
from torch.optim import Adam, SGD, ASGD, Adadelta, Adagrad, Adamax, Rprop, RMSprop

default_params = {
    'config_file': None,
    'num_data_workers': 1,
    'random_seed': 1373,
    'cuda_device': 0,
    'minibatch_size': 256,
    'num_epochs': 20,
    'save_location': './results/inception.pth',
    'tiny_sizes': False,
    'age_weight': 0.0,
    'attr_weight': 1.0,
    'single_attr': None,
    'norm': 'batch',  # batch or None or layer or group or instance
    'spectral': False,
    'weight_norm': False,
    'lr': 0.001,
    'weight_decay': 0.05,
    'optimizer': 'adam'  # adam or sgd or asgd
}


def weight(m):
    if params['weight_norm']:
        return nn.utils.weight_norm(m)
    return m


def spectral(m):
    if params['spectral']:
        return nn.utils.spectral_norm(m)
    return m


def norm(input_channels):
    mode = params['norm']
    if mode == 'batch':
        return nn.BatchNorm1d(input_channels)
    elif mode == 'layer':
        return nn.GroupNorm(1, input_channels)
    elif mode == 'instance':
        return nn.GroupNorm(input_channels, input_channels)
    elif mode == 'group':
        return nn.GroupNorm(input_channels // 2, input_channels)
    return nn.Sequential()


def get_conv(input_channels, kernel_size, stride):
    return nn.Sequential(weight(spectral(
        nn.Conv1d(input_channels, input_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                  stride=stride))), nn.ReLU(inplace=True), norm(input_channels))


class InceptionModule(nn.Module):
    k_size = {0: 3, 1: 5, 2: 9, 3: 15}

    def __init__(self, input_channels, stride, num_branches):
        super().__init__()
        self.conv = nn.ModuleList([get_conv(input_channels, self.k_size[i], stride) for i in range(num_branches)])
        self.aggregate = nn.Sequential(
            weight(spectral(nn.Conv1d(len(self.conv) * input_channels, input_channels, kernel_size=1, padding=0))),
            nn.ReLU(inplace=True), norm(input_channels))
        self.residual = nn.AvgPool1d(kernel_size=stride)

    def forward(self, x):
        return self.residual(x) + self.aggregate(torch.cat([conv(x) for conv in self.conv], dim=1))


class ChronoNet_old(nn.Module):
    num_block_map = {2 ** (8 + i): i + 1 for i in range(9)}

    def __init__(self, num_channels, seq_len, target_classes, network_channels=8, stride=4, num_branches=2):
        super().__init__()
        self.num_classes = target_classes
        network = [InceptionModule(network_channels, stride, num_branches) for i in range(self.num_block_map[seq_len])]
        print('num_blocks', len(network))
        self.network = nn.Sequential(weight(spectral(nn.Conv1d(num_channels, network_channels, kernel_size=1))),
                                     *network, nn.AdaptiveAvgPool1d(1))
        self.linear = weight(spectral(nn.Linear(network_channels, target_classes)))

    def forward(self, x):
        h = self.network(x).squeeze()
        return self.linear(h), h


class ChronoNet(nn.Module):
    num_block_map = {2 ** (8 + i): i + 1 for i in range(9)}

    def __init__(self, num_channels, seq_len, target_classes, network_channels=8, stride=4, num_branches=2):
        super().__init__()
        self.num_classes = target_classes
        self.network = nn.Sequential(nn.Conv1d(num_channels, 8, kernel_size=1), get_conv(8, kernel_size=3, stride=1),
                                     get_conv(8, kernel_size=3, stride=1), nn.MaxPool1d(kernel_size=4),
                                     get_conv(8, kernel_size=3, stride=1), get_conv(8, kernel_size=3, stride=1),
                                     nn.MaxPool1d(kernel_size=4), get_conv(8, kernel_size=3, stride=1),
                                     get_conv(8, kernel_size=3, stride=1), nn.MaxPool1d(kernel_size=2),
                                     get_conv(8, kernel_size=3, stride=1), get_conv(8, kernel_size=3, stride=1),
                                     nn.MaxPool1d(kernel_size=2), get_conv(8, kernel_size=3, stride=1),
                                     get_conv(8, kernel_size=3, stride=1), nn.MaxPool1d(kernel_size=2),
                                     nn.AdaptiveAvgPool1d(1))
        self.linear = weight(spectral(nn.Linear(network_channels, target_classes)))

    def forward(self, x):
        h = self.network(x).squeeze()
        return self.linear(h), h


def calc_loss(x):
    y_pred, _ = network(cudize(x['x']))
    y = cudize(x['y'])
    acc = None
    if params['age_weight'] != 0.0:
        loss_age = loss_function_age(y_pred[:, 0], y[:, 0])
    else:
        loss_age = 0.0
    if params['attr_weight'] != 0.0:
        start_index = 1 if params['age_weight'] != 0.0 else 0
        if params['single_attr'] is None:
            loss_attr = loss_function_attrs(y_pred[:, start_index:], y[:, 1:]) * num_attrs
        else:
            loss_attr = loss_function_attrs(y_pred[:, start_index], y[:, 1 + params['single_attr']])
            res = accuracy(y_pred[:, start_index], y[:, 1 + params['single_attr']])
            acc = res
    else:
        loss_attr = 0.0
    return loss_attr * params['attr_weight'] + loss_age * params['age_weight'], acc


def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = torch.sigmoid(output) >= 0.5
    truth = target >= 0.5
    acc = pred.eq(truth).float().mean()
    return acc


if __name__ == '__main__':
    params = parse_config(default_params, [EEGDataset, ChronoNet], False)
    train_dataset, val_dataset = EEGDataset.from_config(**params['EEGDataset'])
    depth = train_dataset.max_dataset_depth - train_dataset.model_dataset_depth_offset
    train_dataset.model_depth = val_dataset.model_depth = depth
    train_dataset.alpha = val_dataset.alpha = 1.0
    mkdir('./results')

    train_dataloader = DataLoader(train_dataset, params['minibatch_size'], shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, params['minibatch_size'], shuffle=False, drop_last=False)

    output_size = train_dataset.y.shape[1]
    if params['attr_weight'] == 0.0:
        output_size = 1
    elif params['single_attr'] is not None:
        output_size = 2
        print('majority voting', params['single_attr'], train_dataset.y[:, 1 + params['single_attr']].mean(),
              val_dataset.y[:, 1 + params['single_attr']].mean())
    if params['age_weight'] == 0.0:
        output_size -= 1
    assert output_size > 0
    network = cudize(ChronoNet(train_dataset.num_channels, train_dataset.seq_len, output_size,
                               **params['ChronoNet']))
    network.train()
    num_attrs = train_dataset.y.shape[1] - 1
    print('num_attrs', num_attrs)
    print('output_size', output_size)
    print('num_params', num_params(network))
    print('train_size', train_dataset.shape)
    optimizer = {'adam': Adam, 'sgd': SGD, 'asgd': ASGD,
                 'adadelta': Adadelta, 'adagrad': Adagrad, 'adamax': Adamax,
                 'rprop': Rprop, 'rmsprop': RMSprop}[params['optimizer'].lower()]
    optimizer = optimizer(trainable_params(network), lr=params['lr'], weight_decay=params['weight_decay'])
    loss_function_age = nn.MSELoss()
    loss_function_attrs = nn.BCEWithLogitsLoss()
    best_loss = None
    epochs_tqdm = tqdm(range(params['num_epochs']), total=params['num_epochs'], dynamic_ncols=True)
    for _ in epochs_tqdm:
        network.train()
        train_tqdm = tqdm(train_dataloader, dynamic_ncols=True)
        for i, x in enumerate(train_tqdm):
            loss, _ = calc_loss(x)
            network.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                train_tqdm.set_description('training loss ' + str(loss.item()))
        train_tqdm.close()
        network.eval()
        with torch.no_grad():
            total_loss = 0
            total_acc = 0
            for x in tqdm(val_dataloader, dynamic_ncols=True):
                loss, acc = calc_loss(x)
                total_acc += acc.item()
                total_loss += loss.item()
            new_loss = total_loss / len(val_dataloader)
            new_acc = total_acc / len(val_dataloader)
            epochs_tqdm.set_description('validation loss ' + str(new_loss) + ' %' + str(new_acc))
            if best_loss is None or new_loss < best_loss:
                torch.save(network, params['save_location'])
                best_loss = new_loss
    print('min_val_loss', best_loss)
    epochs_tqdm.close()
