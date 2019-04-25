import json

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from sklearn.metrics import confusion_matrix
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from torchvision.transforms import Compose, Normalize, Pad, RandomCrop, ToTensor, RandomHorizontalFlip

from cpc.cpc_train import num_params, dict_add, divide_dict, Adam, BertAdam, ReduceLROnPlateau, calc_iic_stats
from cpc.cpc_network import (PNormPooling, AutoRNN, KPredLoss, OneOneMI, SeqOneMI, NetworkOutput,
                             NetworkLatents, NetworkAccuracies, NetworkLosses, Network)


class GridCifar10(CIFAR10):
    default_transform = Compose([Pad(4, padding_mode='symmetric')])
    norm = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    iic_transforms = [Compose([RandomCrop(32), ToTensor(), norm]),
                      Compose([RandomCrop(32), RandomHorizontalFlip(1.0), ToTensor(), norm])]

    def __init__(self, train, download=False):
        super().__init__(root='~/.torch/data', train=train, download=download, transform=self.default_transform)

    @staticmethod
    def split_image(image):
        return torch.stack([image[:, i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] for i in range(4) for j in range(4)], dim=0)

    @staticmethod
    def prepare_parts(image, iic_idx):
        x = GridCifar10.split_image(GridCifar10.iic_transforms[iic_idx](image))
        x = F.pad(x, (1, 1, 1, 1), 'reflect')
        x_offset, y_offset = np.random.randint(3), np.random.randint(3)
        return x[:, :, x_offset:x_offset + 8, y_offset:y_offset + 8]

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        rand_idx = np.random.randint(2)
        return self.prepare_parts(image, rand_idx), self.prepare_parts(image, 1 - rand_idx), target


class MyResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super().__init__()
        in_planes = 32
        self.inplanes = in_planes
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, in_planes, layers[0])
        self.layer2 = self._make_layer(block, in_planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.z_size = in_planes * 8

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class CifarNetwork(nn.Module):
    def __init__(self, have_global, have_local, have_z_iic, have_c_iic,
                 prediction_k=5, contextualizer_num_layers=1, my_iic=False):
        super().__init__()
        encoder = MyResNet(BasicBlock, [2, 2, 2, 1], zero_init_residual=True)
        z_pooler = PNormPooling(encoder.z_size, be_mean_pool=not have_global)
        if have_z_iic:
            self.z_iic = nn.Sequential(nn.Linear(z_pooler.pool_size, 10), nn.Softmax(-1))
        else:
            self.z_iic = None
        if not have_global and not have_local and not have_c_iic:
            contextualizer = None
            c_pooler = None
            c_pooler_pool_size = 0
            contextualizer_c_size = 0
        else:
            contextualizer = AutoRNN(input_size=encoder.z_size, hidden_size=encoder.z_size // 2,
                                     bidirectional=False, num_layers=contextualizer_num_layers,
                                     dropout=0 if contextualizer_num_layers == 1 else 0.1, cell_type='GRU')
            c_pooler = PNormPooling(contextualizer.c_size, be_mean_pool=not (have_local or have_global))
            c_pooler_pool_size = c_pooler.pool_size
            contextualizer_c_size = contextualizer.c_size
        if have_c_iic:
            self.c_iic = nn.Sequential(nn.Linear(c_pooler_pool_size, 10), nn.Softmax(-1))
        else:
            self.c_iic = None

        prediction_loss_network = KPredLoss(contextualizer_c_size, encoder.z_size, k=prediction_k,
                                            auto_is_bidirectional=False, look_both=False)
        if have_global:
            c_pooled_mi_z_pooled = OneOneMI(c_pooler_pool_size, z_pooler.pool_size, mode='bilinear',
                                            hidden_size=min(c_pooler_pool_size, z_pooler.pool_size) * 2)
        else:
            c_pooled_mi_z_pooled = None
        if have_local:
            c_pooled_mi_z = SeqOneMI(c_pooler_pool_size, encoder.z_size, mode='bilinear',
                                     hidden_size=min(c_pooler_pool_size, encoder.z_size) * 2)
        else:
            c_pooled_mi_z = None

        self.encoder = encoder
        self.z_pooler = z_pooler
        self.contextualizer = contextualizer
        self.c_pooler = c_pooler
        self.prediction_loss_network = prediction_loss_network
        self.c_pooled_mi_z_pooled = c_pooled_mi_z_pooled
        self.c_pooled_mi_z = c_pooled_mi_z

    def half_forward(self, x, no_loss):
        xs = x.size()
        assert xs == (xs[0], 16, 3, 8, 8)
        x = x.view(-1, xs[2], xs[3], xs[4])  # bs*16, 3, 8, 8
        x_encoded = self.encoder(x).view(xs[0], 16, -1)  # bs, 16, z_size
        z = x_encoded.permute(0, 2, 1)
        z_pooled = self.z_pooler(z)
        if self.z_iic is not None:
            z_iic = self.z_iic(z_pooled)
        else:
            z_iic = None
        if self.contextualizer is not None:
            c = self.contextualizer(z)
            c_pooled = self.c_pooler(c)
        else:
            c = None
            c_pooled = None
        if self.c_iic is not None:
            c_iic = self.c_iic(c_pooled)
        else:
            c_iic = None
        if no_loss:
            return NetworkOutput(losses=None, accuracies=None,
                                 latents=NetworkLatents(z, c, z_pooled, c_pooled, z_iic, c_iic))
        prediction_loss, pred_acc = self.prediction_loss_network(c, z)
        if self.c_pooled_mi_z_pooled is not None:
            global_loss, global_accuracy = self.c_pooled_mi_z_pooled(c_pooled, z_pooled)
        else:
            global_loss, global_accuracy = torch.tensor(0.0).to(x), 0.0
        if self.c_pooled_mi_z is not None:
            local_loss, local_accuracy = self.c_pooled_mi_z(c_pooled, z)
        else:
            local_loss, local_accuracy = torch.tensor(0.0).to(x), 0.0
        return NetworkOutput(losses=NetworkLosses(global_loss, local_loss, prediction_loss,
                                                  torch.tensor(0.0).to(x), torch.tensor(0.0).to(x)),
                             accuracies=NetworkAccuracies(global_accuracy, local_accuracy, pred_acc, 0.0, 0.0),
                             latents=NetworkLatents(z, c, z_pooled, c_pooled, z_iic, c_iic))

    def forward(self, x, xp, no_loss=False):
        x1_res = self.half_forward(x, no_loss)
        if (self.z_iic is None and self.c_iic is None) or no_loss:
            return x1_res
        with torch.no_grad():
            x2_res = self.half_forward(xp, no_loss)
        iic_loss_z, iic_accuracy_z = Network.calculate_iic_stats(x1_res.latents.z_iic, x2_res.latents.z_iic, x)
        iic_loss_c, iic_accuracy_c = Network.calculate_iic_stats(x1_res.latents.c_iic, x2_res.latents.c_iic, x)
        return NetworkOutput(losses=NetworkLosses((x1_res.losses.global_ + x2_res.losses.global_) / 2,
                                                  (x1_res.losses.local_ + x2_res.losses.local_) / 2,
                                                  (x1_res.losses.prediction_ + x2_res.losses.prediction_) / 2,
                                                  iic_loss_z, iic_loss_c),
                             accuracies=NetworkAccuracies(
                                 (x1_res.accuracies.global_ + x2_res.accuracies.global_) / 2,
                                 (x1_res.accuracies.local_ + x2_res.accuracies.local_) / 2,
                                 {k: (v + x2_res.accuracies.prediction_[k]) / 2
                                  for k, v in x1_res.accuracies.prediction_.items()},
                                 iic_accuracy_z, iic_accuracy_c), latents=x1_res.latents)


def get_us_acc(network, train_dataloader, val_dataloader, device, k, have_z_iic, have_c_iic):
    with torch.no_grad():
        for j, dl in enumerate([train_dataloader, val_dataloader]):
            preds_z = []
            preds_c = []
            targets = []
            for batch in dl:
                x, xp, target = batch
                x = x.to(device)
                xp = xp.to(device)
                network_return = network.forward(x, xp, no_loss=True)
                if have_z_iic:
                    preds_z.append(network_return.latents.z_iic.argmax(dim=1).cpu().numpy())
                if have_c_iic:
                    preds_c.append(network_return.latents.c_iic.argmax(dim=1).cpu().numpy())
                targets.append(target.numpy())
            if j == 0:
                targets_train = np.concatenate(targets, axis=0)
                if have_z_iic:
                    preds_z_train = np.concatenate(preds_z, axis=0)
                if have_c_iic:
                    preds_c_train = np.concatenate(preds_c, axis=0)
            else:
                targets_val = np.concatenate(targets, axis=0)
                if have_z_iic:
                    preds_z_val = np.concatenate(preds_z, axis=0)
                if have_c_iic:
                    preds_c_val = np.concatenate(preds_c, axis=0)
    if have_z_iic:
        res = {'z_' + k: v for k, v in
               calc_iic_stats(preds_z_train, preds_z_val, targets_train, targets_val, k).items()}
        if have_c_iic:
            res.update({'c_' + k: v for k, v in
                        calc_iic_stats(preds_c_train, preds_c_val, targets_train, targets_val, k).items()})
    else:
        res = {'c_' + k: v for k, v in
               calc_iic_stats(preds_c_train, preds_c_val, targets_train, targets_val, k).items()}
        if have_z_iic:
            res.update({'z_' + k: v for k, v in
                        calc_iic_stats(preds_z_train, preds_z_val, targets_train, targets_val, k).items()})
    return res


def main():
    batch_size = 32
    use_bert_adam = False
    lr = 5e-4
    weight_decay = 0.005
    epochs = 20
    use_scheduler = False
    prediction_loss_weight = 1.0
    global_loss_weight = 1.0
    local_loss_weight = 3.0
    z_iic_loss_weight = 4.0
    c_iic_loss_weight = 8.0
    prediction_k = 8
    summary = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = CifarNetwork(have_global=(global_loss_weight != 0.0), have_local=(local_loss_weight != 0.0),
                           have_z_iic=(z_iic_loss_weight != 0.0), have_c_iic=(c_iic_loss_weight != 0.0),
                           prediction_k=prediction_k, contextualizer_num_layers=1).to(device)
    train_dataloader = DataLoader(GridCifar10(train=True), batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(GridCifar10(train=False), batch_size=batch_size, shuffle=True, drop_last=True)
    num_parameters = num_params(network)
    print('num_parameters', num_parameters)
    if use_bert_adam:
        network_optimizer = BertAdam(network.parameters(), lr=lr, weight_decay=weight_decay,
                                     warmup=0.2, t_total=epochs * len(train_dataloader), schedule='warmup_linear')
    else:
        network_optimizer = Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    if use_scheduler:
        scheduler = ReduceLROnPlateau(network_optimizer, patience=3, verbose=True)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        for training, data_loader in zip((False, True), (val_dataloader, train_dataloader)):
            if training:
                if epoch == epochs - 1:
                    break
                network.train()
            else:
                network.eval()
            total_network_loss = 0.0
            total_prediction_loss = 0.0
            total_global_loss = 0.0
            total_local_loss = 0.0
            total_iic_z_loss = 0.0
            total_iic_c_loss = 0.0
            total_global_accuracy = 0.0
            total_local_accuracy = 0.0
            total_k_pred_acc = {}
            total_pred_acc = 0.0
            total_iic_z_acc = 0.0
            total_iic_c_acc = 0.0
            total_count = 0
            with torch.set_grad_enabled(training):
                for batch in data_loader:
                    x, xp, _ = batch
                    x = x.to(device)
                    xp = xp.to(device)
                    network_return = network.forward(x, xp, no_loss=False)
                    network_loss = prediction_loss_weight * network_return.losses.prediction_
                    network_loss = network_loss + global_loss_weight * network_return.losses.global_
                    network_loss = network_loss + local_loss_weight * network_return.losses.local_
                    network_loss = network_loss + z_iic_loss_weight * network_return.losses.z_iic_
                    network_loss = network_loss + c_iic_loss_weight * network_return.losses.c_iic_

                    bs = batch[0].size(0)
                    total_count += bs
                    total_network_loss += network_loss.item() * bs
                    total_prediction_loss += network_return.losses.prediction_.item() * bs
                    total_global_loss += network_return.losses.global_.item() * bs
                    total_local_loss += network_return.losses.local_.item() * bs
                    total_iic_z_loss += network_return.losses.z_iic_.item() * bs
                    total_iic_c_loss += network_return.losses.c_iic_.item() * bs

                    total_global_accuracy += network_return.accuracies.global_ * bs
                    total_local_accuracy += network_return.accuracies.local_ * bs
                    total_iic_z_acc += network_return.accuracies.z_iic_ * bs
                    total_iic_c_acc += network_return.accuracies.c_iic_ * bs
                    dict_add(total_k_pred_acc, network_return.accuracies.prediction_, bs)
                    len_pred = len(network_return.accuracies.prediction_)
                    if len_pred > 0:
                        total_pred_acc += sum(network_return.accuracies.prediction_.values()) / len_pred * bs

                    if training:
                        network_optimizer.zero_grad()
                        network_loss.backward()
                        network_optimizer.step()

            metrics = dict(net_loss=total_network_loss)
            if network.prediction_loss_network.k > 0 and prediction_loss_weight != 0:
                metrics.update(dict(avg_prediction_acc=total_pred_acc, prediction_loss=total_prediction_loss,
                                    k_prediction_acc=total_k_pred_acc))
            if global_loss_weight != 0:
                metrics.update(dict(global_loss=total_global_loss, global_acc=total_global_accuracy))
            if local_loss_weight != 0:
                metrics.update(dict(local_loss=total_local_loss, local_acc=total_local_accuracy))
            if z_iic_loss_weight != 0:
                metrics.update(dict(z_iic_loss=total_iic_z_loss, z_iic_acc=total_iic_z_acc))
            if c_iic_loss_weight != 0:
                metrics.update(dict(c_iic_loss=total_iic_c_loss, c_iic_acc=total_iic_c_acc))
            divide_dict(metrics, total_count)

            if not training and use_scheduler:
                scheduler.step(metrics['net_loss'])
            if summary:
                print('train' if training else 'validation', epoch, metrics['net_loss'])
            else:
                print('train' if training else 'validation', epoch)
                print(json.dumps(metrics, indent=4))
            if not training and (metrics['net_loss'] < best_val_loss):
                best_val_loss = metrics['net_loss']
                print('update best to', best_val_loss)
                torch.save(network.state_dict(), 'best_network.pth')
                print(json.dumps(get_us_acc(network, train_dataloader, val_dataloader, device, 10,
                                            have_z_iic=(z_iic_loss_weight != 0.0),
                                            have_c_iic=(c_iic_loss_weight != 0.0)), indent=4))
    network.load_state_dict(torch.load('best_network.pth'))
    network.eval()
    print(json.dumps(
        get_us_acc(network, train_dataloader, val_dataloader, device, 10, have_z_iic=(z_iic_loss_weight != 0.0),
                   have_c_iic=(c_iic_loss_weight != 0.0)), indent=4))


if __name__ == '__main__':
    main()
