import json
from utils import cudize, num_params, dict_add, divide_dict, AttrDict
from cpc.cpc_network import Network
from cpc.cpc_loss import calc_iic_stats
from dataset import EEGDataset

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau

hp = AttrDict(
    validation_seed=12658,
    validation_ratio=0.1,
    prediction_k=4,
    cross_entropy=True,
    causal_prediction=True,
    use_transformer=False,
    residual_encoder=False,  # I like to set it to True
    rnn_hidden_multiplier=2,  # I like to set it to 1
    global_mode='mlp',  # I like to set it to bilinear and remove this hyper param
    local_mode='mlp',  # I like to set it to bilinear and remove this hyper param
    num_z_iic_classes=0,
    num_c_iic_classes=0,
    use_scheduler=True,
    use_bert_adam=False,
    bidirectional=False,
    prediction_loss_weight=3.0,
    global_loss_weight=1.0,
    local_loss_weight=2.0,
    z_iic_loss_weight=0.0,
    c_iic_loss_weight=0.0,
    contextualizer_num_layers=1,
    contextualizer_dropout=0,
    encoder_dropout=0.2,  # used to be 0.1
    batch_size=128,
    lr=2e-3,
    epochs=31,
    weight_decay=0.005,
)


def main(summary):
    have_iic = (hp.z_iic_loss_weight != 0.0) or (hp.c_iic_loss_weight != 0.0)
    train_dataset, val_dataset = EEGDataset.from_config(validation_ratio=hp.validation_ratio,
                                                        validation_seed=hp.validation_seed,
                                                        dir_path='./data/prepared_eegs_mat_th5', data_sampling_freq=220,
                                                        start_sampling_freq=1, end_sampling_freq=60, start_seq_len=32,
                                                        num_channels=17, return_long=have_iic)
    real_bs = hp.batch_size // (2 if have_iic else 1)
    train_dataloader = DataLoader(train_dataset, batch_size=real_bs, num_workers=0, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=real_bs, num_workers=0, drop_last=False, pin_memory=True)
    network = cudize(
        Network(train_dataset.num_channels, encoder_dropout=hp.encoder_dropout, bidirectional=hp.bidirectional,
                contextualizer_num_layers=hp.contextualizer_num_layers,
                contextualizer_dropout=hp.contextualizer_dropout, use_transformer=hp.use_transformer,
                causal_prediction=hp.causal_prediction,
                prediction_k=hp.prediction_k * (hp.prediction_loss_weight != 0.0),
                have_global=(hp.global_loss_weight != 0.0), have_local=(hp.local_loss_weight != 0.0),
                residual_encoder=hp.residual_encoder, rnn_hidden_multiplier=hp.rnn_hidden_multiplier,
                global_mode=hp.global_mode, local_mode=hp.local_mode,
                num_z_iic_classes=hp.num_z_iic_classes * (hp.z_iic_loss_weight != 0.0),
                num_c_iic_classes=hp.num_c_iic_classes * (hp.c_iic_loss_weight != 0.0)))
    num_parameters = num_params(network)
    print('num_parameters', num_parameters)
    if hp.use_bert_adam:
        network_optimizer = BertAdam(network.parameters(), lr=hp.lr, weight_decay=hp.weight_decay,
                                     warmup=0.2, t_total=hp.epochs * len(train_dataloader), schedule='warmup_linear')
    else:
        network_optimizer = Adam(network.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    if hp.use_scheduler:
        scheduler = ReduceLROnPlateau(network_optimizer, patience=3, verbose=True)
    best_val_loss = float('inf')
    for epoch in range(hp.epochs):
        for training, data_loader in zip((False, True), (val_dataloader, train_dataloader)):
            if training:
                if epoch == hp.epochs - 1:
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
                    x = cudize(batch['x'])
                    network_return = network.forward(x, input_is_long=have_iic, no_loss=False)
                    network_loss = hp.prediction_loss_weight * network_return.losses.prediction_
                    network_loss = network_loss + hp.global_loss_weight * network_return.losses.global_
                    network_loss = network_loss + hp.local_loss_weight * network_return.losses.local_
                    network_loss = network_loss + hp.z_iic_loss_weight * network_return.losses.z_iic_
                    network_loss = network_loss + hp.c_iic_loss_weight * network_return.losses.c_iic_

                    bs = x.size(0)
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
            if network.prediction_loss_network.k > 0 and hp.prediction_loss_weight != 0:
                metrics.update(dict(avg_prediction_acc=total_pred_acc, prediction_loss=total_prediction_loss,
                                    k_prediction_acc=total_k_pred_acc))
            if hp.global_loss_weight != 0:
                metrics.update(dict(global_loss=total_global_loss, global_acc=total_global_accuracy))
            if hp.local_loss_weight != 0:
                metrics.update(dict(local_loss=total_local_loss, local_acc=total_local_accuracy))
            if hp.hp.z_iic_loss_weight != 0:
                metrics.update(dict(z_iic_loss=total_iic_z_loss, z_iic_acc=total_iic_z_acc))
            if hp.hp.c_iic_loss_weight != 0:
                metrics.update(dict(c_iic_loss=total_iic_c_loss, c_iic_acc=total_iic_c_acc))
            divide_dict(metrics, total_count)

            if not training and hp.use_scheduler:
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


if __name__ == '__main__':
    main(summary=False)
