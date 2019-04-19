# calculate FID of {real, permuted, shifted, zeroed, salt and peppered, concatenated, mode collapsed} * progression
# check our FID calculation's variance over 50 runs compared to their mean (reliability of the metric) * progression
# check our FID over training (improved model generates better results)(also considers progression)
# experiment on samples needed for FID calculation => plot(calculated_on_small, calculated_on_large)
# * all meters?(swd, bins) + different weights experiment
from cpc_train import hp
from ndb import NDB
from cpc_network import Network
from plugins import FidCalculator
from dataset import ThinEEGDataset, EEGDataset
from utils import cudize, AttrDict, resample_signal, dict_add, divide_dict, merge_pred_accs

import torch
import numpy as np
from torch.utils.data import DataLoader


def calc_mean_cov(name, x):
    return {name + '_mean': torch.mean(x, 0), name + '_cov': FidCalculator.torch_cov(x, rowvar=False)}


prediction_hp = AttrDict(generate_long_sequence=True, pool_or_stride='stride', use_shared_sinc=True, seed=12658,
                         prediction_k=6, validation_ratio=0.1, ds_stride=0.5, num_channels=17, use_sinc_encoder=False,
                         causal_prediction=True, use_transformer=False, use_scheduler=True, use_bert_adam=False,
                         bidirectional=False, prediction_loss_weight=1.0, global_loss_weight=0.0, local_loss_weight=0.0,
                         contextualizer_num_layers=1, contextualizer_dropout=0, encoder_dropout=0.2,
                         encoder_activation='relu', tiny_encoder=False, batch_size=128, lr=2e-3, epochs=32,
                         weight_decay=0.01, )

local_hp = AttrDict(generate_long_sequence=True, pool_or_stride='stride', use_shared_sinc=True, seed=12658,
                    prediction_k=1, validation_ratio=0.1, ds_stride=0.5, num_channels=17, use_sinc_encoder=False,
                    causal_prediction=True, use_transformer=False, use_scheduler=True, use_bert_adam=False,
                    bidirectional=False, prediction_loss_weight=0.0, global_loss_weight=0.0, local_loss_weight=1.0,
                    contextualizer_num_layers=1, contextualizer_dropout=0, encoder_dropout=0.2,
                    encoder_activation='relu', tiny_encoder=False, batch_size=128, lr=2e-3, epochs=32,
                    weight_decay=0.01)


def calculate_stats(dataset, net, scale_up, scale_down, skip_depth, num_samples, mode, current_hp, real_ndb):
    samples = {'normal': [], 'permute_0': [], 'permute_1': [], 'permute_2': [], 'permute_3': [], 'shift_0': [],
               'shift_1': [], 'shift_2': [], 'shift_3': [], 'concat': [], 'zero_0': [], 'zero_1': [], 'zero_2': [],
               'zero_3': [], 'zero_4': [], 'noise_1': [], 'noise_2': [], 'noise_3': [], 'collapse': []}
    if mode in ['shift']:
        i = 0
        while i < num_samples:
            prev_id = np.random.randint(len(dataset))
            i1, _ = dataset.data_pointers[prev_id]
            i2, _ = dataset.data_pointers[prev_id + 1]
            if i1 != i2:
                continue
            i += 1
            x = torch.cat([dataset[prev_id], dataset[prev_id + 1]], dim=1)
            start = np.random.randint(dataset.seq_len - 100)
            samples['shift_0'].append(
                torch.cat([x[0:1, start:start + dataset.seq_len], x[1:, start + 10:start + 10 + dataset.seq_len]],
                          dim=1).unsqueeze(0))
            samples['shift_1'].append(
                torch.cat([x[0:2, start:start + dataset.seq_len], x[2:, start + 50:start + 50 + dataset.seq_len]],
                          dim=1).unsqueeze(0))
            samples['shift_2'].append(
                torch.cat([x[-1:, start:start + dataset.seq_len], x[:-1, start + 50:start + 50 + dataset.seq_len]],
                          dim=1).unsqueeze(0))
            samples['shift_3'].append(
                torch.cat([x[-2:, start:start + dataset.seq_len], x[:-2, start + 10:start + 10 + dataset.seq_len]],
                          dim=1).unsqueeze(0))
    elif mode in ['zero']:
        dataloader = DataLoader(dataset, batch_size=current_hp.batch_size, shuffle=True, drop_last=True)
        i = 0
        for b in dataloader:
            bs = b.size(0)
            i += bs
            samples['zero_0'].append(
                torch.cat([b[:, 0:3],
                           torch.zeros(bs, dataset.num_channels - 3, dataset.seq_len)], dim=1))
            samples['zero_1'].append(
                torch.cat([b[:, 0:4],
                           torch.zeros(bs, dataset.num_channels - 4, dataset.seq_len)], dim=1))
            start = np.random.randint(dataset.seq_len // 2)
            duration = np.random.randint(dataset.seq_len // 4)
            samples['zero_2'].append(torch.cat(
                [torch.cat([b[:, 0:1, :start],
                            torch.zeros(bs, 1, duration), b[:, 0:1, start + duration:]], dim=2), b[:, 1:]], dim=1))
            samples['zero_3'].append(torch.cat(
                [torch.cat([b[:, 0:2, :start],
                            torch.zeros(bs, 2, duration), b[:, 0:2, start + duration:]], dim=2), b[:, 2:]], dim=1))
            samples['zero_4'].append(torch.cat([b[:, :, :start],
                                                torch.zeros(bs, dataset.num_channels, dataset.seq_len),
                                                b[:, :, start + duration:]], dim=2))
            if i >= num_samples:
                break
    elif mode in ['noise']:
        dataloader = DataLoader(dataset, batch_size=current_hp.batch_size, shuffle=True, drop_last=True)
        i = 0
        for b in dataloader:
            bs = b.size(0)
            i += bs
            samples['noise_1'].append(b + torch.randn_like(b) * 0.01)
            samples['noise_2'].append(b + torch.randn_like(b) * 0.05)
            samples['noise_3'].append(b + torch.randn_like(b) * 0.2)
            if i >= num_samples:
                break
    elif mode in ['collapse']:
        dataloader = DataLoader(dataset, batch_size=current_hp.batch_size, shuffle=True, drop_last=True)
        i = 0
        for b in dataloader:
            bs = b.size(0)
            i += bs * 10
            samples['collapse'].append(b.repeat(10, 1, 1))
            if i >= num_samples:
                break
    elif mode in ['concat']:
        for i in range(num_samples):
            x = torch.cat([dataset[np.random.randint(len(dataset))], dataset[np.random.randint(len(dataset))]], dim=1)
            start = np.random.randint(dataset.seq_len // 4, dataset.seq_len // 2)
            samples['concat'].append(x[:, start:start + dataset.seq_len].unsqueeze(0))
    elif mode in ['normal', 'permute', 'tiny', 'downsample']:
        dataloader = DataLoader(dataset, batch_size=current_hp.batch_size, shuffle=True, drop_last=True)
        i = 0
        for b in dataloader:
            if mode == 'normal':
                samples['x'].append(b)
            else:
                samples['permute_0'].append(b[:, [1, 0, 2, 3, 4]])
                samples['permute_1'].append(b[:, [0, 1, 2, 4, 3]])
                samples['permute_2'].append(b[:, [0, 1, 3, 2, 4]])
                samples['permute_3'].append(b[:, [4, 2, 1, 3, 0]])
            i += b.size(0)
            if mode == 'tiny':
                if i >= (num_samples // 10):
                    break
            else:
                if i >= num_samples:
                    break
    samples = {k: torch.cat(v, dim=0) for k, v in samples.items() if len(v) > 0}  # full sized x
    if real_ndb is None:
        real_ndb = {}
        x = samples['normal']
        max_seq_len = x.size(2)
        seq_lens = [max_seq_len]
        for i in reversed(range(skip_depth, len(scale_up))):
            seq_lens = [int(seq_lens[0] * scale_down[i] / scale_up[i])] + seq_lens
        for seq_len in seq_lens:
            this_x = resample_signal(x, max_seq_len, seq_len, True)
            this_x = resample_signal(this_x, seq_len, max_seq_len, True)
            real_ndb[seq_len] = NDB(this_x.view(this_x.size(0), -1).numpy(), stage=seq_len)
    return {k: calculate_network_stats(v, net, scale_up, scale_down, skip_depth, current_hp, real_ndb) for k, v in
            samples.items()}, real_ndb


def calculate_network_stats(x, net: Network, scale_up, scale_down, skip_depth, current_hp, real_ndb):
    max_seq_len = x.size(2)
    seq_lens = [max_seq_len]
    for i in reversed(range(skip_depth, len(scale_up))):
        seq_lens = [int(seq_lens[0] * scale_down[i] / scale_up[i])] + seq_lens
    stats = {}
    with torch.no_grad():
        for seq_len in seq_lens:
            this_x = resample_signal(x, max_seq_len, seq_len, True)
            this_x = resample_signal(this_x, seq_len, max_seq_len, True)  # resample to give to the net
            total_network_loss = 0.0
            total_prediction_loss = 0.0
            total_global_discriminator_loss = 0.0
            total_local_discriminator_loss = 0.0
            total_global_accuracy_one = 0.0
            total_global_accuracy_two = 0.0
            total_local_accuracy_one = 0.0
            total_local_accuracy_two = 0.0
            total_pred_acc = {}
            total_count = 0
            all_z = []
            all_c = []
            all_zp = []
            all_cp = []
            all_x = []
            for i in range(x.size(0) // 128):
                all_x.append(this_x[i * 128:(i + 1) * 128])
                prediction_loss, global_discriminator_loss, local_discriminator_loss, c_pooled, global_accuracy, local_accuracy, pred_accuracy = net(
                    cudize(all_x[-1]))
                global_accuracy_one, global_accuracy_two = global_accuracy
                local_accuracy_one, local_accuracy_two = local_accuracy
                network_loss = current_hp.prediction_loss_weight * prediction_loss + current_hp.global_loss_weight * global_discriminator_loss + current_hp.local_loss_weight * local_discriminator_loss
                this_batch_size = all_x[-1].size(0)
                total_count += this_batch_size
                total_network_loss += network_loss.item() * this_batch_size
                total_prediction_loss += prediction_loss.item() * this_batch_size
                total_global_discriminator_loss += global_discriminator_loss.item() * this_batch_size
                total_local_discriminator_loss += local_discriminator_loss.item() * this_batch_size
                total_global_accuracy_one += global_accuracy_one * this_batch_size
                total_global_accuracy_two += global_accuracy_two * this_batch_size
                total_local_accuracy_one += local_accuracy_one * this_batch_size
                total_local_accuracy_two += local_accuracy_two * this_batch_size
                dict_add(total_pred_acc, pred_accuracy, this_batch_size)
                z, c, zp, cp = net.inference_forward(x)
                all_z.append(z.view(-1, z.size(1)).cpu())
                all_c.append(c.view(-1, z.size(1)).cpu())
                all_zp.append(zp.cpu())
                all_cp.append(cp.cpu())
                all_x[-1] = all_x[-1].view(this_batch_size, -1)

            total_global_accuracy_one /= total_count
            total_global_accuracy_two /= total_count
            total_local_accuracy_one /= total_count
            total_local_accuracy_two /= total_count
            divide_dict(total_pred_acc, total_count)

            total_prediction_loss /= total_count
            total_pred_acc = merge_pred_accs(total_pred_acc, net.prediction_loss_network.k,
                                             net.prediction_loss_network.bidirectional)
            total_global_discriminator_loss /= total_count
            total_global_accuracy = (total_global_accuracy_one + total_global_accuracy_two) / 2
            total_local_discriminator_loss /= total_count
            total_local_accuracy = (total_local_accuracy_one + total_local_accuracy_two) / 2
            total_network_loss /= total_count

            ndb, js = real_ndb[seq_len].evaluate(torch.stack(all_x, dim=0).cpu())

            metrics = dict(prediction_loss=total_prediction_loss, prediction_acc=total_pred_acc,
                           global_loss=total_global_discriminator_loss, global_acc=total_global_accuracy,
                           local_loss=total_local_discriminator_loss, local_acc=total_local_accuracy,
                           net_loss=total_network_loss, **calc_mean_cov('z', all_z), **calc_mean_cov('c', all_c),
                           **calc_mean_cov('zp', all_zp), **calc_mean_cov('cp', all_cp), ndb_score=ndb, ndb_js=js)
            stats[seq_len] = metrics
    return stats


def main(num_samples):
    # NOTE, this are model dependent and it's far better to read them from a yml file
    skip_depth = 0

    progression_scale_up = EEGDataset.progression_scale_up
    progression_scale_down = EEGDataset.progression_scale_down
    train_dataset, val_dataset = ThinEEGDataset.from_config(validation_ratio=hp.validation_ratio,
                                                            num_channels=hp.num_channels, stride=hp.ds_stride)
    real_ndb = None
    for current_hp, model_address in zip([hp, prediction_hp, local_hp],
                                         ['default_-7.531810902716695', 'prediction_-2.450593529493725',
                                          'local_-0.5012809535156667']):
        network = Network(train_dataset.num_channels, generate_long_sequence=current_hp.generate_long_sequence,
                          pooling=current_hp.pool_or_stride == 'pool', encoder_dropout=current_hp.encoder_dropout,
                          use_sinc_encoder=current_hp.use_sinc_encoder, use_shared_sinc=current_hp.use_shared_sinc,
                          bidirectional=current_hp.bidirectional,
                          contextualizer_num_layers=current_hp.contextualizer_num_layers,
                          contextualizer_dropout=current_hp.contextualizer_dropout,
                          use_transformer=current_hp.use_transformer, causal_prediction=current_hp.causal_prediction,
                          prediction_k=current_hp.prediction_k, encoder_activation=current_hp.encoder_activation,
                          tiny_encoder=current_hp.tiny_encoder)
        network.load_state_dict(torch.load('./results/cpc_trained/' + model_address + '.pth', map_location='cpu'))
        network = cudize(network)
        for i in range(10):  # for stability checks
            real_stats, real_ndb = calculate_stats(train_dataset, network, progression_scale_up, progression_scale_down,
                                                   skip_depth, num_samples, 'normal', current_hp, real_ndb)
            val_stats, _ = calculate_stats(val_dataset, network, progression_scale_up, progression_scale_down,
                                           skip_depth,
                                           num_samples, 'normal', current_hp, real_ndb)
            permuted_stats, _ = calculate_stats(train_dataset, network, progression_scale_up, progression_scale_down,
                                                skip_depth, num_samples, 'permute', current_hp, real_ndb)
            shifted_stats, _ = calculate_stats(train_dataset, network, progression_scale_up, progression_scale_down,
                                               skip_depth, num_samples, 'shift', current_hp, real_ndb)
            concatenated_stats, _ = calculate_stats(train_dataset, network, progression_scale_up,
                                                    progression_scale_down,
                                                    skip_depth, num_samples, 'concat', current_hp, real_ndb)
            downsampled_stats, _ = calculate_stats(train_dataset, network, progression_scale_up, progression_scale_down,
                                                   skip_depth, num_samples, 'downsample', current_hp, real_ndb)
            tiny_stats, _ = calculate_stats(train_dataset, network, progression_scale_up, progression_scale_down,
                                            skip_depth, num_samples, 'tiny', current_hp, real_ndb)
            zeroed_stats, _ = calculate_stats(train_dataset, network, progression_scale_up, progression_scale_down,
                                              skip_depth, num_samples, 'zero', current_hp, real_ndb)
            noised_stats, _ = calculate_stats(train_dataset, network, progression_scale_up, progression_scale_down,
                                              skip_depth, num_samples, 'noise', current_hp, real_ndb)
            collapsed_stats, _ = calculate_stats(train_dataset, network, progression_scale_up, progression_scale_down,
                                                 skip_depth, num_samples, 'collapse', current_hp, real_ndb)
            # TODO (over time and different truncation threshold)
            # normal_generated_stats, _ = calculate_stats(train_dataset, network, progression_scale_up,
            #                                             progression_scale_down, skip_depth, num_samples, 'generated',
            #                                             current_hp, real_ndb)
            # averaged_generated_stats, _ = calculate_stats(train_dataset, network, progression_scale_up,
            #                                               progression_scale_down, skip_depth, num_samples, 'averaged',
            #                                               current_hp, real_ndb)
            # truncated_generated_stats, _ = calculate_stats(train_dataset, network, progression_scale_up,
            #                                                progression_scale_down, skip_depth, num_samples,
            #                                                'truncation', current_hp, real_ndb)


if __name__ == '__main__':
    main(1024 * 32)
