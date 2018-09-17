import os
import time
import torch
import matplotlib
import numpy as np
from glob import glob
from scipy import misc
from copy import deepcopy
from datetime import timedelta
from torch.autograd import Variable
from utils import generate_samples, cudize
from torch.utils.trainer.plugins.plugin import Plugin
from torch.utils.trainer.plugins import LossMonitor, Logger

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DepthManager(Plugin):
    def __init__(self,  # everything starts from 0 or 1
                 create_dataloader_fun,
                 create_rlg,
                 max_depth,
                 tick_kimg_default,
                 has_attention,
                 get_optimizer,
                 default_lr,
                 disable_progression=False,
                 depth_offset=0,  # starts form 0
                 attention_transition_kimg=400,
                 minibatch_default=256,
                 # all overrides start from depth_offset+1
                 minibatch_overrides={4: 128, 5: 128, 6: 128, 7: 64, 8: 64, 9: 32, 10: 32, 11: 16, 12: 16},
                 tick_kimg_overrides={4: 4, 5: 4, 6: 4, 7: 3, 8: 3, 9: 2, 10: 2, 11: 1, 12: 1},
                 lod_training_kimg=400,
                 lod_training_kimg_overrides={1: 200, 2: 200, 3: 200, 4: 200},
                 lod_transition_kimg=400,
                 lod_transition_kimg_overrides={1: 200, 2: 200, 3: 200, 4: 200}):
        super(DepthManager, self).__init__([(1, 'iteration')])
        self.minibatch_default = minibatch_default
        self.minibatch_overrides = minibatch_overrides
        self.tick_kimg_default = tick_kimg_default
        self.tick_kimg_overrides = tick_kimg_overrides
        self.create_dataloader_fun = create_dataloader_fun
        self.create_rlg = create_rlg
        self.trainer = None
        self.depth = -1
        self.alpha = -1
        self.get_optimizer = get_optimizer
        self.disable_progression = disable_progression
        self.depth_offset = depth_offset
        self.max_depth = max_depth
        self.default_lr = default_lr
        self.attention_transition_kimg = attention_transition_kimg
        self.alpha_map, (self.start_gamma, self.end_gamma) = self.pre_compute_alpha_map(self.depth_offset, max_depth,
                                                                                        lod_training_kimg,
                                                                                        lod_training_kimg_overrides,
                                                                                        lod_transition_kimg,
                                                                                        lod_transition_kimg_overrides,
                                                                                        has_attention,
                                                                                        attention_transition_kimg)

    def register(self, trainer):
        self.trainer = trainer
        self.trainer.stats['minibatch_size'] = self.minibatch_default
        self.trainer.stats['alpha'] = {'log_name': 'alpha', 'log_epoch_fields': ['{val:.2f}'], 'val': self.alpha}
        if self.start_gamma is not None:
            self.trainer.stats['gamma'] = {'log_name': 'gamma', 'log_epoch_fields': ['{val:.2f}'], 'val': 0}
        self.iteration()

    @staticmethod
    def pre_compute_alpha_map(start_depth, max_depth, lod_training_kimg, lod_training_kimg_overrides,
                              lod_transition_kimg, lod_transition_kimg_overrides, has_attention,
                              attention_transition_kimg):
        start_gamma = None
        end_gamma = None
        points = []
        pointer = 0
        for i in range(start_depth, max_depth):
            pointer += int(lod_training_kimg_overrides.get(i + 1, lod_training_kimg) * 1000)
            points.append(pointer)
            if (i == max_depth - 1) and has_attention:
                start_gamma = pointer
                end_gamma = pointer + int(attention_transition_kimg * 1000)
            pointer += int(lod_transition_kimg_overrides.get(i + 1, lod_transition_kimg) * 1000)
            points.append(pointer)
        return points, (start_gamma, end_gamma)

    def calc_progress(self, cur_nimg=None):
        if cur_nimg is None:
            cur_nimg = self.trainer.cur_nimg
        depth = self.depth_offset
        alpha = 1.0
        for i, point in enumerate(self.alpha_map):
            if cur_nimg == point:
                break
            if cur_nimg > point and i % 2 == 0:
                depth += 1
            if cur_nimg < point and i % 2 == 1:
                alpha = (cur_nimg - self.alpha_map[i - 1]) / (point - self.alpha_map[i - 1])
                break
            if cur_nimg < point:
                break
        depth = min(self.max_depth, depth)
        if self.disable_progression:
            depth = self.max_depth
            alpha = 1.0
        return depth, alpha

    def iteration(self, *args):
        depth, alpha = self.calc_progress()
        dataset = self.trainer.dataset
        if depth != self.depth:
            self.trainer.D.depth = self.trainer.G.depth = dataset.model_depth = depth
            self.depth = depth
            minibatch_size = self.minibatch_overrides.get(depth - self.depth_offset, self.minibatch_default)
            self.trainer.optimizer_g, self.trainer.optimizer_d = self.get_optimizer(
                self.minibatch_default * self.default_lr / minibatch_size)
            self.data_loader = self.create_dataloader_fun(minibatch_size)
            self.trainer.dataiter = iter(self.data_loader)
            self.trainer.random_latents_generator = self.create_rlg(minibatch_size)
            tick_duration_kimg = self.tick_kimg_overrides.get(depth - self.depth_offset, self.tick_kimg_default)
            self.trainer.tick_duration_nimg = int(tick_duration_kimg * 1000)
            self.trainer.stats['minibatch_size'] = minibatch_size
        if alpha != self.alpha:
            self.trainer.D.alpha = self.trainer.G.alpha = dataset.alpha = alpha
            self.alpha = alpha
        self.trainer.stats['depth'] = depth
        self.trainer.stats['alpha']['val'] = alpha
        if self.start_gamma is not None:
            cur_kimg = self.trainer.cur_nimg
            if self.disable_progression:
                gamma = cur_kimg / self.attention_transition_kimg
            else:
                gamma = (cur_kimg - self.start_gamma) / (self.end_gamma - self.start_gamma)
            gamma = min(1, max(0, gamma))
            self.trainer.D.set_gamma(gamma)
            self.trainer.G.set_gamma(gamma)
            self.trainer.stats['gamma']['val'] = gamma


class LRScheduler(Plugin):

    def __init__(self, lr_scheduler_d, lr_scheduler_g):
        super(LRScheduler, self).__init__([(1, 'iteration')])
        self.lrs_d = lr_scheduler_d
        self.lrs_g = lr_scheduler_g

    def register(self, trainer):
        self.trainer = trainer
        self.iteration()

    def iteration(self, *args):
        self.lrs_d.step(self.trainer.cur_nimg)
        self.lrs_g.step(self.trainer.cur_nimg)


class EfficientLossMonitor(LossMonitor):

    def __init__(self, loss_no, stat_name, monitor_threshold, monitor_warmup, monitor_patience):
        super(EfficientLossMonitor, self).__init__()
        self.loss_no = loss_no
        self.stat_name = stat_name
        self.threshold = monitor_threshold
        self.warmup = monitor_warmup
        self.patience = monitor_patience
        self.counter = 0

    def _get_value(self, iteration, *args):
        val = args[self.loss_no].item()
        if val != val:
            raise ValueError('loss value is NaN :((')
        return val

    def epoch(self, idx):
        super(EfficientLossMonitor, self).epoch(idx)
        if idx > self.warmup:
            loss_value = self.trainer.stats[self.stat_name]['epoch_mean']
            if abs(loss_value) > self.threshold:
                self.counter += 1
                if self.counter > self.patience:
                    raise ValueError('loss value exceeded the threshold')
            else:
                self.counter = 0


class AbsoluteTimeMonitor(Plugin):
    stat_name = 'time'

    def __init__(self, base_time=0):
        super(AbsoluteTimeMonitor, self).__init__([(1, 'epoch')])
        self.base_time = base_time
        self.start_time = time.time()
        self.epoch_start = self.start_time
        self.start_nimg = None
        self.epoch_time = 0

    def register(self, trainer):
        self.trainer = trainer
        self.start_nimg = trainer.cur_nimg
        self.trainer.stats['sec'] = {'log_format': ':.1f'}

    def epoch(self, epoch_index):
        cur_time = time.time()
        tick_time = cur_time - self.epoch_start
        self.epoch_start = cur_time
        kimg_time = tick_time / (self.trainer.cur_nimg - self.start_nimg) * 1000
        self.start_nimg = self.trainer.cur_nimg
        self.trainer.stats['time'] = timedelta(seconds=time.time() - self.start_time + self.base_time)
        self.trainer.stats['sec']['tick'] = tick_time
        self.trainer.stats['sec']['kimg'] = kimg_time


class SaverPlugin(Plugin):
    last_pattern = 'network-snapshot-{}-{}.dat'

    def __init__(self, checkpoints_path, keep_old_checkpoints=True, network_snapshot_ticks=50):
        super().__init__([(network_snapshot_ticks, 'epoch'), (1, 'end')])
        self.checkpoints_path = checkpoints_path
        self.keep_old_checkpoints = keep_old_checkpoints

    def register(self, trainer):
        self.trainer = trainer

    def epoch(self, epoch_index):
        if not self.keep_old_checkpoints:
            self._clear(self.last_pattern.format('*', '*'))
        for model, name in [(self.trainer.G, 'generator'), (self.trainer.D, 'discriminator')]:
            torch.save(
                model,
                os.path.join(
                    self.checkpoints_path,
                    self.last_pattern.format(name, '{:06}'.format(self.trainer.cur_nimg // 1000))
                )
            )

    def end(self, *args):
        self.epoch(*args)

    def _clear(self, pattern):
        pattern = os.path.join(self.checkpoints_path, pattern)
        for file_name in glob(pattern):
            os.remove(file_name)


class OutputGenerator(Plugin):

    def __init__(self, sample_fn, checkpoints_dir, seq_len, max_freq, res_len, samples_count=8,
                 output_snapshot_ticks=25):
        super(OutputGenerator, self).__init__([(1, 'epoch'), (1, 'end')])
        self.sample_fn = sample_fn
        self.samples_count = samples_count
        self.res_len = res_len
        self.checkpoints_dir = checkpoints_dir
        self.seq_len = seq_len
        self.max_freq = max_freq
        self.my_g_clone = None
        self.output_snapshot_ticks = output_snapshot_ticks

    @staticmethod
    def flatten_params(model):
        return deepcopy(list(p.data for p in model.parameters()))

    @staticmethod
    def load_params(flattened, model):
        for p, avg_p in zip(model.parameters(), flattened):
            p.data.copy_(avg_p)

    def register(self, trainer):
        self.trainer = trainer
        self.my_g_clone = self.flatten_params(self.trainer.G)

    @staticmethod
    def get_images(seq_len, frequency, epoch, generated):
        num_channels = generated.shape[1]
        t = np.linspace(0, seq_len / frequency, seq_len)
        f = np.fft.rfftfreq(seq_len, d=1. / frequency)
        images = []
        for index in range(len(generated)):
            fig, (axs) = plt.subplots(num_channels, 2)
            if num_channels == 1:
                axs = axs.reshape(1, -1)
            fig.set_figheight(20)
            fig.set_figwidth(20)
            for ch in range(num_channels):
                data = generated[index, ch, :]
                axs[ch][0].plot(t, data, color=(0.8, 0, 0, 0.5), label='time domain')
                axs[ch][1].semilogy(f, np.abs(np.fft.rfft(data)), color=(0.8, 0, 0, 0.5), label='freq domain')
                axs[ch][0].set_ylim([-1.1, 1.1])
                axs[ch][0].legend()
                axs[ch][1].legend()
            fig.suptitle('epoch: {}, sample: {}'.format(epoch, index))
            fig.canvas.draw()
            image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)
            plt.close(fig)
        return images

    def epoch(self, epoch_index):
        for p, avg_p in zip(self.trainer.G.parameters(), self.my_g_clone):
            avg_p.mul_(0.9).add_(0.1, p.data)
        if epoch_index % self.output_snapshot_ticks == 0 or epoch_index == 'end':
            z = self.sample_fn(self.samples_count)
            if not isinstance(z, (list, tuple)):
                z = (z,)
            gen_input = (cudize(Variable(x)) for x in z)
            original_param = self.flatten_params(self.trainer.G)
            self.load_params(self.my_g_clone, self.trainer.G)
            out = generate_samples(self.trainer.G, gen_input)
            self.load_params(original_param, self.trainer.G)
            frequency = self.max_freq * out.shape[2] / self.seq_len
            res_len = min(self.res_len, out.shape[2])
            images = self.get_images(res_len, frequency, epoch_index, out[:, :, :res_len])
            for i, image in enumerate(images):
                misc.imsave(os.path.join(self.checkpoints_dir, '{}_{}.png'.format(epoch_index, i)), image)

    def end(self, *args):
        self.epoch('end')


class TeeLogger(Logger):

    def __init__(self, log_file, exp_name, *args, **kwargs):
        super(TeeLogger, self).__init__(*args, **kwargs)
        self.log_file = open(log_file, 'a', 1)
        self.exp_name = exp_name

    def log(self, msg):
        print(self.exp_name, msg, flush=True)
        self.log_file.write(msg + '\n')

    def epoch(self, epoch_idx):
        self._log_all('log_epoch_fields')
