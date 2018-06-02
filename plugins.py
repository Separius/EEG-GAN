import os
import time
from datetime import timedelta
from glob import glob
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.trainer.plugins import LossMonitor, Logger
from torch.utils.trainer.plugins.plugin import Plugin
from utils import generate_samples, cudize
from scipy import misc
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DepthManager(Plugin):

    def __init__(self,
                 create_dataloader_fun,
                 create_rlg,
                 max_depth,
                 minibatch_default=32,
                 minibatch_overrides={6: 32, 7: 32, 8: 32},
                 tick_kimg_default=20,
                 tick_kimg_overrides={3: 16, 4: 8, 5: 8, 6: 4, 7: 4, 8: 4},
                 lod_training_nimg=400 * 1000,
                 lod_transition_nimg=100 * 1000,
                 max_lod=None,  # calculate and put values if you want to compare to original impl lod
                 depth_offset=None):
        super(DepthManager, self).__init__([(1, 'iteration')])
        self.minibatch_default = minibatch_default
        self.minibatch_overrides = minibatch_overrides
        self.tick_kimg_default = tick_kimg_default
        self.tick_kimg_overrides = tick_kimg_overrides
        self.create_dataloader_fun = create_dataloader_fun
        self.create_rlg = create_rlg
        self.lod_training_nimg = lod_training_nimg
        self.lod_transition_nimg = lod_transition_nimg
        self.trainer = None
        self.depth = -1
        self.alpha = -1
        self.max_depth = max_depth
        self.max_lod = max_lod
        self.depth_offset = depth_offset

    def register(self, trainer):
        self.trainer = trainer
        self.trainer.stats['minibatch_size'] = self.minibatch_default
        self.trainer.stats['alpha'] = {'log_name': 'alpha', 'log_epoch_fields': ['{val:.2f}'], 'val': self.alpha}
        if self.max_lod is not None and self.depth_offset is not None:
            self.trainer.stats['lod'] = {'log_name': 'lod', 'log_epoch_fields': ['{val:.2f}'], 'val': self.lod}
        self.iteration()

    @property
    def lod(self):
        if self.max_lod is not None and self.depth_offset is not None:
            return self.max_lod - self.depth_offset - self.depth - self.alpha + 1
        return -1

    def iteration(self, *args):
        cur_nimg = self.trainer.cur_nimg
        full_passes, remaining_nimg = divmod(cur_nimg, self.lod_training_nimg + self.lod_transition_nimg)
        train_passes_rem, remaining_nimg = divmod(remaining_nimg, self.lod_training_nimg)
        depth = min(self.max_depth, full_passes + train_passes_rem)
        alpha = remaining_nimg / self.lod_transition_nimg \
            if train_passes_rem > 0 and full_passes + train_passes_rem == depth else 1.0
        dataset = self.trainer.dataset
        if depth != self.depth:
            self.trainer.D.depth = self.trainer.G.depth = dataset.model_depth = depth
            self.depth = depth
            minibatch_size = self.minibatch_overrides.get(depth, self.minibatch_default)
            self.trainer.dataiter = iter(self.create_dataloader_fun(minibatch_size))
            self.trainer.random_latents_generator = self.create_rlg(minibatch_size)
            # print(self.trainer.random_latents_generator().size())
            tick_duration_kimg = self.tick_kimg_overrides.get(depth, self.tick_kimg_default)
            self.trainer.tick_duration_nimg = tick_duration_kimg * 1000
            self.trainer.stats['minibatch_size'] = minibatch_size
        if alpha != self.alpha:
            self.trainer.D.alpha = self.trainer.G.alpha = dataset.alpha = alpha
            self.alpha = alpha
        self.trainer.stats['depth'] = depth
        self.trainer.stats['alpha']['val'] = alpha
        if self.max_lod is not None and self.depth_offset is not None:
            self.trainer.stats['lod']['val'] = self.lod


class LRScheduler(Plugin):

    def __init__(self,
                 lr_scheduler_d,
                 lr_scheduler_g):
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

    def __init__(self, loss_no, stat_name):
        super(EfficientLossMonitor, self).__init__()
        self.loss_no = loss_no
        self.stat_name = stat_name

    def _get_value(self, iteration, *args):
        val = args[self.loss_no] if self.loss_no < 2 else args[self.loss_no].mean()
        return val.item()


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

    def __init__(self, checkpoints_path, keep_old_checkpoints=False, network_snapshot_ticks=40):
        super().__init__([(network_snapshot_ticks, 'epoch'), (1, 'end')])
        self.checkpoints_path = checkpoints_path
        self.keep_old_checkpoints = keep_old_checkpoints
        self._best_val_loss = float('+inf')

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
                    self.last_pattern.format(name,
                                             '{:06}'.format(self.trainer.cur_nimg // 1000))
                )
            )

    def end(self, *args):
        self.epoch(*args)

    def _clear(self, pattern):
        pattern = os.path.join(self.checkpoints_path, pattern)
        for file_name in glob(pattern):
            os.remove(file_name)


class OutputGenerator(Plugin):

    def __init__(self, sample_fn, checkpoints_dir, seq_len, samples_count=6, output_snapshot_ticks=3, res_len=256):
        super(OutputGenerator, self).__init__([(output_snapshot_ticks, 'epoch'), (1, 'end')])
        self.sample_fn = sample_fn
        self.samples_count = samples_count
        self.res_len = res_len
        self.checkpoints_dir = checkpoints_dir
        self.seq_len = seq_len
        self.max_freq = 80

    def register(self, trainer):
        self.trainer = trainer

    @staticmethod
    def save_plot(seq_len, frequency, epoch, checkpoints_dir, generated):
        num_channels = generated.shape[1]
        t = np.linspace(0, seq_len / frequency, seq_len)
        f = np.fft.rfftfreq(seq_len, d=1. / frequency)
        for index in range(len(generated)):
            fig, (axs) = plt.subplots(num_channels, 2)
            if num_channels == 1:
                axs = axs.reshape(1, -1)
            fig.set_figheight(20)
            fig.set_figwidth(20)
            for ch in range(num_channels):
                data = generated[index, ch, :]
                axs[ch][0].plot(t, data, color=(0.8, 0, 0, 0.5), label='fake')
                axs[ch][1].plot(f, np.abs(np.fft.rfft(data)), color=(0.8, 0, 0, 0.5), label='fake_fft')
                axs[ch][0].set_ylim([-1.1, 1.1])
                axs[ch][0].legend()
                axs[ch][1].legend()
            fig.suptitle('epoch: {}, sample: {}'.format(epoch, index))
            fig.canvas.draw()
            image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            misc.imsave(os.path.join(checkpoints_dir, '{epoch}_{index}.png'.format(epoch=epoch, index=index)), image)
            plt.close(fig)

    def epoch(self, epoch_index):
        gen_input = cudize(Variable(self.sample_fn(self.samples_count)))
        out = generate_samples(self.trainer.G, gen_input)
        frequency = int(self.max_freq * out.shape[2] / self.seq_len)
        res_len = min(self.res_len, out.shape[2])
        self.save_plot(res_len, frequency, epoch_index, self.checkpoints_dir, out[:, :, :res_len])

    def end(self, *args):
        self.epoch(*args)


class TeeLogger(Logger):

    def __init__(self, log_file, *args, **kwargs):
        super(TeeLogger, self).__init__(*args, **kwargs)
        self.log_file = open(log_file, 'a', 1)

    def log(self, msg):
        print(msg, flush=True)
        self.log_file.write(msg + '\n')

    def epoch(self, epoch_idx):
        self._log_all('log_epoch_fields')
