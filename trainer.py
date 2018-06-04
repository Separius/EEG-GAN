import heapq
from utils import cudize, trainable_params
import torch


class Trainer(object):

    def __init__(self, D, G, D_loss, G_loss, optimizer_d, optimizer_g, dataset, random_latents_generator,
                 grad_clip=None, D_training_repeats=1, tick_kimg_default=2, resume_nimg=0):
        self.D = D
        self.G = G
        self.D_loss = D_loss
        self.G_loss = G_loss
        self.D_training_repeats = D_training_repeats
        self.optimizer_d = optimizer_d
        self.optimizer_g = optimizer_g
        self.dataset = dataset
        self.cur_nimg = resume_nimg
        self.random_latents_generator = random_latents_generator
        self.tick_start_nimg = self.cur_nimg
        self.tick_duration_nimg = int(tick_kimg_default * 1000)
        self.iterations = 0
        self.cur_tick = 0
        self.time = 0
        self.grad_clip = grad_clip
        self.stats = {
            'kimg_stat': {'val': self.cur_nimg / 1000., 'log_epoch_fields': ['{val:8.3f}'], 'log_name': 'kimg'},
            'tick_stat': {'val': self.cur_tick, 'log_epoch_fields': ['{val:5}'], 'log_name': 'tick'}
        }
        self.plugin_queues = {
            'iteration': [],
            'epoch': [],
            's': [],
            'end': []
        }

    def register_plugin(self, plugin):
        plugin.register(self)
        intervals = plugin.trigger_interval
        if not isinstance(intervals, list):
            intervals = [intervals]
        for (duration, unit) in intervals:
            queue = self.plugin_queues[unit]
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, *args):
        args = (time,) + args
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            plugin = queue[0][2]
            getattr(plugin, queue_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item)

    def run(self, total_kimg=1):
        for q in self.plugin_queues.values():
            heapq.heapify(q)
        total_nimg = int(total_kimg * 1000)
        try:
            while self.cur_nimg < total_nimg:
                self.train()
                if self.cur_nimg >= self.tick_start_nimg + self.tick_duration_nimg or self.cur_nimg >= total_nimg:
                    self.cur_tick += 1
                    self.tick_start_nimg = self.cur_nimg
                    self.stats['kimg_stat']['val'] = self.cur_nimg / 1000.
                    self.stats['tick_stat']['val'] = self.cur_tick
                    self.call_plugins('epoch', self.cur_tick)
        except KeyboardInterrupt:
            return
        self.call_plugins('end', 1)

    def _clip(self, model):
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(trainable_params(model), self.grad_clip)

    def train(self):
        fake_latents_in = cudize(self.random_latents_generator())
        for i in range(self.D_training_repeats):
            real_images_expr = cudize(next(self.dataiter))
            self.cur_nimg += real_images_expr.size(0)
            D_loss = self.D_loss(self.D, self.G, real_images_expr, fake_latents_in)
            D_loss.backward()
            self._clip(self.D)
            self.optimizer_d.step()
            fake_latents_in = cudize(self.random_latents_generator())
        G_loss = self.G_loss(self.G, self.D, fake_latents_in)
        G_loss.backward()
        self._clip(self.G)
        self.optimizer_g.step()
        self.iterations += 1
        self.call_plugins('iteration', self.iterations, *(G_loss, D_loss))
