import heapq
from utils import cudize
from network import Generator, Discriminator


class Trainer(object):
    def __init__(self, discriminator: Discriminator, generator: Generator, d_loss, g_loss, dataset,
                 random_latents_generator, resume_nimg, optimizer_g, optimizer_d, d_training_repeats: int = 5,
                 tick_kimg_default: float = 5.0):
        assert d_training_repeats >= 1
        self.d_training_repeats = d_training_repeats
        self.discriminator = discriminator
        self.generator = generator
        self.d_loss = d_loss
        self.g_loss = g_loss
        self.dataset = dataset
        self.cur_nimg = resume_nimg
        self.random_latents_generator = random_latents_generator
        self.tick_start_nimg = self.cur_nimg
        self.tick_duration_nimg = int(tick_kimg_default * 1000)
        self.iterations = 0
        self.cur_tick = 0
        self.time = 0
        self.lr_scheduler_g = None
        self.lr_scheduler_d = None
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.stats = {
            'kimg_stat': {'val': self.cur_nimg / 1000., 'log_epoch_fields': ['{val:8.3f}'], 'log_name': 'kimg'},
            'tick_stat': {'val': self.cur_tick, 'log_epoch_fields': ['{val:5}'], 'log_name': 'tick'}
        }
        self.plugin_queues = {
            'iteration': [],
            'epoch': [],  # this is tick
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

    def train(self):
        if self.lr_scheduler_g is not None:
            self.lr_scheduler_g.step(self.cur_nimg / self.d_training_repeats)
        fake_latents_in = cudize(next(self.random_latents_generator))
        for i in range(self.d_training_repeats):
            if self.lr_scheduler_d is not None:
                self.lr_scheduler_d.step(self.cur_nimg)
            real_images_expr = cudize(next(self.dataiter))
            self.cur_nimg += real_images_expr['x'].size(0)
            d_loss = self.d_loss(self.discriminator, self.generator, real_images_expr, fake_latents_in)
            d_loss.backward()
            self.optimizer_d.step()
            fake_latents_in = cudize(next(self.random_latents_generator))
        g_loss = self.g_loss(self.discriminator, self.generator, real_images_expr, fake_latents_in)
        g_loss.backward()
        self.optimizer_g.step()
        self.iterations += 1
        self.call_plugins('iteration', self.iterations, *(g_loss, d_loss))
