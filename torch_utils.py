from collections import defaultdict


class Plugin(object):

    def __init__(self, interval=None):
        if interval is None:
            interval = []
        self.trigger_interval = interval

    def register(self, trainer):
        raise NotImplementedError


class Monitor(Plugin):

    def __init__(self, running_average=True, epoch_average=True, smoothing=0.7,
                 precision=None, number_format=None, unit=''):
        if precision is None:
            precision = 4
        if number_format is None:
            number_format = '.{}f'.format(precision)
        number_format = ':' + number_format
        super(Monitor, self).__init__([(1, 'iteration'), (1, 'tick')])

        self.smoothing = smoothing
        self.with_running_average = running_average
        self.with_epoch_average = epoch_average

        self.log_format = number_format
        self.log_unit = unit
        self.log_epoch_fields = None
        self.log_iter_fields = ['{last' + number_format + '}' + unit]
        if self.with_running_average:
            self.log_iter_fields += [' ({running_avg' + number_format + '}' + unit + ')']
        if self.with_epoch_average:
            self.log_epoch_fields = ['{epoch_mean' + number_format + '}' + unit]

    def register(self, trainer):
        self.trainer = trainer
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['log_format'] = self.log_format
        stats['log_unit'] = self.log_unit
        stats['log_iter_fields'] = self.log_iter_fields
        if self.with_epoch_average:
            stats['log_epoch_fields'] = self.log_epoch_fields
        if self.with_epoch_average:
            stats['epoch_stats'] = (0, 0)

    def iteration(self, *args):
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['last'] = self._get_value(*args)

        if self.with_epoch_average:
            stats['epoch_stats'] = tuple(sum(t) for t in
                                         zip(stats['epoch_stats'], (stats['last'], 1)))

        if self.with_running_average:
            previous_avg = stats.get('running_avg', 0)
            stats['running_avg'] = previous_avg * self.smoothing + \
                                   stats['last'] * (1 - self.smoothing)

    def epoch(self, idx):
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        if self.with_epoch_average:
            epoch_stats = stats['epoch_stats']
            stats['epoch_mean'] = epoch_stats[0] / epoch_stats[1]
            stats['epoch_stats'] = (0, 0)


class LossMonitor(Monitor):
    stat_name = 'loss'

    def _get_value(self, iteration, input, target, output, loss):
        return loss.item()


class Logger(Plugin):
    alignment = 4
    separator = '#' * 80

    def __init__(self, fields, interval=None):
        if interval is None:
            interval = [(1, 'iteration'), (1, 'tick')]
        super(Logger, self).__init__(interval)
        self.field_widths = defaultdict(lambda: defaultdict(int))
        self.fields = list(map(lambda f: f.split('.'), fields))

    def _join_results(self, results):
        joined_out = map(lambda i: (i[0], ' '.join(i[1])), results)
        joined_fields = map(lambda i: '{}: {}'.format(i[0], i[1]), joined_out)
        return '\t'.join(joined_fields)

    def log(self, msg):
        print(msg)

    def register(self, trainer):
        self.trainer = trainer

    def gather_stats(self):
        result = {}
        return result

    def _align_output(self, field_idx, output):
        for output_idx, o in enumerate(output):
            if len(o) < self.field_widths[field_idx][output_idx]:
                num_spaces = self.field_widths[field_idx][output_idx] - len(o)
                output[output_idx] += ' ' * num_spaces
            else:
                self.field_widths[field_idx][output_idx] = len(o)

    def _gather_outputs(self, field, log_fields, stat_parent, stat, require_dict=False):
        output = []
        name = ''
        if isinstance(stat, dict):
            log_fields = stat.get(log_fields, [])
            name = stat.get('log_name', '.'.join(field))
            for f in log_fields:
                output.append(f.format(**stat))
        elif not require_dict:
            name = '.'.join(field)
            number_format = stat_parent.get('log_format', '')
            unit = stat_parent.get('log_unit', '')
            fmt = '{' + number_format + '}' + unit
            output.append(fmt.format(stat))
        return name, output

    def _log_all(self, log_fields, prefix=None, suffix=None, require_dict=False):
        results = []
        for field_idx, field in enumerate(self.fields):
            parent, stat = None, self.trainer.stats
            for f in field:
                parent, stat = stat, stat[f]
            name, output = self._gather_outputs(field, log_fields,
                                                parent, stat, require_dict)
            if not output:
                continue
            self._align_output(field_idx, output)
            results.append((name, output))
        if not results:
            return
        output = self._join_results(results)
        if prefix is not None:
            self.log(prefix)
        self.log(output)
        if suffix is not None:
            self.log(suffix)

    def iteration(self, *args):
        self._log_all('log_iter_fields')

    def epoch(self, epoch_idx):
        self._log_all('log_epoch_fields',
                      prefix=self.separator + '\nEpoch summary:',
                      suffix=self.separator,
                      require_dict=True)