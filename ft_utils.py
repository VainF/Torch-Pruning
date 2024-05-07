import operator

from loguru import logger

__all__ = ["AverageMeter", "ProgressMonitor", "PerformanceScoreboard"]


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, fmt='%.6f'):
        self.fmt = fmt
        self.max = self.min = self.val = self.avg = self.sum = self.count = 0

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = self.val if self.val > val else val
        self.min = val if self.val > val else self.val

    def __str__(self):
        s = self.fmt % self.avg
        return s


class Monitor:
    """This is an abstract interface for data loggers

    Train monitors log the progress of the training process to some backend.
    This backend can be a file, a web service, or some other means to collect and/or
    display the training
    """

    def __init__(self):
        pass

    def update(self, epoch, step_idx, step_num, prefix, meter_dict):
        raise NotImplementedError


class ProgressMonitor(Monitor):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def update(self, epoch, step_idx, step_num, prefix, meter_dict):
        msg = prefix
        if epoch > -1:
            msg += ' [%d][%5d/%5d]   ' % (epoch, step_idx, int(step_num))
        else:
            msg += ' [%5d/%5d]   ' % (step_idx, int(step_num))
        for k, v in meter_dict.items():
            msg += k + ' '
            if isinstance(v, AverageMeter):
                msg += str(v)
            else:
                msg += '%.6f' % v
            msg += '   '
        self.logger.info(msg)


class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch
