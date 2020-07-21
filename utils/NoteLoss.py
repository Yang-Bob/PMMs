
import torch
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.data.append(val)

class Loss_total():

    def __init__(self, args):
        self.total = AverageMeter()
        self.part1 = AverageMeter()
        self.part2 = AverageMeter()
        self.disp_interval = args.disp_interval

    def updateloss(self, loss_val, loss_part1=0, loss_part2=0):
        self.total.update(loss_val.data.item(), 1)
        self.part1.update(loss_part1.data.item(),1) if isinstance(loss_part1, torch.Tensor) else self.part1.update(0,1)
        self.part2.update(loss_part2.data.item(),1) if isinstance(loss_part2, torch.Tensor) else self.part2.update(0,1)

    def logloss(self, log_file):
        count = self.total.count
        loss_val_float = self.total.val
        out_str = '%d, %.4f\n' % (count, loss_val_float)
        log_file.write(out_str)

