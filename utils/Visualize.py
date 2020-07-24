import math
import cv2
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage.morphology as skm
from config import settings

class visualize_loss_evl_train():
    def __init__(self, args):
        self.disp_interval = args.disp_interval

    def visualize(self, args, count, Loss, Evaluation, begin_time):

        if count % self.disp_interval == 0:
            loss_mean = np.mean(Loss.total.data[-100:])
            loss1 = np.mean(Loss.part1.data[-100:])
            loss2 = np.mean(Loss.part2.data[-100:])
            mean_IOU = np.mean(np.take(Evaluation.iou_list, np.where(np.array(Evaluation.iou_list) > 0)))

            step_time = time.time() - begin_time
            remaining_time = step_time*(args.max_steps-count)/self.disp_interval/3600
            print(args.arch, 'Group:%d \t Step:%d \t Loss:%.3f \t '
                  'Part1: %.3f \t Part2: %.3f \t  mean_IOU: %.4f \t'
                  'Step time: %.4f s \t Remaining time: %.4f h' % (args.group, count, loss_mean,
                                                  loss1.cpu().data.numpy() if isinstance(loss1,
                                                                                         torch.Tensor) else loss1,
                                                  loss2.cpu().data.numpy() if isinstance(loss2,
                                                                                         torch.Tensor) else loss2,
                                                                     mean_IOU, step_time, remaining_time))
def print_best(Best_Note):
    print("---------------------------------FINAL BEST RESULT ---------------------------------")
    print("best_ BMVC_IOU ", Best_Note.best_mean)
    print("group0_iou", Best_Note.best0)
    print("group1_iou", Best_Note.best1)
    print("group2_iou", Best_Note.best2)
    print("group3_iou", Best_Note.best3)
    print("group0_step ", Best_Note.best0_step)
    print("group1_step ", Best_Note.best1_step)
    print("group2_step ", Best_Note.best2_step)
    print("group3_step ", Best_Note.best3_step)
