
import os
import cv2
import torch
import json
import numpy as np
import argparse
import time
import torch.nn.functional as F
from data.LoadDataSeg import val_loader
from utils import NoteEvaluation
from networks import *
from utils.Restore import restore

from config import settings


DATASET = 'voc'
SNAPSHOT_DIR =settings.SNAPSHOT_DIR
if DATASET =='coco':
    SNAPSHOT_DIR = SNAPSHOT_DIR+'/coco'


GPU_ID = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--arch", type=str,default='FRPMMs')
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)

    parser.add_argument("--group", type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument('--restore_step', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mode', type=str, default='val')
    parser.add_argument('--dataset', type=str, default=DATASET)

    return parser.parse_args()

def get_model(args):

    model = eval(args.arch).OneModel(args)

    model = model.cuda()

    return model

def val(args):
    model = get_model(args)
    model.eval()

    evaluations = NoteEvaluation.Evaluation(args)

    for group in range(4):

        print("-------------GROUP %d-------------" % (group))

        args.group = group
        evaluations.group =args.group
        val_dataloader = val_loader(args)
        restore(args, model)
        it = 0

        for data in val_dataloader:
            begin_time = time.time()
            it = it+1
            query_img, query_mask, support_img, support_mask, idx, size = data

            query_img, query_mask, support_img, support_mask, idx \
                = query_img.cuda(), query_mask.cuda(), support_img.cuda(), support_mask.cuda(), idx.cuda()

            with torch.no_grad():

                logits = model(query_img, support_img, support_mask)

                query_img = F.upsample(query_img, size=(size[0], size[1]), mode='bilinear')
                query_mask = F.upsample(query_mask, size=(size[0], size[1]), mode='nearest')

                values, pred = model.get_pred(logits, query_img)
                evaluations.update_evl(idx, query_mask, pred, 0)
            end_time = time.time()
            ImgPerSec = 1/(end_time-begin_time)

            print("It has tested %d, %.2f images/s" %(it,ImgPerSec), end="\r")
        print("Group %d: %.4f " %(args.group, evaluations.group_mean_iou[args.group]))

    iou = evaluations.iou_list
    print('IOU:', iou)
    mIoU = np.mean(iou)
    print('mIoU: ', mIoU)
    print("group0_iou", evaluations.group_mean_iou[0])
    print("group1_iou", evaluations.group_mean_iou[1])
    print("group2_iou", evaluations.group_mean_iou[2])
    print("group3_iou", evaluations.group_mean_iou[3])
    print(evaluations.group_mean_iou)
    #print(evaluations.iou_list)

    return mIoU, iou, evaluations



if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    val(args)
