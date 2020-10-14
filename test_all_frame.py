
import os
import cv2
import torch
import json
import numpy as np
import argparse
from torch.autograd import Variable
from data.LoadDataSeg import val_loader
from utils import NoteEvaluation
from networks import *
from utils.Restore import restore

from config import settings
from test import val as VAL
#from test_5shot import val as VAL

from utils.Restore import Save_Evaluations
from utils.Visualize import print_best

DATASET = 'voc'
SNAPSHOT_DIR =settings.SNAPSHOT_DIR
#SNAPSHOT_DIR = SNAPSHOT_DIR+'/coco'
if DATASET =='coco':
    SNAPSHOT_DIR = SNAPSHOT_DIR+'/coco'

START = 5000
END = 205000

GPU_ID = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--arch", type=str,default='FRPMMs')
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)

    parser.add_argument("--group", type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument('--interval', type=int, default=5000)
    parser.add_argument('--start', type=int, default=START)
    parser.add_argument('--end', type=int, default=END)
    parser.add_argument('--restore_step', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mode', type=str, default='val')
    parser.add_argument('--dataset', type=str, default=DATASET)

    return parser.parse_args()

def get_model(args):

    model = eval(args.arch).OneModel(args)

    model = model.cuda()

    return model


if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    Best_Note = NoteEvaluation.note_best()
    File_Evaluations = Save_Evaluations(args)
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    for i in range(args.start, args.end, args.interval):
        print("---------------------------------EVALUATE STEP %d---------------------------------" % (i))
        args.restore_step = i

        mIoU, iou, evaluations = VAL(args)

        Best_Note.update(mIoU, args.restore_step, iou, evaluations)
        File_Evaluations.update_date(args.restore_step, mIoU, evaluations)
        print("-------------")
        print("best_BMVC_IOU ", Best_Note.best_mean)
        print("best_group0_iou", Best_Note.best0)
        print("best_group1_iou", Best_Note.best1)
        print("best_group2_iou", Best_Note.best2)
        print("best_group3_iou", Best_Note.best3)

    print_best(Best_Note)
    File_Evaluations.update_best(Best_Note)
    File_Evaluations.update_best_eachgroup(Best_Note)
    File_Evaluations.save_file()


