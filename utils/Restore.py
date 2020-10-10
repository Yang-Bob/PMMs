
import os
import shutil
import torch
import pandas as pd
import numpy as np

def restore(args, model):

    group = args.group
    savedir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(group, args.num_folds))
    if args.restore_step=='best':
        filename='%s.pth.tar'%(args.restore_step)
    else:
        filename='step_%d.pth.tar'%(args.restore_step)
    snapshot = os.path.join(savedir, filename)
    assert os.path.exists(snapshot), "Snapshot file %s does not exist."%(snapshot)

    checkpoint = torch.load(snapshot)
    model.load_state_dict(checkpoint['state_dict'])

    print('Loaded weights from %s'%(snapshot))

def get_model_para_number(model):
    total_number = 0
    for para in model.parameters():
        total_number += torch.numel(para)

    return total_number

def get_save_dir(args):
    snapshot_dir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(args.group, args.num_folds))
    return snapshot_dir

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savedir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(args.group, args.num_folds))
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    savepath = os.path.join(savedir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def save_model(args, count, model, optimizer):
    if count % args.save_interval == 0 and count > 0:
        save_checkpoint(args,
                        {
                            'state_dict': model.state_dict()
                        }, is_best=False,
                        filename='step_%d.pth.tar'
                                 % (count))

class Save_Evaluations():
    def __init__(self, args):
        self.savedir = 'evaluation/' + args.arch + '_' + args.dataset+'_'+'val.csv'
        self.Note_Iou = []
        self.col = []
        self.ind = ['Group0','Group1','Group2','Group3','Mean']
        self.dataset = args.dataset
        self.update_class_index()
        for i in range(5):
            self.Note_Iou.append([])
    def get_val_id_list(self, group):
        num_classes = 80
        num_folds = 4
        num = int(num_classes / num_folds)
        # val_set = [self.group * num + v for v in range(num)]
        val_set = [group + num_folds * v for v in range(num)]
        return val_set
    def update_class_index(self):
        self.class_indexes = []
        for group in range(4):
            if self.dataset == 'coco':
                self.class_indexes.append(self.get_val_id_list(group))
            if self.dataset == 'voc':
                self.class_indexes = range(group * 5, (group + 1) * 5)
    def save_file(self):
        test = pd.DataFrame(data=self.Note_Iou, index=self.ind, columns=self.col)
        test.to_csv(self.savedir)
    def update_date(self, restore_step, mIoU, evaluations):
        self.col.append(restore_step)

        self.Note_Iou[0].append(evaluations.group_mean_iou[0])
        self.Note_Iou[1].append(evaluations.group_mean_iou[1])
        self.Note_Iou[2].append(evaluations.group_mean_iou[2])
        self.Note_Iou[3].append(evaluations.group_mean_iou[3])
        self.Note_Iou[4].append(mIoU)

    def update_best(self, Best_Note):
        self.col.append(Best_Note.restore_step)
        self.Note_Iou[0].append(Best_Note.group0_iou)
        self.Note_Iou[1].append(Best_Note.group1_iou)
        self.Note_Iou[2].append(Best_Note.group2_iou)
        self.Note_Iou[3].append(Best_Note.group3_iou)
        self.Note_Iou[4].append(Best_Note.BMVC_IOU)
    def update_best_eachgroup(self, Best_Note):
        self.col.append('Best')
        self.Note_Iou[0].append(Best_Note.best0)
        self.Note_Iou[1].append(Best_Note.best1)
        self.Note_Iou[2].append(Best_Note.best2)
        self.Note_Iou[3].append(Best_Note.best3)
        self.Note_Iou[4].append(Best_Note.best_mean)

        self.col.append('Best_Step')
        self.Note_Iou[0].append(Best_Note.best0_step)
        self.Note_Iou[1].append(Best_Note.best1_step)
        self.Note_Iou[2].append(Best_Note.best2_step)
        self.Note_Iou[3].append(Best_Note.best3_step)
        self.Note_Iou[4].append(Best_Note.best_mean)
