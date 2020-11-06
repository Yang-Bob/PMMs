
import numpy as np

def measure(y_in, pred_in):
    thresh = .5
    y = y_in>thresh
    pred = pred_in>thresh
    tp = np.logical_and(y,pred).sum()
    tn = np.logical_and(np.logical_not(y), np.logical_not(pred)).sum()
    fp = np.logical_and(np.logical_not(y), pred).sum()
    fn = np.logical_and(y, np.logical_not(pred)).sum()
    return tp, tn, fp, fn

class Evaluation():
    def __init__(self, args):
        if args.dataset == 'coco':
            self.num_classes = 80
        if args.dataset == 'voc':
            self.num_classes = 20
        self.num_folds=4
        self.group_class_num = self.num_classes/4
        self.batch_size = args.batch_size
        self.disp_interval = args.disp_interval
        self.clear_num = 200
        self.group = args.group
        self.group_mean_iou = [0]*4
        self.setup()

    def get_val_id_list(self):
        num = int(self.num_classes / self.num_folds)
        val_set = [self.group + self.num_folds * v for v in range(num)]

        return val_set

    def setup(self):
        self.tp_list = [0] * self.num_classes
        self.total_list = [0] * self.num_classes
        self.iou_list = [0] * self.num_classes

    def update_class_index(self):
        if self.num_classes == 80:
            self.class_indexes = self.get_val_id_list()
        if self.num_classes == 20:
            self.class_indexes = range(self.group * 5, (self.group + 1) * 5)

    def update_evl(self, idx, query_mask, pred, count):
        self.update_class_index()
        if count==self.clear_num:
            self.setup()

        for i in range(self.batch_size):
            id = idx[i].item()
            tp, total = self.test_in_train(query_mask[i],pred[i])

            self.tp_list[id] += tp
            self.total_list[id] += total
        self.iou_list = [self.tp_list[ic] /
                    float(max(self.total_list[ic], 1))
                    for ic in range(self.num_classes)]
        self.group_mean_iou[self.group] = np.mean(np.take(self.iou_list, self.class_indexes))

    def test_in_train(self,query_label, pred):
        pred = pred.data.cpu().numpy().astype(np.int32)
        query_label = query_label.cpu().numpy().astype(np.int32)

        tp, tn, fp, fn = measure(query_label, pred)
        total = tp + fp + fn

        return tp, total

class note_best(object):
    def __init__(self):
        self.init_independent()
        
    def init_independent(self):
        self.best0 = 0
        self.best1 = 0
        self.best2 = 0
        self.best3 = 0
        self.best0_step = 0
        self.best1_step = 0
        self.best2_step = 0
        self.best3_step = 0
        self.best_mean = 0

    def update(self, mIou, restore_step, iou_list, evaluations):
        self.update_independent_fold(restore_step, iou_list, evaluations)

    def update_independent_fold(self, restore_step, iou_list, evaluations):
        g0 = evaluations.group_mean_iou[0]
        g1 = evaluations.group_mean_iou[1]
        g2 = evaluations.group_mean_iou[2]
        g3 = evaluations.group_mean_iou[3]

        if g0 > self.best0:
            self.best0 = g0
            self.best0_step = restore_step
        if g1 > self.best1:
            self.best1 = g1
            self.best1_step = restore_step
        if g2 > self.best2:
            self.best2 = g2
            self.best2_step = restore_step
        if g3 > self.best3:
            self.best3 = g3
            self.best3_step = restore_step
        self.best_mean = (self.best0+self.best1+self.best2+self.best3)/4
