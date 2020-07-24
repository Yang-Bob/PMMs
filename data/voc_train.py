from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import random
import PIL.Image as Image
import numpy as np
from config import settings

#random.seed(1385)

class voc_train():

    """Face Landmarks dataset."""

    def __init__(self, args, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_classes = 20
        self.group = args.group
        self.num_folds = args.num_folds
        #self.binary_map_dir = os.path.join(settings.DATA_DIR, 'VOCdevkit2012/VOC2012/', 'Binary_map_aug/train') #val
        self.data_list_dir = os.path.join('data_list/train')
        self.img_dir = os.path.join(settings.DATA_DIR, 'VOCdevkit2012/VOC2012/', 'JPEGImages/')
        self.mask_dir = os.path.join(settings.DATA_DIR, 'VOCdevkit2012/VOC2012/', 'SegmentationClassAug/')
        #self.binary_mask_dir = os.path.join(settings.DATA_DIR, 'VOCdevkit2012/VOC2012/', 'Binary_map_aug/train/')

        self.train_id_list = self.get_train_id_list()
        self.list_splite = self.get_total_list()
        self.list_splite_len = len(self.list_splite)
        self.list_class = self.get_class_list()

        self.transform = transform
        self.count = 0
        self.random_generator = random.Random()
        self.len = args.max_steps *args.batch_size *2
        #self.random_generator.shuffle(self.list_splite)
        #self.random_generator.seed(1385)

    def get_train_id_list(self):
        num = int(self.num_classes/ self.num_folds)
        val_set = [self.group * num + v for v in range(num)]
        train_set = [x for x in range(self.num_classes) if x not in val_set]

        return train_set

    def get_total_list(self):
        new_exist_class_list = []

        fold_list = [0, 1, 2, 3]
        fold_list.remove(self.group)

        for fold in fold_list:
            f = open(os.path.join(self.data_list_dir, 'split%1d_train.txt' % (fold)))
            while True:
                item = f.readline()
                if item == '':
                    break
                img_name = item[:11]
                cat = int(item[13:15]) -1
                new_exist_class_list.append([img_name, cat])
        print("Total images are : ", len(new_exist_class_list))
        # if need filter
        new_exist_class_list = self.filte_multi_class(new_exist_class_list)
        return new_exist_class_list

    def filte_multi_class(self, exist_class_list):

        new_exist_class_list = []
        for name, class_ in exist_class_list:

            mask_path = self.mask_dir + name + '.png'
            mask = cv2.imread(mask_path)
            labels = np.unique(mask[:,:,0])

            labels = [label - 1 for label in labels if label != 255 and label != 0]
            if set(labels).issubset(self.train_id_list):
                new_exist_class_list.append([name, class_])
        print("Total images after filted are : ", len(new_exist_class_list))
        return new_exist_class_list


    def get_class_list(self):
        list_class = {}
        for i in range(self.num_classes):
            list_class[i] = []
        for name, class_ in self.list_splite:
            list_class[class_].append(name)

        return list_class

    def read_img(self, name):
        path = self.img_dir + name + '.jpg'
        img = Image.open(path)

        return img

    def read_mask(self, name, category):
        path = self.mask_dir + name + '.png'
        mask = cv2.imread(path)

        mask[mask!=category+1] = 0
        mask[mask==category+1] = 1

        return mask[:,:,0].astype(np.float32)
    '''
    def read_binary_mask(self, name, category):
        path = self.binary_mask_dir +str(category+1)+'/'+ name + '.png'
        mask = cv2.imread(path)/255

        return mask[:,:,0].astype(np.float32)
    '''
    def load_frame(self, support_name, query_name, class_):
        support_img = self.read_img(support_name)
        query_img = self.read_img(query_name)
        support_mask = self.read_mask(support_name, class_)
        query_mask = self.read_mask(query_name, class_)

        #support_mask = self.read_binary_mask(support_name, class_)
        #query_mask = self.read_binary_mask(query_name, class_)

        return query_img, query_mask, support_img, support_mask, class_

    def random_choose(self):
        class_ = np.random.choice(self.train_id_list, 1, replace=False)[0]
        cat_list = self.list_class[class_]
        sample_img_ids_1 = np.random.choice(len(cat_list), 2, replace=False)

        query_name = cat_list[sample_img_ids_1[0]]
        support_name = cat_list[sample_img_ids_1[1]]

        return support_name, query_name, class_

    def __len__(self):
        # return len(self.image_list)
        return  self.len

    def __getitem__(self, idx):
        support_name, query_name, class_ = self.random_choose()

        query_img, query_mask, support_img, support_mask, class_ = self.load_frame(support_name, query_name, class_)

        if self.transform is not None:
            query_img, query_mask = self.transform(query_img, query_mask)
            support_img, support_mask = self.transform(support_img, support_mask)

        self.count = self.count + 1

        return query_img, query_mask, support_img, support_mask, class_
