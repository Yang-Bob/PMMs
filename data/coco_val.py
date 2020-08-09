from __future__ import print_function
from __future__ import absolute_import

# random.seed(1234)
# from .transforms import functional
import os
import cv2
import random
import torch
import PIL.Image as Image
import numpy as np
from config import settings
from pycocotools.coco import COCO
#random.seed(1385)

'''
1. change seed 
2. filt small object
'''

class coco_val():

    """Face Landmarks dataset."""

    def __init__(self, args, transform=None, k_shot=1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_classes = 80
        self.group = args.group
        self.num_folds = args.num_folds

        self.dataDir='/home/ubuntu/Dataset/MSCOCO2017'
        self.dataType='val2017' # train2017 val2017
        self.annFile='{}/annotations/instances_{}.json'.format(self.dataDir, self.dataType)
        self.coco=COCO(self.annFile)

        #self.nms = self.get_nms()
        self.val_id_list = self.get_val_id_list()
        self.coco_all_id = self.coco.getCatIds()
        self.val_coco_id_list = self.get_val_coco_id_list()
        self.list_splite = self.get_total_list()
        self.list_splite_len = len(self.list_splite)
        self.list_class = self.get_class_list()

        self.transform = transform
        self.count = 0
        self.random_generator = random.Random()
        self.k_shot = k_shot
        #self.len = args.max_steps *args.batch_size *2
        #self.random_generator.shuffle(self.list_splite)


        #self.random_generator.seed(1385)
        #self.split = args.split

    def get_nms(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        return nms

    def get_val_coco_id_list(self):
        val_coco_id_list = []
        for i in self.val_id_list:
            cls = self.coco_all_id[i]
            val_coco_id_list.append(cls)

        return val_coco_id_list

    def get_val_id_list(self):
        num = int(self.num_classes/ self.num_folds)
        #val_set = [self.group * num + v for v in range(num)]
        val_set = [self.group + self.num_folds * v for v in range(num)]

        #train_set = [x for x in range(self.num_classes) if x not in val_set]

        return val_set

    def get_category(self, annotations):
        category_id_list = []
        for ann in annotations:
            category_id_list.append(ann['category_id'])
        category = np.array(category_id_list)
        category = np.unique(category)
        return category

    def get_total_list(self):
        new_exist_class_list = []
        for coco_id in self.val_coco_id_list:
            imgIds = self.coco.getImgIds(catIds=coco_id);
            for i in range(len(imgIds)):
                img = self.coco.loadImgs(imgIds[i])[0]
                annIds = self.coco.getAnnIds(imgIds=img['id'], iscrowd=None)  # catIds=catIds,
                anns = self.coco.loadAnns(annIds)
                label = self.get_category(anns)
                ##filt the img not in train set
                #if set(label.tolist()).issubset(self.train_coco_id_list):
                new_exist_class_list.append(img['id'])

        new_exist_class_list_unique = list(set(new_exist_class_list))
        print("Total images are : ", len(new_exist_class_list_unique))
        return new_exist_class_list_unique


    def get_class_list(self):
        list_class = {}
        for i in range(self.num_classes):
            list_class[i] = []
        for name in self.list_splite:
            annIds = self.coco.getAnnIds(imgIds=name, iscrowd=None)  # catIds=catIds,
            anns = self.coco.loadAnns(annIds)
            labels = self.get_category(anns)
            for class_ in labels:
                # decode coco label to our label
                class_us = self.coco_all_id.index(class_)
                list_class[class_us].append(name)

        return list_class

    def read_img(self, name):
        img = self.coco.loadImgs(name)[0]
        path = self.dataDir + '/images/' + img['file_name']
        img = Image.open(path)

        return img

    def read_mask(self, name, category):

        img = self.coco.loadImgs(name)[0]

        annIds = self.coco.getAnnIds(imgIds=name, catIds=category, iscrowd=None)  # catIds=catIds,
        anns = self.coco.loadAnns(annIds)

        mask = self.get_mask(img, anns, category)

        return mask.astype(np.float32)

    def polygons_to_mask2(self, img_shape, polygons):

        mask = np.zeros(img_shape, dtype=np.uint8)
        polygons = np.asarray([polygons], np.int32)  # 这里必须是int32，其他类型使用fillPoly会报错
        # cv2.fillPoly(mask, polygons, 1) # 非int32 会报错
        cv2.fillConvexPoly(mask, polygons, 1)  # 非int32 会报错
        return mask

    def get_mask(self, img, annotations, category_id):
        len_ann = len(annotations)

        half_mask = []
        final_mask = []

        for ann in annotations:
            if ann['category_id'] == category_id:
                if ann['iscrowd'] == 1:
                    continue
                seg1 = ann['segmentation']
                seg = seg1[0]
                for j in range(0, len(seg), 2):
                    x = seg[j]
                    y = seg[j + 1]
                    mas = [x, y]
                    half_mask.append(mas)
                final_mask.append(half_mask)
                half_mask = []

        mask0 = self.polygons_to_mask2([img['height'],img['width']], final_mask[0])
        for i in range(1, len(final_mask)):
            maskany = self.polygons_to_mask2([img['height'],img['width']], final_mask[i])
            mask0 += maskany

        mask0[mask0 > 1] = 1

        return mask0

    def load_frame(self, support_name, query_name, class_):
        support_img = self.read_img(support_name)
        query_img = self.read_img(query_name)
        class_coco = self.coco_all_id[class_]
        support_mask = self.read_mask(support_name, class_coco)
        query_mask = self.read_mask(query_name, class_coco)

        #support_mask = self.read_binary_mask(support_name, class_)
        #query_mask = self.read_binary_mask(query_name, class_)



        return query_img.convert('RGB'), query_mask, support_img.convert('RGB'), support_mask, class_

    def load_frame_k_shot(self, support_name_list, query_name, class_):
        class_coco = self.coco_all_id[class_]

        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name, class_coco)

        support_img_list = []
        support_mask_list = []

        for support_name in support_name_list:
            support_img = self.read_img(support_name)
            support_mask = self.read_mask(support_name, class_coco)
            support_img_list.append(support_img.convert('RGB'))
            support_mask_list.append(support_mask)

        return query_img.convert('RGB'), query_mask, support_img_list, support_mask_list

    def random_choose(self):
        class_ = np.random.choice(self.val_id_list, 1, replace=False)[0]
        cat_list = self.list_class[class_]
        sample_img_ids_1 = np.random.choice(len(cat_list), 2, replace=False)

        query_name = cat_list[sample_img_ids_1[0]]
        support_name = cat_list[sample_img_ids_1[1]]

        return support_name, query_name, class_

    def random_choose_k(self):
        class_ = np.random.choice(self.val_id_list, 1, replace=False)[0]
        cat_list = self.list_class[class_]
        sample_img_ids_1 = np.random.choice(len(cat_list), self.k_shot+1, replace=False)

        query_name = cat_list[sample_img_ids_1[0]]

        support_name_list = []
        for i in range(self.k_shot):
            support_name = cat_list[sample_img_ids_1[i+1]]
            support_name_list.append(support_name)

        return support_name_list, query_name, class_

    def get_1_shot(self, idx):
        support_name, query_name, class_ = self.random_choose()

        img = self.coco.loadImgs(query_name)[0]
        size = [img['height'], img['width']]

        query_img, query_mask, support_img, support_mask, class_ = self.load_frame(support_name, query_name,
                                                                                   class_)  # class_ is cooc laebl

        if self.transform is not None:
            query_img, query_mask = self.transform(query_img, query_mask)
            support_img, support_mask = self.transform(support_img, support_mask)

        self.count = self.count + 1

        return query_img, query_mask, support_img, support_mask, class_, size

    def get_k_shot(self, idx):
        support_name_list, query_name, class_ = self.random_choose_k()

        img = self.coco.loadImgs(query_name)[0]
        size = [img['height'], img['width']]

        query_img, query_mask, support_img_list, support_mask_list = self.load_frame_k_shot(support_name_list, query_name, class_)  # class_ is cooc laebl

        if self.transform is not None:
            query_img, query_mask = self.transform(query_img, query_mask)
            for i in range(len(support_mask_list)):
                support_temp_img = support_img_list[i]
                support_temp_mask = support_mask_list[i]
                support_temp_img, support_temp_mask = self.transform(support_temp_img, support_temp_mask)
                support_temp_img = support_temp_img.unsqueeze(dim=0)
                support_temp_mask = support_temp_mask.unsqueeze(dim=0)

                #print(support_temp_img.shape)
                if i ==0:
                    support_img = support_temp_img
                    support_mask = support_temp_mask
                else:
                    support_img = torch.cat([support_img, support_temp_img], dim=0)
                    support_mask = torch.cat([support_mask, support_temp_mask], dim=0)

        self.count = self.count + 1

        return query_img, query_mask, support_img, support_mask, class_, size

    def __len__(self):
        # return len(self.image_list)
        return  1000


    def __getitem__(self, idx):


        if self.k_shot==1:
            query_img, query_mask, support_img, support_mask, class_, size  = self.get_1_shot(idx)# , size
        else:

            query_img, query_mask, support_img, support_mask, class_, size = self.get_k_shot(idx)  # , size

        return query_img, query_mask, support_img, support_mask, class_, size


