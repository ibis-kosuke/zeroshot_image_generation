from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def prepare_data(data):
        # sort data by the length in a decreasing order
    imgs, atts, image_atts, class_ids, keys = data
    if cfg.CUDA:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].cuda()
        atts = atts.to(torch.float32)
        atts = atts.cuda()
        image_atts = image_atts.to(torch.float32).cuda()
    ##atts: batch x 312
    return imgs, atts, image_atts, class_ids, keys


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, args,
                 base_size=64,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform

        self.split = args.split
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None

        self.filenames, self.att_data = self.load_att_data(data_dir, self.split)
        self.filename_to_att = self.load_image_att(data_dir)

        self.class_id = self.load_class_id(data_dir, self.split, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_image_att(self, data_dir):
        filepath = os.path.join(data_dir, "filename_to_att.pickle")
        with open(filepath, "rb") as f:
            filename_to_att = pickle.load(f)
        return filename_to_att

    def load_att_data(self, data_dir, split): 
        filenames = self.load_filenames(data_dir, split)
        att_dir = os.path.join(data_dir,"CUB_200_2011/attributes")
        att_np = np.zeros((312, 200)) #for CUB
        with open(att_dir+"/class_attribute_labels_continuous.txt", "r") as f:
            for ind, line in enumerate(f.readlines()):
                line = line.strip("\n")
                line = list(map(float, line.split()))
                att_np[:,ind] = line
        return filenames, att_np
   
    def load_class_id(self, data_dir, split, total_num):
        if split=="train":
            filepath = '%s/%s/class_info_seen.pickle' % (data_dir, split)
        else:
            if split=="test_seen":
                filepath = '%s/%s/class_info_seen.pickle' % (data_dir, split)
            else: # split == "test_unseen"
                filepath = '%s/%s/class_info_unseen.pickle' %(data_dir, split)
        
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                class_id = pickle.load(f, encoding="latin1")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        if split=="train":
            filepath = '%s/%s/filenames_seen.pickle' % (data_dir, split)
        else:
            if split=="test_seen":
                filepath = '%s/%s/filenames_seen.pickle' % (data_dir, split)
            else: # split == "test_unseen"
                filepath = '%s/%s/filenames_unseen.pickle' %(data_dir, split)
        
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f, encoding="latin1")
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        # random select a sentence
        class_att = self.att_data[:,cls_id-1] ##312
        image_att = self.filename_to_att[key]

        return imgs, class_att, image_att, cls_id, key
            
    def __len__(self):
        return len(self.filenames)
