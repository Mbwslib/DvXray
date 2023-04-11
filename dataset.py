from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from utils import cvtColor, preprocess_input


class data_loader(Dataset):

    def __init__(self, annotation_lines, input_shape):
        super(data_loader, self).__init__()

        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)

        self.input_shape = input_shape

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        index = index % self.length

        line = self.annotation_lines[index].split()

        image_ol = cvtColor(Image.open(line[0]).resize(self.input_shape, Image.BILINEAR))
        image_sd = cvtColor(Image.open(line[1]).resize(self.input_shape, Image.BILINEAR))

        image_ol = self.get_random_data(image_ol)
        image_sd = self.get_random_data(image_sd)

        image_ol = np.transpose(preprocess_input(image_ol), (2, 0, 1))
        image_sd = np.transpose(preprocess_input(image_sd), (2, 0, 1))

        gt = np.array(list(map(int, line[2].split(','))))

        return image_ol, image_sd, gt


    def get_random_data(self, image, hue=.1, sat=0.7, val=0.4):

        image_data = np.array(image, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data

def DvX_dataset_collate(batch):

    img_ols, img_sds, gt_s = [], [], []

    for img_ol, img_sd, gt in batch:
        img_ols.append(img_ol)
        img_sds.append(img_sd)
        gt_s.append(gt)

    img_ols = torch.from_numpy(np.array(img_ols)).type(torch.FloatTensor)
    img_sds = torch.from_numpy(np.array(img_sds)).type(torch.FloatTensor)
    gt_s = torch.from_numpy(np.array(gt_s)).type(torch.FloatTensor)

    return img_ols, img_sds, gt_s