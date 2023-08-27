# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import cv2



def mask_crop(img, mask_list=None):
    """
    裁剪眼底图的四周黑边
    :param img: cv2 Image (RGB)
    :param mask_list: mask image list
    :return:
    """
    img_tmp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_tmp[img_tmp == img_tmp[0, 0]] = 0
    img_tmp[img_tmp < 15] = 0
    img_tmp = cv2.medianBlur(img_tmp, 7)
    # 裁剪
    rowsum = np.sum(img_tmp, axis=1) > 0
    top = rowsum.argmax(axis=0)
    bottom = len(rowsum) - rowsum[::-1].argmax(axis=0) - 1
    colsum = np.sum(img_tmp, axis=0) > 0
    left = colsum.argmax(axis=0)
    right = len(colsum) - colsum[::-1].argmax(axis=0) - 1
    # 切片
    img = img[top:bottom, left:right, :]
    s = max(img.shape[0:2])
    img_f = np.zeros((s, s, 3), np.uint8)
    ax, ay = (s - img.shape[1]) // 2, (s - img.shape[0]) // 2
    img_f[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = img

    if mask_list is None:
        return img_f
    else:
        new_mask_list = []
        for mask_i in mask_list:
            tmp = mask_i[top:bottom, left:right]
            tmp_f = np.zeros((s, s), np.uint8)
            tmp_f[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = tmp
            new_mask_list.append(tmp_f)
        return img_f, new_mask_list


def resize(image, mask_list=None, W=1280, H=1280):
    image = cv2.resize(image, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
    if mask_list is None:
        return image
    else:
        new_mask_list = []
        for mask_i in mask_list:
            new_mask_list.append(cv2.resize(mask_i, dsize=(W, H), interpolation=cv2.INTER_CUBIC))
        return image, new_mask_list