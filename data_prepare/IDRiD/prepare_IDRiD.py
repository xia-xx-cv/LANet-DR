# coding: utf-8
import os
import pandas as pd
import numpy as np
import cv2
from data_prepare.tools import *

import random
random.seed(2021)

"""
对IDRiD_seg数据集做预处理
1、裁剪四周黑色边缘, 得到正方形
2、全部resize到1024*1024
保存
"""

ori_dir = "/home/mi/Zhankun_work/data/IDRID/Segmentation/"

save_dir = "/home/mi/Zhankun_work/data/IDRID/Segmentation_after/"

phaselist = ["train", "validation", "test"]
lesions_list = ["EX", "HE", "MA", "SE"]


def main():
    # 将原训练集拆分出验证集
    old_train_dir = ori_dir + "Original Images/" + "train/"
    old_train_list = os.listdir(old_train_dir)
    index = list(range(len(old_train_list)))
    random.shuffle(index)
    new_train_list = []
    for idx in index[0:40]:
        new_train_list.append(old_train_list[idx])
    new_validation_list = []
    for idx in index[40:]:
        new_validation_list.append(old_train_list[idx])


    for phase in phaselist:
        if phase == "train" or phase == "validation":
            split = "train"
        elif phase == "test":
            split = "test"
        oimg_dir = ori_dir + "Original Images/" + split + "/"
        label_dir = ori_dir + "All Segmentation Groundtruths/" + split + "/"
        if phase == "train":
            imglist_ = new_train_list
        elif phase == "validation":
            imglist_ = new_validation_list
        elif phase == "test":
            imglist_ = os.listdir(oimg_dir)

        imglist_.sort()
        imglist = [imgname_i.split(".")[0] for imgname_i in imglist_]
        for img_name in imglist:
            print(img_name)
            img = cv2.imread(os.path.join(oimg_dir, img_name + ".jpg"))[:,:,::-1].astype(np.uint8)
            #  EX   HE  MA  SE
            mask = []
            mask_p_t = os.path.join(label_dir + "3. Hard Exudates/", img_name+"_EX.tif")
            if os.path.isfile(mask_p_t):
                mask.append(cv2.imread(mask_p_t)[:,:,2])
            else:
                mask.append(np.zeros(shape=img.shape[0:2], dtype=np.uint8))

            mask_p_t = os.path.join(label_dir + "2. Haemorrhages/", img_name+"_HE.tif")
            if os.path.isfile(mask_p_t):
                mask.append(cv2.imread(mask_p_t)[:,:,2])
            else:
                mask.append(np.zeros(shape=img.shape[0:2], dtype=np.uint8))

            mask_p_t = os.path.join(label_dir + "1. Microaneurysms/", img_name+"_MA.tif")
            if os.path.isfile(mask_p_t):
                mask.append(cv2.imread(mask_p_t)[:, :, 2])
            else:
                mask.append(np.zeros(shape=img.shape[0:2], dtype=np.uint8))

            mask_p_t = os.path.join(label_dir + "4. Soft Exudates/", img_name+"_SE.tif")
            if os.path.isfile(mask_p_t):
                mask.append(cv2.imread(mask_p_t)[:, :, 2])
            else:
                mask.append(np.zeros(shape=img.shape[0:2], dtype=np.uint8))

            # 统裁剪黑边，和 resize
            img, mask = mask_crop(img, mask)
            # 缩放
            img, mask = resize(img, mask, 1024, 1024)

            # 保存
            sp_t = save_dir+"OriginalImages/" + phase + "/"
            if not os.path.exists(sp_t):
                os.makedirs(sp_t)
            t = cv2.imwrite(sp_t + img_name+".jpg", img[:,:,::-1])
            print(t)
            sp_t = save_dir+"AllSegmentationGroundtruths/"+phase+"/EX/"
            if not os.path.exists(sp_t):
                os.makedirs(sp_t)
            t = cv2.imwrite(sp_t +img_name+".tif", mask[0])
            print(t)
            sp_t = save_dir+"AllSegmentationGroundtruths/"+phase+"/HE/"
            if not os.path.exists(sp_t):
                os.makedirs(sp_t)
            t = cv2.imwrite(sp_t +img_name+".tif", mask[1])
            print(t)
            sp_t = save_dir+"AllSegmentationGroundtruths/"+phase+"/MA/"
            if not os.path.exists(sp_t):
                os.makedirs(sp_t)
            t = cv2.imwrite(sp_t +img_name+".tif", mask[2])
            print(t)
            sp_t = save_dir+"AllSegmentationGroundtruths/"+phase+"/SE/"
            if not os.path.exists(sp_t):
                os.makedirs(sp_t)
            t = cv2.imwrite(sp_t +img_name+".tif", mask[3])
            print(t)



if __name__ == "__main__":
    main()


