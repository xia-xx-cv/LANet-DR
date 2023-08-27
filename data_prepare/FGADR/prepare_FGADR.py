# coding: utf-8
import os
import pandas as pd
import numpy as np
import cv2
from data_prepare.tools import *

import random
random.seed(2021)

"""
对FGADR_seg数据集做预处理
1、全部resize到1024*1024
保存
"""

ori_dir = "/home/mi/Zhankun_work/data/FGADR/FGADR-Seg-set_Release/Seg-set/"

save_dir = "/home/mi/Zhankun_work/data/FGADR/FGADR-Seg-set_Release/Seg-set-after/"

phaselist = ["train", "validation", "test"]
lesions_list = ["EX", "HE", "MA", "SE"]

def main():
    for phase in phaselist:
        oimg_dir = ori_dir + "Original_Images/"
        label_dir = ori_dir

        f = open(phase + ".txt", "r", encoding="utf8")
        imglist_ = f.readlines()
        f.close()
        imglist = [imgname_i.strip("\n").split()[0].split(".")[0] for imgname_i in imglist_]

        for img_name in imglist:
            print(img_name)
            img = cv2.imread(os.path.join(oimg_dir, img_name + ".png"))[:,:,::-1].astype(np.uint8)
            #  EX   HE  MA  SE
            mask = []
            mask_p_t = os.path.join(label_dir + "HardExudate_Masks/", img_name+".png")
            if os.path.isfile(mask_p_t):
                mask.append(cv2.imread(mask_p_t)[:,:,2])
            else:
                mask.append(np.zeros(shape=img.shape[0:2], dtype=np.uint8))

            mask_p_t = os.path.join(label_dir + "Hemohedge_Masks/", img_name+".png")
            if os.path.isfile(mask_p_t):
                mask.append(cv2.imread(mask_p_t)[:,:,2])
            else:
                mask.append(np.zeros(shape=img.shape[0:2], dtype=np.uint8))

            mask_p_t = os.path.join(label_dir + "Microaneurysms_Masks/", img_name+".png")
            if os.path.isfile(mask_p_t):
                mask.append(cv2.imread(mask_p_t)[:, :, 2])
            else:
                mask.append(np.zeros(shape=img.shape[0:2], dtype=np.uint8))

            mask_p_t = os.path.join(label_dir + "SoftExudate_Masks/", img_name+".png")
            if os.path.isfile(mask_p_t):
                mask.append(cv2.imread(mask_p_t)[:, :, 2])
            else:
                mask.append(np.zeros(shape=img.shape[0:2], dtype=np.uint8))


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


