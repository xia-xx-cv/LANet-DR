# coding: utf-8
import os
import pandas as pd
import numpy as np
import cv2
from data_prepare.tools import *

"""
对DDR_seg数据集做预处理
1、裁剪四周黑色边缘, 得到正方形
2、全部resize到1024^1024
保存
"""

ori_dir = "/home/mi/Zhankun_work/data/DDR/lesion_segmentation/"

save_dir = "/home/mi/Zhankun_work/data/DDR/lesion_segmentation_after/"

phaselist = ["train", "validation", "test"]
lesions_list = ["EX", "HE", "MA", "SE"]


def main():
    for phase in phaselist:
        oimg_dir = ori_dir + phase + "/image/"
        label_dir = ori_dir + phase + "/label/"
        imglist_ = os.listdir(oimg_dir)
        imglist_.sort()
        imglist = [imgname_i.split(".")[0] for imgname_i in imglist_]
        for img_name in imglist:
            print(img_name)
            #  EX   HE  MA  SE
            mask = []
            mask.append(cv2.imread(os.path.join(label_dir + "EX/", img_name+".tif"))[:,:,0])
            mask.append(cv2.imread(os.path.join(label_dir + "HE/", img_name+".tif"))[:,:,0])
            mask.append(cv2.imread(os.path.join(label_dir + "MA/", img_name+".tif"))[:,:,0])
            mask.append(cv2.imread(os.path.join(label_dir + "SE/", img_name+".tif"))[:,:,0])
            img = cv2.imread(os.path.join(oimg_dir, img_name + ".jpg"))[:,:,::-1].astype(np.uint8)
            # 统裁剪黑边，和 resize
            img, mask = mask_crop(img, mask)
            # 缩放
            img, mask = resize(img, mask, 1024, 1024)

            # 保存
            sp_t = save_dir+"OriginalImages/" + phase + "/"
            if not os.path.exists(sp_t):
                os.mkdir(sp_t)
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
