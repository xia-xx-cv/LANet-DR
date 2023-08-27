# coding: utf-8
import os
import pandas as pd
import numpy as np
import cv2


"""
对 DDR_grading 数据集做预处理
1、裁剪四周黑色边缘, 得到正方形
2、全部resize到1024*1024
保存
"""

ori_dir = "/home/mi/Zhankun_work/data/DDR/DR_grading/"
save_dir = "/home/mi/Zhankun_work/data/DDR/DR_grading_after/OriginalImages/"

phaselist = ["train", "valid", "test"]
# lesions_list = ["EX", "HE", "MA", "SE"]


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

    return img_f


def resize(image, mask_list=None, W=1024, H=1024):
    image = cv2.resize(image, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
    return image


def main():
    for phase in phaselist:
        oimg_dir = ori_dir + phase
        imglist_ = os.listdir(oimg_dir)
        imglist_.sort()
        imglist = [imgname_i.split(".")[0] for imgname_i in imglist_]
        for img_name in imglist:
            print(img_name)
            img = cv2.imread(os.path.join(oimg_dir, img_name + ".jpg"))[:,:,::-1].astype(np.uint8)
            # 统裁剪黑边，和 resize
            img = mask_crop(img, mask_list=None)
            # 缩放
            img = resize(img, None, 1024, 1024)
            # 保存
            t = cv2.imwrite(save_dir+phase+"/"+img_name+".jpg", img[:,:,::-1])
            print(t)

if __name__ == "__main__":
    main()
