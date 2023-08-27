# -*- coding: utf-8 -*-

import os
import glob
import scr_config as config
from preprocess import clahe_gridsize
import cv2
import torch.nn as nn

def get_images(image_dir, label_dir, preprocess='7', phase='train', withMasks=False, classes=2):
    if phase == 'train':
        setname = 'train'
    elif phase == 'eval':
        setname = "validation"
    elif phase == 'test':
        setname = 'test'
    else:
        raise ValueError("utils phase ERROR!!!")

    limit = 2
    grid_size = 8
    if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE' + preprocess)):
        os.mkdir(os.path.join(image_dir, 'Images_CLAHE' + preprocess))
    if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname)):
        os.mkdir(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname))
        
        # compute mean brightess
        meanbright = 0.
        images_number = 0
        for tempsetname in ['train', 'validation', 'test']:
            imgs_ori = glob.glob(os.path.join(image_dir, 'OriginalImages/'+ tempsetname + '/*.jpg'))
            imgs_ori.sort()
            images_number += len(imgs_ori)
            # mean brightness.
            for img_path in imgs_ori:
                img_name = os.path.split(img_path)[-1].split('.')[0]
                mask_path = "./mask.png"
                gray = cv2.imread(img_path, 0)
                mask_img = cv2.imread(mask_path, 0)
                brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
                meanbright += brightness
        meanbright /= images_number
        
        imgs_ori = glob.glob(os.path.join(image_dir, 'OriginalImages/' + setname + '/*.jpg'))
        
        preprocess_dict = {'0': [False, False, None], '1': [False, False, meanbright], '2': [False, True, None], '3': [False, True, meanbright], '4': [True, False, None], '5': [True, False, meanbright], '6': [True, True, None], '7': [True, True, meanbright]}
        for img_path in imgs_ori:
            img_name = os.path.split(img_path)[-1].split('.')[0]
            mask_path = os.path.join(image_dir, 'AllSegmentationGroundtruths', setname, 'Mask', img_name + '_MASK.tif')
            clahe_img = clahe_gridsize(img_path, mask_path, denoise=preprocess_dict[preprocess][0], contrastenhancement=preprocess_dict[preprocess][1], brightnessbalance=preprocess_dict[preprocess][2], cliplimit=limit, gridsize=grid_size)
            cv2.imwrite(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname, os.path.split(img_path)[-1]), clahe_img)

    label_path = os.path.join(label_dir, setname+".txt")
    with open(label_path, "r", encoding="utf8") as f:
        lines = f.readlines()
    datas = []
    for line in lines:
        name, label = line.rstrip().split()
        if label != '5':  
            label_tmp = int(label)
            if classes == 2:
                if label != "4":
                    if label_tmp > 0:
                        label_tmp = 1
                    datas.append(
                        (name, label_tmp)  
                    )
            elif classes == 5:
                datas.append(
                    (name, label_tmp)  
                )


    mask_paths = []
    image_paths = []
    label_list = []
    for name, label in datas:
        image_paths.append(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname, name))
        label_list.append(label)
    if withMasks:
        mask_path = os.path.join(image_dir, 'AllSegmentationGroundtruths', setname)
        lesions = ['EX', 'HE', 'MA', 'SE', 'Mask']
        lesion_abbvs = ['EX', 'HE', 'MA', 'SE', 'MASK']
        for image_path in image_paths:
            paths = []
            name = os.path.split(image_path)[1].split('.')[0]
            for lesion, lesion_abbv in zip(lesions, lesion_abbvs):
                if lesion=="Mask":
                    candidate_path = os.path.join(mask_path, lesion, name + '_MASK.tif')
                else:
                    candidate_path = os.path.join(mask_path, lesion, name + '.tif')
                if os.path.exists(candidate_path):
                    paths.append(candidate_path)
                else:
                    paths.append(None)
            mask_paths.append(paths)
        return image_paths, label_list, mask_paths
    else:
        return image_paths, label_list


if __name__=="__main__":
    """
    test 
    """
    image_dir = config.IMAGE_DIRS["DDR"]
    label_dir = config.LABEL_DIRS["DDR"]
    get_images(image_dir, label_dir, preprocess='7', phase='test')


