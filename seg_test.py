# -*- coding: utf-8 -*-

import sys
from torch.autograd import Variable
import os
from optparse import OptionParser
import numpy as np
import random
import copy
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

import seg_config as config
from models import *
from seg_utils import get_images
from seg_dataset import SegDataset
from torchvision import datasets, models, transforms
from transform.transforms_group import *
from torch.utils.data import DataLoader, Dataset
import argparse

import cv2
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

def eval_model(model, eval_loader, savepathlist=None):
    """

    :param model:
    :param eval_loader:
    :param savepathlist: 结果保存路径列表[[]]（仅当dataloader中 shuffle=False 且 batchsize=1 时有效）
    :return:
    """
    model.eval()

    batches_count = 0
    batches = {"EX": 0, "HE": 0, "MA": 0, "SE": 0}
    mae_sum = {"EX": 0, "HE": 0, "MA": 0, "SE": 0}
    fscore = {"EX": 0, "HE": 0, "MA": 0, "SE": 0}
    number = 256
    mean_pr = {"EX": 0, "HE": 0, "MA": 0, "SE": 0}
    mean_re = {"EX": 0, "HE": 0, "MA": 0, "SE": 0}
    threshod = np.linspace(0, 1, number, endpoint=False)

    auc_sum = {"EX": 0, "HE": 0, "MA": 0, "SE": 0}
    ap_sum = {"EX": 0, "HE": 0, "MA": 0, "SE": 0}
    dice_sum = {"EX": 0, "HE": 0, "MA": 0, "SE": 0}  # F1, 阈值为0.5

    with torch.set_grad_enabled(False):
        for inputs, true_masks in eval_loader:
            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)

            masks_pred = model(inputs)
            if net_name=="LANet":
                masks_pred, out3, out4, out5, out5_ = masks_pred

                # save img of 4 ceng
                if savepathlist is not None:
                    save_path = savepathlist[batches_count]
                    # if save_path[0].find("3722") >= 0:
                    #     save4cengdir = "/home/mi/Zhankun_work/PycharmProjects/A_DR2022_/A_DR2022/Work_0/results/show_work0/show4ceng/"
                    #     tttt = torch.sigmoid(masks_pred).permute(0, 2, 3, 1).cpu().numpy()
                    #     cv2.imwrite(save4cengdir + "PRED_EX.jpg", np.uint8(tttt[0, :, :, 0] * 255))
                    #     cv2.imwrite(save4cengdir + "PRED_HE.jpg", np.uint8(tttt[0, :, :, 1] * 255))
                    #     cv2.imwrite(save4cengdir + "PRED_MA.jpg", np.uint8(tttt[0, :, :, 2] * 255))
                    #     cv2.imwrite(save4cengdir + "PRED_SE.jpg", np.uint8(tttt[0, :, :, 3] * 255))
                    #     tttt = torch.sigmoid(out3).permute(0, 2, 3, 1).cpu().numpy()
                    #     cv2.imwrite(save4cengdir + "out3_EX.jpg", np.uint8(tttt[0, :, :, 0] * 255))
                    #     cv2.imwrite(save4cengdir + "out3_HE.jpg", np.uint8(tttt[0, :, :, 1] * 255))
                    #     cv2.imwrite(save4cengdir + "out3_MA.jpg", np.uint8(tttt[0, :, :, 2] * 255))
                    #     cv2.imwrite(save4cengdir + "out3_SE.jpg", np.uint8(tttt[0, :, :, 3] * 255))
                    #     tttt = torch.sigmoid(out4).permute(0, 2, 3, 1).cpu().numpy()
                    #     cv2.imwrite(save4cengdir + "out4_EX.jpg", np.uint8(tttt[0, :, :, 0] * 255))
                    #     cv2.imwrite(save4cengdir + "out4_HE.jpg", np.uint8(tttt[0, :, :, 1] * 255))
                    #     cv2.imwrite(save4cengdir + "out4_MA.jpg", np.uint8(tttt[0, :, :, 2] * 255))
                    #     cv2.imwrite(save4cengdir + "out4_SE.jpg", np.uint8(tttt[0, :, :, 3] * 255))
                    #     tttt = torch.sigmoid(out5).permute(0, 2, 3, 1).cpu().numpy()
                    #     cv2.imwrite(save4cengdir + "out5_EX.jpg", np.uint8(tttt[0, :, :, 0] * 255))
                    #     cv2.imwrite(save4cengdir + "out5_HE.jpg", np.uint8(tttt[0, :, :, 1] * 255))
                    #     cv2.imwrite(save4cengdir + "out5_MA.jpg", np.uint8(tttt[0, :, :, 2] * 255))
                    #     cv2.imwrite(save4cengdir + "out5_SE.jpg", np.uint8(tttt[0, :, :, 3] * 255))

            masks_soft_batch = torch.sigmoid(masks_pred).permute(0, 2, 3, 1).cpu().numpy()
            masks_hard_batch = true_masks.permute(0, 2, 3, 1).cpu().numpy()

            GT_EX = masks_hard_batch[:, :, :, 0]
            GT_HE = masks_hard_batch[:, :, :, 1]
            GT_MA = masks_hard_batch[:, :, :, 2]
            GT_SE = masks_hard_batch[:, :, :, 3]
            PRED_EX = masks_soft_batch[:, :, :, 0]
            PRED_HE = masks_soft_batch[:, :, :, 1]
            PRED_MA = masks_soft_batch[:, :, :, 2]
            PRED_SE = masks_soft_batch[:, :, :, 3]

            # save results
            if savepathlist is not None:
                save_path = savepathlist[batches_count]
                cv2.imwrite(save_path[0], np.uint8(PRED_EX[0] * 255))
                cv2.imwrite(save_path[1], np.uint8(PRED_HE[0] * 255))
                cv2.imwrite(save_path[2], np.uint8(PRED_MA[0] * 255))
                cv2.imwrite(save_path[3], np.uint8(PRED_SE[0] * 255))

            if GT_EX.sum()!=0:
                batches["EX"] += 1
                # batch mae
                mae_sum["EX"] += np.mean(np.abs(PRED_EX - GT_EX))
                # batch f1
                precision = np.zeros(number)
                recall = np.zeros(number)
                for i in range(number):
                    temp = (PRED_EX >= threshod[i]).astype(float)
                    precision[i] = (temp * GT_EX).sum() / (temp.sum() + 1e-12)
                    recall[i] = (temp * GT_EX).sum() / (GT_EX.sum() + 1e-12)
                mean_pr["EX"] += precision
                mean_re["EX"] += recall
                fscore["EX"] = mean_pr["EX"] * mean_re["EX"] * 2 / (mean_pr["EX"] + mean_re["EX"] + 1e-12)
                GT_EX_flatten = GT_EX.reshape(-1)
                PRED_EX_flatten = PRED_EX.reshape(-1)
                # batch auc
                auc_sum["EX"] += roc_auc_score(y_true=GT_EX_flatten, y_score=PRED_EX_flatten)
                # batch ap
                ap_batch = average_precision_score(GT_EX_flatten, PRED_EX_flatten)
                ap_sum["EX"] += ap_batch
                # # batch DICE
                premask_ = np.where(PRED_EX_flatten > 0.5, 1, 0)
                dice_sum["EX"] += f1_score(GT_EX_flatten, premask_)
            if GT_HE.sum()!=0:
                batches["HE"] += 1
                # batch mae
                mae_sum["HE"] += np.mean(np.abs(PRED_HE - GT_HE))
                # batch f1
                precision = np.zeros(number)
                recall = np.zeros(number)
                for i in range(number):
                    temp = (PRED_HE >= threshod[i]).astype(float)
                    precision[i] = (temp * GT_HE).sum() / (temp.sum() + 1e-12)
                    recall[i] = (temp * GT_HE).sum() / (GT_HE.sum() + 1e-12)
                mean_pr["HE"] += precision
                mean_re["HE"] += recall
                fscore["HE"] = mean_pr["HE"] * mean_re["HE"] * 2 / (mean_pr["HE"] + mean_re["HE"] + 1e-12)
                GT_HE_flatten = GT_HE.reshape(-1)
                PRED_HE_flatten = PRED_HE.reshape(-1)
                # batch auc
                auc_sum["HE"] += roc_auc_score(y_true=GT_HE_flatten, y_score=PRED_HE_flatten)
                # batch ap
                ap_batch = average_precision_score(GT_HE_flatten, PRED_HE_flatten)
                ap_sum["HE"] += ap_batch
                # # batch DICE
                premask_ = np.where(PRED_HE_flatten > 0.5, 1, 0)
                dice_sum["HE"] += f1_score(GT_HE_flatten, premask_)
            if GT_MA.sum()!=0:
                batches["MA"] += 1
                # batch mae
                mae_sum["MA"] += np.mean(np.abs(PRED_MA - GT_MA))
                # batch f1
                precision = np.zeros(number)
                recall = np.zeros(number)
                for i in range(number):
                    temp = (PRED_MA >= threshod[i]).astype(float)
                    precision[i] = (temp * GT_MA).sum() / (temp.sum() + 1e-12)
                    recall[i] = (temp * GT_MA).sum() / (GT_MA.sum() + 1e-12)
                mean_pr["MA"] += precision
                mean_re["MA"] += recall
                fscore["MA"] = mean_pr["MA"] * mean_re["MA"] * 2 / (mean_pr["MA"] + mean_re["MA"] + 1e-12)
                GT_MA_flatten = GT_MA.reshape(-1)
                PRED_MA_flatten = PRED_MA.reshape(-1)
                # batch auc
                auc_sum["MA"] += roc_auc_score(y_true=GT_MA_flatten, y_score=PRED_MA_flatten)
                # batch ap
                ap_batch = average_precision_score(GT_MA_flatten, PRED_MA_flatten)
                ap_sum["MA"] += ap_batch
                # # batch DICE
                premask_ = np.where(PRED_MA_flatten > 0.5, 1, 0)
                dice_sum["MA"] += f1_score(GT_MA_flatten, premask_)
            if GT_SE.sum()!=0:
                batches["SE"] += 1
                # batch mae
                mae_sum["SE"] += np.mean(np.abs(PRED_SE - GT_SE))
                # batch f1
                precision = np.zeros(number)
                recall = np.zeros(number)
                for i in range(number):
                    temp = (PRED_SE >= threshod[i]).astype(float)
                    precision[i] = (temp * GT_SE).sum() / (temp.sum() + 1e-12)
                    recall[i] = (temp * GT_SE).sum() / (GT_SE.sum() + 1e-12)
                mean_pr["SE"] += precision
                mean_re["SE"] += recall
                fscore["SE"] = mean_pr["SE"] * mean_re["SE"] * 2 / (mean_pr["SE"] + mean_re["SE"] + 1e-12)
                GT_SE_flatten = GT_SE.reshape(-1)
                PRED_SE_flatten = PRED_SE.reshape(-1)
                # batch auc
                auc_sum["SE"] += roc_auc_score(y_true=GT_SE_flatten, y_score=PRED_SE_flatten)
                # batch ap
                ap_batch = average_precision_score(GT_SE_flatten, PRED_SE_flatten)
                ap_sum["SE"] += ap_batch
                # # batch DICE
                premask_ = np.where(PRED_SE_flatten > 0.5, 1, 0)
                dice_sum["SE"] += f1_score(GT_SE_flatten, premask_)

            batches_count += 1


        f_score_EX = np.max(fscore["EX"]) / batches["EX"]
        f_score_HE = np.max(fscore["HE"]) / batches["HE"]
        f_score_MA = np.max(fscore["MA"]) / batches["MA"]
        f_score_SE = np.max(fscore["SE"]) / batches["SE"]
        ap_ex = ap_sum["EX"] / batches["EX"]
        ap_he = ap_sum["HE"] / batches["HE"]
        ap_ma = ap_sum["MA"] / batches["MA"]
        ap_se = ap_sum["SE"] / batches["SE"]
        dice_ex = dice_sum["EX"] / batches["EX"]
        dice_he = dice_sum["HE"] / batches["HE"]
        dice_ma = dice_sum["MA"] / batches["MA"]
        dice_se = dice_sum["SE"] / batches["SE"]
        return {"MAE-EX": mae_sum["EX"] / batches["EX"],
                "MAE-HE": mae_sum["HE"] / batches["HE"],
                "MAE-MA": mae_sum["MA"] / batches["MA"],
                "MAE-SE": mae_sum["SE"] / batches["SE"],
                "MAE": (mae_sum["EX"] / batches["EX"] + mae_sum["HE"] / batches["HE"] +
                       mae_sum["MA"] / batches["MA"] + mae_sum["SE"] / batches["SE"]) / 4,
                "f-score-EX": f_score_EX,
                "f-score-HE": f_score_HE,
                "f-score-MA": f_score_MA,
                "f-score-SE": f_score_SE,
                "f-score": (f_score_EX+f_score_HE+f_score_MA+f_score_SE) / 4,
                "AUC-EX": auc_sum["EX"] / batches["EX"],
                "AUC-HE": auc_sum["HE"] / batches["HE"],
                "AUC-MA": auc_sum["MA"] / batches["MA"],
                "AUC-SE": auc_sum["SE"] / batches["SE"],
                "AUC": (auc_sum["EX"] / batches["EX"] + auc_sum["HE"] / batches["HE"] +
                        auc_sum["MA"] / batches["MA"] + auc_sum["SE"] / batches["SE"]) / 4,
                "AP-EX": ap_ex,
                "AP-HE": ap_he,
                "AP-MA": ap_ma,
                "AP-SE": ap_se,
                "mAP": (ap_ex + ap_he + ap_ma + ap_se) / 4,
                "dice-EX": dice_ex,
                "dice-HE": dice_he,
                "dice-MA": dice_ma,
                "dice-SE": dice_se,
                "mdice": (dice_ex + dice_he + dice_ma + dice_se) / 4
            }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--useGPU', type=int, default=0,
                        help='-1: "cpu"; 0, 1, ...: "cuda:x";')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--dataset', type=str, default="try_data",
                        help='IDRiD_seg, DDR_seg, FGADR_seg')
    parser.add_argument('--preprocess', type=str, default='7',
                        help='preprocessing type')
    parser.add_argument('--imagesize', type=int, default=512,
                        help='image size')
    parser.add_argument('--net', type=str, default='UNet',
                        help='UNet, DeepLab, EADNet, LANet,  MTUNet, SambyalModel, RAUNet ')
    parser.add_argument('--model', type=str,
                        help='model path')
    args = parser.parse_args()

    print("test: " + args.model)

    if args.useGPU >= 0:
        device = torch.device("cuda:{}".format(args.useGPU))
    else:
        device = torch.device("cpu")

    lesions = config.LESIONS
    num_lesion = 0
    for key, value in lesions.items():
        if value:
            num_lesion = num_lesion + 1
    #Set random seed for Pytorch and Numpy for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    image_dir = config.IMAGE_DIRS[args.dataset]
    image_size = args.imagesize

    net_name = args.net

    if net_name == "LANet":
        model = LANet(n_channels=3, n_classes=num_lesion)
    else:
        raise ValueError("--net is not allowable??? Please Check!!!")

    resume = args.model

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        print("| epoch: {}\n| step: {}\n".format(checkpoint["epoch"], checkpoint["step"]))
        model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded from {}'.format(resume))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    model.to(device)

    test_image_paths, test_mask_paths = get_images(image_dir, args.preprocess, phase='test')
    test_dataset = SegDataset(test_image_paths, test_mask_paths, lesions=lesions, transform=
                            Compose([
                                Resize(size=image_size)
                            ]))

    savepathlist = []
    save_dir = args.model[:-8] + "-RESULT_IMAGES/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    lesions_list = ["EX", "HE", "MA", "SE"]
    for mask_path4 in test_image_paths:
        newp = []
        for lesion_i in range(4):
            mask_path = mask_path4
            newp.append(os.path.join(save_dir, mask_path.split(".")[-2].split("\\")[-1].split("/")[-1] + "-" + lesions_list[lesion_i] + ".png"))
        savepathlist.append(newp)

    test_loader = DataLoader(test_dataset, 1, shuffle=False)

    re_result = eval_model(model, test_loader, savepathlist=savepathlist)
    print(re_result)
    import pandas as pd
    df = pd.DataFrame(re_result, index=[0])
    df.to_csv(args.model[:-8] + "-test_metrics.csv")

    
