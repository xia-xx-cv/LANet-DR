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

import scr_config as config
from models import *
from scr_utils import get_images
from scr_dataset import ScrDataset
from torchvision import datasets, models, transforms
from transform.transforms_group import *
from torch.utils.data import DataLoader, Dataset
import argparse

import cv2
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix

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

    labels_cls_list = []
    out_sm_cls_list = []
    pred_cls_list = []
    val_loss_sum = 0
    sum_num = 0
    n = 0

    show_flag = True
    with torch.no_grad():
        for eval_item in eval_loader:

            if len(eval_item)==2:
                inputs, label = eval_item
            else:
                inputs, label, true_masks = eval_item
                true_masks = true_masks.to(device=device, dtype=torch.float)

            inputs = inputs.to(device=device, dtype=torch.float)
            label = label.to(device=device)

            m_out = model(inputs)
            if len(m_out)==2:
                scr_output, masks_pred = m_out
            else:
                scr_output = m_out
                masks_pred = None

            if masks_pred is not None:
                masks_soft_batch = masks_pred.permute(0, 2, 3, 1).cpu().numpy()
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

            # DR screening
            labels_cls_list.append(label)
            out_sm_cls_list.append(scr_output.softmax(dim=1))
            result_cls = torch.max(scr_output, dim=1)[1]
            sum_num = sum_num + torch.eq(result_cls, label).sum().item()
            n = n + len(label)
            pred_cls_list.append(result_cls)

            batches_count += 1


        acc = sum_num / n
        val_loss = val_loss_sum / (batches_count)

        labels_cls_list = torch.cat(labels_cls_list, dim=0)
        out_sm_cls_list = torch.cat(out_sm_cls_list, dim=0)
        pred_cls_list = torch.cat(pred_cls_list, dim=0)
        labels_cls_list = labels_cls_list.detach().cpu()
        out_sm_cls_list = out_sm_cls_list.detach().cpu()
        pred_cls_list = pred_cls_list.detach().cpu()

        if 1:
            report = classification_report(labels_cls_list, pred_cls_list, digits=3)
            print('@ classification_report ==\n', report)
        f1 = f1_score(labels_cls_list, pred_cls_list, average='micro')
        try:
            num_classes = 2
            if num_classes > 2:
                auc = roc_auc_score(labels_cls_list, out_sm_cls_list, multi_class='ovr')
            else:
                auc = roc_auc_score(labels_cls_list, out_sm_cls_list[:, 1])
        except ValueError:
            auc = -1
        try:
            kapa = cohen_kappa_score(labels_cls_list, pred_cls_list)
        except:
            kapa = -1

        msg = "%s loss=%.4f, acc=%.4f, f1=%.4f, auc=%.4f, kapa=%.4f" % ("eval:", val_loss, acc, f1, auc, kapa)
        print(msg)

        res = {
            "loss": val_loss,
            "acc": acc,
            "f1": f1,
            "auc": auc,
            "kapa": kapa
        }
        return res

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--useGPU', type=int, default=0,
                        help='-1: "cpu"; 0, 1, ...: "cuda:x";')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--dataset', type=str, default="DDR",
                        help='DDR_seg, DDR, FGADR, FGADR_sel')
    parser.add_argument('--preprocess', type=str, default='7',
                        help='preprocessing type')
    parser.add_argument('--imagesize', type=int, default=512,
                        help='image size')
    parser.add_argument('--withMasks', type=str2bool, default=False,
                        help='dataset with masks?')

    parser.add_argument('--net', type=str, default='UNet',
                        help='LASNet ')
    parser.add_argument('--model', type=str,
                        help='model path')
    args = parser.parse_args()

    print("=======\ntest: " + args.model + "=======")

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
    label_dir = config.LABEL_DIRS[args.dataset]
    image_size = args.imagesize

    net_name = args.net

    if net_name == "LASNet":
        model = LASNet(n_channels=3, num_classes=2, output_ch=num_lesion, snapshot=None)
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

    test_getimages = get_images(image_dir, label_dir, args.preprocess, phase='test', withMasks=args.withMasks)
    if args.withMasks:
        eval_image_paths, eval_label_list, eval_mask_paths = test_getimages
    else:
        eval_image_paths, eval_label_list = test_getimages
        eval_mask_paths = None
    eval_dataset = ScrDataset(eval_image_paths, eval_mask_paths, label_list=eval_label_list, lesions=lesions, transform=
                            Compose([
                                Resize(size=image_size)
                            ]))

    savepathlist = []
    save_dir = args.model[:-8] + "-RESULT_IMAGES/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    lesions_list = ["EX", "HE", "MA", "SE"]
    for mask_path4 in eval_image_paths:
        newp = []
        for lesion_i in range(4):
            mask_path = mask_path4
            newp.append(os.path.join(save_dir, mask_path.split(".")[-2].split("\\")[-1].split("/")[-1] + "-" + lesions_list[lesion_i] + ".png"))
        savepathlist.append(newp)

    test_loader = DataLoader(eval_dataset, 1, shuffle=False)

    re_result = eval_model(model, test_loader, savepathlist=savepathlist)
    print(re_result)
    import pandas as pd
    df = pd.DataFrame(re_result, index=[0])
    df.to_csv(args.model[:-8] + "-test_metrics.csv")

    
