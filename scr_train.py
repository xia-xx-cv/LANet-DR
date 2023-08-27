# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import random
import copy
from sklearn.metrics import precision_recall_curve, average_precision_score
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

import scr_config as config
from scr_utils import get_images
from scr_dataset import ScrDataset
from torchvision import datasets, models, transforms
from transform.transforms_group import *
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import argparse
from tensorboardX import SummaryWriter
import warnings
from sklearn.metrics import f1_score
import torchvision.utils as t_utils
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, cohen_kappa_score
warnings.filterwarnings('ignore')

from models import LASNet


rotation_angle = config.ROTATION_ANGEL
batchsize = config.TRAIN_BATCH_SIZE

class BCEwithLogistic_loss(nn.Module):
    def __init__(self, weight=None,  pos_weight=None, **kwargs):
        super(BCEwithLogistic_loss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.pos_weight = pos_weight

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = 0
        if self.pos_weight is not None:
            pos_weight = target * (self.pos_weight - 1)
            pos_weight = pos_weight + 1
        else:
            pos_weight = None

        for i in range(target.shape[1]):
            if self.pos_weight is not None:
                bceloss = nn.BCEWithLogitsLoss(pos_weight=pos_weight[:, i])(predict[:, i], target[:, i])
            else:
                bceloss = nn.BCEWithLogitsLoss()(predict[:, i], target[:, i])

            if self.weight is not None:
                assert self.weight.shape[0] == target.shape[1], \
                    'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                bceloss = bceloss * self.weight[i]
            total_loss =total_loss+ bceloss
        total_loss = total_loss.type(torch.FloatTensor)
        return total_loss/target.shape[1]


class LabelSmoothingLossCanonical(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1, weight=None):
        super(LabelSmoothingLossCanonical, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim
        self.weight = weight

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / pred.size(self.dim)
        tmp = -true_dist * pred
        if self.weight is not None:
            tmp = torch.sum(tmp, dim=self.dim)
            loss = 0
            weight_sum = 0
            for i in range(tmp.shape[0]):
                loss = loss + tmp[i] * self.weight[target[i]]
                weight_sum = weight_sum + self.weight[target[i]]
            loss = loss / weight_sum
        else:
            loss = torch.mean(torch.sum(tmp, dim=self.dim))
        return loss


def eval_model(model, eval_loader, criterion=None, device=torch.device("cuda:1"), writer=None, epoch=0):
    model.eval()

    batches_count = 0
    batches = {"EX": 0, "HE": 0, "MA": 0, "SE": 0}
    ap_sum = {"EX": 0, "HE": 0, "MA": 0, "SE": 0}
    dice_sum = {"EX": 0, "HE": 0, "MA": 0, "SE": 0}  # F1, 阈值为0.5

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

            # DR screening
            labels_cls_list.append(label)
            out_sm_cls_list.append(scr_output.softmax(dim=1))
            result_cls = torch.max(scr_output, dim=1)[1]
            sum_num = sum_num + torch.eq(result_cls, label).sum().item()
            n = n + len(label)
            pred_cls_list.append(result_cls)

            loss = criterion(scr_output, label)
            val_loss_sum = val_loss_sum + loss.detach().item()

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
    # labels_cls_list, pred_cls_list
    part = pred_cls_list ^ labels_cls_list             
    pcount = np.bincount(part)             
    tp_list = list(pred_cls_list & labels_cls_list)    
    fp_list = list(pred_cls_list & ~labels_cls_list)   
    tp = tp_list.count(1)                  
    fp = fp_list.count(1)                  
    tn = pcount[0] - tp                    
    fn = pcount[1] - fp                    
    if tp+fp == 0:
        precision = -1
    else:
        precision = tp / (tp+fp)            
    if tp+fn == 0:
        sensitivity = -1
    else:
        sensitivity = tp / (tp+fn)           

    msg = "%s loss=%.4f, acc=%.4f, precision=%.4f, sensitivity=%.4f, f1=%.4f, auc=%.4f, kapa=%.4f" % ("eval:", val_loss, acc, precision, sensitivity, f1, auc, kapa)
    print(msg)

    res = {
        "loss": val_loss,
        "acc": acc,
        "precision": precision,
        "sensitivity": sensitivity,
        "f1": f1,
        "auc": auc,
        "kapa": kapa
    }
    return res


def train_model(model, lesion, preprocess, train_loader, eval_loader, criterion, optimizer, scheduler,
    batch_size, num_epochs=5, start_epoch=0, start_step=0, device=torch.device("cuda:1")):
    model.to(device=device)
    tot_step_count = start_step

    best_acc = -1.

    dir_checkpoint = "./results/" + run_id
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    logdir = "./runs/" + run_id
    writer = SummaryWriter(log_dir=logdir)

    log_txt_path = "./results/" + run_id + "/evalLog.csv"
    with open(log_txt_path, 'a') as f:
        f.write("epoch,loss,acc,    isSave,{}\n".format(time.asctime(time.localtime(time.time()))) )
    db_size = len(train_loader)
    for epoch in range(start_epoch, num_epochs):
        localtime = time.asctime(time.localtime(time.time()))
        print('{} - Starting epoch {}/{}.\n'.format(localtime, epoch + 1, start_epoch+num_epochs))

        model.train()
        for train_item in train_loader:
            if len(train_item)==2:
                inputs, label = train_item
            else:
                inputs, label, true_masks = train_item
                true_masks = true_masks.to(device=device, dtype=torch.float)
            inputs = inputs.to(device=device, dtype=torch.float)
            label = label.to(device=device)
            m_out = model(inputs)
            if len(m_out)==2:
                scr_outputs, masks_pred = m_out
            else:
                scr_outputs = m_out
                masks_pred = None

            loss = criterion(scr_outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # tensorboard loss
            writer.add_scalar('loss', loss.item(), global_step=tot_step_count)
            # tensorboard lr
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=tot_step_count)
            tot_step_count += 1

        # update lr
        scheduler.step()

        if (epoch + 1) % 2 == 0:
            eval_re = eval_model(model, eval_loader, criterion=criterion, device=device, writer=writer, epoch=epoch)
            # tensorboard
            writer.add_scalar('eval-loss', eval_re["loss"], global_step=epoch)
            writer.add_scalar('eval-acc', eval_re["acc"], global_step=epoch)
            writer.add_scalar('eval-precision', eval_re["precision"], global_step=epoch)
            writer.add_scalar('eval-sensitivity', eval_re["sensitivity"], global_step=epoch)
            writer.add_scalar('eval-f1', eval_re["f1"], global_step=epoch)
            writer.add_scalar('eval-auc', eval_re["auc"], global_step=epoch)
            writer.add_scalar('eval-kapa', eval_re["kapa"], global_step=epoch)

            with open(log_txt_path, 'a') as f:
                f.write(str(epoch)+","+",".join([str(eval_re_i) for eval_re_i in list(eval_re.values())]) + ",")
            print("eval-{}: {}".format(epoch, "".join(["{}:{:.4f}, ".format(k, v) for k,v in eval_re.items()])))
            if eval_re["acc"] > best_acc :
                print("best acc is {}".format(eval_re["acc"]))
                with open(log_txt_path, 'a') as f:
                    f.write("best acc, save state! \n")
                best_acc = eval_re["acc"]
                state = {'epoch': epoch,
                    'step': tot_step_count,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
                torch.save(state, os.path.join(dir_checkpoint, run_id + '.pth.tar'))
                torch.save(state, os.path.join(dir_checkpoint, run_id + str(epoch) + '.pth.tar'))
                
            else:
                with open(log_txt_path, 'a') as f:
                    f.write("\n")
    writer.close()



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
    parser.add_argument('--useGPU', type=int, default=5,
                        help='-1: "cpu"; 0, 1, ...: "cuda:x";')
    parser.add_argument('--seed', type=int, default=2021)

    parser.add_argument('--dataset', type=str, default="DDR",
                        help='DDR_seg, DDR, FGADR, NC_dr')
    parser.add_argument('--preprocess', type=str, default='7',
                        help='preprocessing type')
    parser.add_argument('--imagesize', type=int, default=512,
                        help='image size')
    parser.add_argument('--withMasks', type=str2bool, default=False,
                        help='dataset with masks?')

    parser.add_argument('--classes', type=int, default=2,
                        help='分类种类数，2为he123d4，5为01234')

    parser.add_argument('--keep', type=str2bool, default=True,
                        help='keep segNet weights when multi task for training?')
    parser.add_argument('--epochs', type=int, default=30,
                        help='num of epochs')
    parser.add_argument('--balanceSample', type=str2bool, default=False,
                        help='balanceSample')
    parser.add_argument('--net', type=str, default='MTUNet',
                        help='LASNet')
    parser.add_argument('--loadretrained', type=str2bool, default=True,
                        help='pretrained seg model?')
                        
    parser.add_argument('--scrlr', type=float, default=0.0003,
                        help='(init) learning rate')
    parser.add_argument('--scrLossSmooth', type=float, default=0.2,
                        help=' smooth value in LabelSmoothingLossCanonical')
    parser.add_argument('--numworkers', type=int, default=16,
                        help='num_workers')
    parser.add_argument('--run_iters', type=int, default=10,
                        help='')


    args = parser.parse_args()

    test_acc_list = np.zeros(args.run_iters, dtype=np.float)
    test_precision_list = np.zeros(args.run_iters, dtype=np.float)
    test_sensitivity_list = np.zeros(args.run_iters, dtype=np.float)
    test_f1_list = np.zeros(args.run_iters, dtype=np.float)
    test_auc_list = np.zeros(args.run_iters, dtype=np.float)
    test_kapa_list = np.zeros(args.run_iters, dtype=np.float)

    for run_iter_id in range(args.run_iters):
        run_id = "{}_{}_{}_{}-{}_{}_{}-{}_{}-{}_{}-{}-{}".format(
            args.dataset, args.preprocess, args.imagesize, args.withMasks,
            args.keep, args.epochs, args.balanceSample,
            args.net, args.loadretrained,
            args.scrlr, args.scrLossSmooth,
            args.classes,
            run_iter_id
        )
        print(run_id)

        if args.useGPU >= 0:
            device = torch.device("cuda:{}".format(args.useGPU))
        else:
            device = torch.device("cpu")

        lesions = config.LESIONS
        num_lesion = 0
        for key, value in lesions.items():
            if value:
                num_lesion = num_lesion + 1

        image_dir = config.IMAGE_DIRS[args.dataset]
        label_dir = config.LABEL_DIRS[args.dataset]
        image_size = args.imagesize
        if args.loadretrained:
            pretrainedmodelpath = config.PRETRAINED_MODEL_PATHS[args.dataset]
        else:
            pretrainedmodelpath = None


        net_name = args.net
        if net_name == "LASNet":
            model = LASNet(n_channels=3, num_classes=args.classes, output_ch=num_lesion, snapshot=pretrainedmodelpath)
            base, head = [], []
            for name, param in model.named_parameters():
                if 'segNet' in name:
                    if args.keep:
                        param.requires_grad = False  # 固定权值
                    base.append(param)
                else:
                    head.append(param)
            optimizer = torch.optim.AdamW(head, lr=args.scrlr, betas=(0.9, 0.99),
                                        eps=1e-8, weight_decay=0.005)
        else:
            raise ValueError("--net is not allowable??? Please Check!!!")



        resume = False
        if resume:
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                start_epoch = checkpoint['epoch']+1
                start_step = checkpoint['step']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
                print('Model loaded from {}'.format(resume))
            else:
                print("=> no checkpoint found at '{}'".format(resume))
                raise ValueError(" ??? no checkpoint found at '{}'".format(resume))
        else:
            start_epoch = 0
            start_step = 0

        train_getimages= get_images(image_dir, label_dir, args.preprocess, phase='train', withMasks=args.withMasks, classes=args.classes)
        if args.withMasks:
            train_image_paths, train_label_list, train_mask_paths = train_getimages
        else:
            train_image_paths, train_label_list = train_getimages
            train_mask_paths = None

        eval_getimages = get_images(image_dir, label_dir, args.preprocess, phase='eval', withMasks=args.withMasks, classes=args.classes)
        if args.withMasks:
            eval_image_paths, eval_label_list, eval_mask_paths = eval_getimages
        else:
            eval_image_paths, eval_label_list = eval_getimages
            eval_mask_paths = None

        test_getimages = get_images(image_dir, label_dir, args.preprocess, phase='test', withMasks=args.withMasks, classes=args.classes)
        if args.withMasks:
            test_image_paths, test_label_list, test_mask_paths = test_getimages
        else:
            test_image_paths, test_label_list = test_getimages
            test_mask_paths = None

        train_dataset = ScrDataset(train_image_paths, train_mask_paths, label_list=train_label_list, lesions=lesions, transform=
                                Compose([
                                    RandomHorizontalFlip(),
                                    RandomRotation(rotation_angle),
                                    Resize(size=image_size)
                    ]))
        eval_dataset = ScrDataset(eval_image_paths, eval_mask_paths, label_list=eval_label_list, lesions=lesions, transform=
                                Compose([
                                    Resize(size=image_size)
                                ]))
        test_dataset = ScrDataset(test_image_paths, test_mask_paths, label_list=test_label_list, lesions=lesions, transform=
                                Compose([
                                    Resize(size=image_size)
                                ]))
        if args.balanceSample:
            countTrainD = np.zeros(6)
            for _l in train_label_list:
                countTrainD[_l] += 1
            print("Count of train dataset (not dataloader) :{}".format(countTrainD))
            train_weights = 1. / countTrainD
            train_sampleweights = torch.tensor([train_weights[i] for i in train_label_list], dtype=torch.float)
            sampler = WeightedRandomSampler(train_sampleweights,
                                            len(train_sampleweights))
            train_loader = DataLoader(train_dataset, batchsize, sampler=sampler,  num_workers=args.numworkers)
        else:
            train_loader = DataLoader(train_dataset, batchsize, shuffle=True, num_workers=args.numworkers)

        eval_loader = DataLoader(eval_dataset, batchsize, shuffle=False, num_workers=args.numworkers)
        test_loader = DataLoader(test_dataset, batchsize, shuffle=False, num_workers=args.numworkers)

        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)  # T
        criterion = LabelSmoothingLossCanonical(smoothing=args.scrLossSmooth)

        train_model(model, lesions, args.preprocess, train_loader, eval_loader, criterion, optimizer, scheduler,
            batchsize, num_epochs=args.epochs, start_epoch=start_epoch, start_step=start_step, device=device)

        # TEST
        best_pth_path = os.path.join("./results/" + run_id, run_id + '.pth.tar')
        print("=> loading best checkpoint '{}'".format(best_pth_path))
        checkpoint = torch.load(best_pth_path)
        print("| epoch: {}\n| step: {}\n".format(checkpoint["epoch"], checkpoint["step"]))
        model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded from {}'.format(resume))
        test_re = eval_model(model, test_loader, criterion=criterion, device=device)
        
        test_acc_list[run_iter_id] = test_re["acc"]
        test_precision_list[run_iter_id] = test_re["precision"]
        test_sensitivity_list[run_iter_id] = test_re["sensitivity"]
        test_f1_list[run_iter_id] = test_re["f1"]
        test_auc_list[run_iter_id] = test_re["auc"]
        test_kapa_list[run_iter_id] = test_re["kapa"]
        save_test_result_path = os.path.join("./results/" + run_id, 'test-' + run_id + '.txt')
        with open(save_test_result_path, 'w') as f:
            content = ''
            content += '{}\t{}\t{}\t{}\t{}\t{}\n'.format('acc', 'precision', 'sensitivity', 'f1', 'auc', 'kapa')
            content += '{}\t{}\t{}\t{}\t{}\t{}\n'.format(test_re['acc'], test_re['precision'], test_re['sensitivity'], test_re['f1'], test_re['auc'], test_re['kapa'])
            print(content)
            f.write(content)
        print("============ finished ==============")
        print(run_id)
        print("====================================")

    # 循环完所有run_iter_id
    # 计算均值和标准差
    test_acc_avg = {"mean":test_acc_list.mean(), "std":test_acc_list.std()}
    test_precision_avg = {"mean":test_precision_list.mean(), "std":test_precision_list.std()}
    test_sensitivity_avg = {"mean":test_sensitivity_list.mean(), "std":test_sensitivity_list.std()}
    test_f1_avg = {"mean":test_f1_list.mean(), "std":test_f1_list.std()}
    test_auc_avg = {"mean":test_auc_list.mean(), "std":test_auc_list.std()}
    test_kapa_avg = {"mean":test_kapa_list.mean(), "std":test_kapa_list.std()}
    save_all_test_result_path = os.path.join("./results/" + run_id, '_allTestRes_'+run_id + '.txt')
    with open(save_all_test_result_path, 'w') as f:
        content = ''
        content += '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('runIterId', 'acc', 'precision', 'sensitivity', 'f1', 'auc', 'kapa')
        for runIterId in range(args.run_iters):
            content += '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(runIterId, 
                test_acc_list[runIterId], 
                test_precision_list[runIterId], 
                test_sensitivity_list[runIterId], 
                test_f1_list[runIterId], 
                test_auc_list[runIterId], 
                test_kapa_list[runIterId]
                )

        content += '{}\t{}({})\t{}({})\t{}({})\t{}({})\t{}({})\t{}({})\n'.format(
                            "ALL",
                            test_acc_avg['mean'], test_acc_avg['mean'],
                            test_precision_avg['mean'], test_precision_avg['std'],
                            test_sensitivity_avg['mean'], test_sensitivity_avg['std'],
                            test_f1_avg['mean'], test_f1_avg['std'],
                            test_auc_avg['mean'], test_auc_avg['std'],
                            test_kapa_avg['mean'], test_kapa_avg['std'])
        print(content)
        f.write(content)
