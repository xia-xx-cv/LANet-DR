# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import random
from sklearn.metrics import precision_recall_curve, average_precision_score
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from pathlib import Path
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import argparse
from tensorboardX import SummaryWriter
import warnings
from sklearn.metrics import f1_score
import torchvision.utils as t_utils

import seg_config as config
from models import LANet
from seg_utils import get_images
from seg_dataset import SegDataset
from transform.transforms_group import *

warnings.filterwarnings('ignore')




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


class DiceBCELoss(nn.Module):
    def __init__(self, diceweight=None, size_average=True, pos_weight=None):
        super(DiceBCELoss, self).__init__()
        self.diceweight = diceweight
        self.pos_weight = pos_weight

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs_sig = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        inputs_sig = inputs_sig.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs_sig * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_sig.sum() + targets.sum() + smooth)

        if self.pos_weight is not None:
            pos_weight = targets * (self.pos_weight - 1)
            pos_weight = pos_weight + 1
        else:
            pos_weight = None
        BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(inputs, targets)
        Dice_BCE = BCE + dice_loss * self.diceweight

        return Dice_BCE


def eval_model(model, eval_loader, device=torch.device("cuda:1"), writer=None, epoch=0):
    model.eval()

    batches_count = 0
    batches = {"EX": 0, "HE": 0, "MA": 0, "SE": 0}
    ap_sum = {"EX": 0, "HE": 0, "MA": 0, "SE": 0}
    dice_sum = {"EX": 0, "HE": 0, "MA": 0, "SE": 0}  # F1, 阈值为0.5

    show_flag = True
    with torch.no_grad():
        for inputs, true_masks in eval_loader:
            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)

            masks_pred = model(inputs)
            if net_name=="LANet":
                masks_pred, out3, out4, out5, out5_ = masks_pred

            masks_soft_batch = torch.sigmoid(masks_pred).permute(0, 2, 3, 1).cpu().numpy()
            masks_hard_batch = true_masks.permute(0, 2, 3, 1).cpu().numpy()

            true_masks_indices = true_masks.permute(0, 2, 3, 1)
            true_masks_indices = true_masks_indices.reshape(-1, true_masks_indices.shape[-1])
            masks_pred_transpose = masks_pred.permute(0, 2, 3, 1)  
            masks_pred_transpose = masks_pred_transpose.reshape(-1, masks_pred_transpose.shape[-1]) 
            loss_bce = criterion(masks_pred_transpose, true_masks_indices).item()


            for test_b_i in range(len(inputs)):
                GT_EX = masks_hard_batch[test_b_i, :, :, 0]
                GT_HE = masks_hard_batch[test_b_i, :, :, 1]
                GT_MA = masks_hard_batch[test_b_i, :, :, 2]
                GT_SE = masks_hard_batch[test_b_i, :, :, 3]
                PRED_EX = masks_soft_batch[test_b_i, :, :, 0]
                PRED_HE = masks_soft_batch[test_b_i, :, :, 1]
                PRED_MA = masks_soft_batch[test_b_i, :, :, 2]
                PRED_SE = masks_soft_batch[test_b_i, :, :, 3]
                show_flag_i = 0
                if GT_EX.sum()!=0:
                    show_flag_i += 1
                    batches["EX"] += 1
                    GT_EX_flatten = GT_EX.reshape(-1)
                    PRED_EX_flatten = PRED_EX.reshape(-1)
                    # batch ap
                    ap_batch = average_precision_score(GT_EX_flatten, PRED_EX_flatten)
                    ap_sum["EX"] += ap_batch
                    # # batch DICE
                    premask_ = np.where(PRED_EX_flatten > 0.5, 1, 0)
                    dice_sum["EX"] += f1_score(GT_EX_flatten, premask_)
                if GT_HE.sum()!=0:
                    show_flag_i += 1
                    batches["HE"] += 1
                    GT_HE_flatten = GT_HE.reshape(-1)
                    PRED_HE_flatten = PRED_HE.reshape(-1)
                    # batch ap
                    ap_batch = average_precision_score(GT_HE_flatten, PRED_HE_flatten)
                    ap_sum["HE"] += ap_batch
                    # # batch DICE
                    premask_ = np.where(PRED_HE_flatten > 0.5, 1, 0)
                    dice_sum["HE"] += f1_score(GT_HE_flatten, premask_)
                if GT_MA.sum()!=0:
                    show_flag_i += 1
                    batches["MA"] += 1
                    GT_MA_flatten = GT_MA.reshape(-1)
                    PRED_MA_flatten = PRED_MA.reshape(-1)
                    # batch ap
                    ap_batch = average_precision_score(GT_MA_flatten, PRED_MA_flatten)
                    ap_sum["MA"] += ap_batch
                    # # batch DICE
                    premask_ = np.where(PRED_MA_flatten > 0.5, 1, 0)
                    dice_sum["MA"] += f1_score(GT_MA_flatten, premask_)
                if GT_SE.sum()!=0:
                    show_flag_i += 1
                    batches["SE"] += 1
                    GT_SE_flatten = GT_SE.reshape(-1)
                    PRED_SE_flatten = PRED_SE.reshape(-1)
                    # batch ap
                    ap_batch = average_precision_score(GT_SE_flatten, PRED_SE_flatten)
                    ap_sum["SE"] += ap_batch
                    # # batch DICE
                    premask_ = np.where(PRED_SE_flatten > 0.5, 1, 0)
                    dice_sum["SE"] += f1_score(GT_SE_flatten, premask_)
                if show_flag and show_flag_i == 4:
                    if writer is not None:
                        img_grid = t_utils.make_grid(
                            torch.unsqueeze(true_masks[test_b_i], dim=1),  
                            nrow=4,  
                            padding=2,  
                            normalize=False,  
                            range=None, 
                            scale_each=False,  
                            pad_value=1,  
                        )
                        writer.add_image("test_x", inputs[test_b_i], epoch, dataformats='CHW')
                        writer.add_image("GT_x", img_grid, epoch, dataformats='CHW')
                        img_grid = t_utils.make_grid(
                            torch.unsqueeze(masks_pred[test_b_i].sigmoid(), dim=1), 
                            nrow=4,  
                            padding=2,  
                            normalize=False, 
                            range=None, 
                            scale_each=False,  
                            pad_value=1, 
                        )
                        writer.add_image("preds_x", img_grid, epoch, dataformats='CHW')
                    
                    show_flag = False
            
            batches_count += 1


    ap_ex = ap_sum["EX"] / batches["EX"]
    ap_he = ap_sum["HE"] / batches["HE"]
    ap_ma = ap_sum["MA"] / batches["MA"]
    ap_se = ap_sum["SE"] / batches["SE"]
    dice_ex = dice_sum["EX"] / batches["EX"]
    dice_he = dice_sum["HE"] / batches["HE"]
    dice_ma = dice_sum["MA"] / batches["MA"]
    dice_se = dice_sum["SE"] / batches["SE"]
    return {"loss": loss_bce,
            "AP-EX": ap_ex,
            "AP-HE": ap_he,
            "AP-MA": ap_ma,
            "AP-SE": ap_se,
            "mAP": (ap_ex+ap_he+ap_ma+ap_se)/4,
            "dice-EX": dice_ex,
            "dice-HE": dice_he,
            "dice-MA": dice_ma,
            "dice-SE": dice_se,
            "mdice": (dice_ex+dice_he+dice_ma+dice_se)/4
            }



def train_model(model, lesion, preprocess, train_loader, eval_loader, criterion, g_optimizer, g_scheduler, 
    batch_size, num_epochs=5, start_epoch=0, start_step=0, device=torch.device("cuda:1")):
    model.to(device=device)
    tot_step_count = start_step

    best_map = -1.
    best_mdice = -1.
    dir_checkpoint = "./results/" + run_id
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    logdir = "./runs/" + run_id
    writer = SummaryWriter(log_dir=logdir)

    log_txt_path = "./results/" + run_id + "/evalLog.csv"
    with open(log_txt_path, 'a') as f:
        f.write("epoch,loss,AP-EX,AP-HE,AP-MA,AP-SE,mAP,dice-ex,dice-he,dice-ma,dice-se,mdice,isSave,{}\n".format(time.asctime(time.localtime(time.time()))))

    db_size = len(train_loader)
    for epoch in range(start_epoch, num_epochs):
        localtime = time.asctime(time.localtime(time.time()))
        print('{} - Starting epoch {}/{}.\n'.format(localtime, epoch + 1, start_epoch+num_epochs))
        if g_scheduler is not None:
            g_scheduler.step()
        else:
            pass

        model.train()
        for inputs, true_masks in train_loader:
            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)

            # step lr for LANet
            if g_scheduler is None:
                # net_name == "LANet"
                niter = tot_step_count
                lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, num_epochs * db_size, niter, ratio=1.)
                g_optimizer.param_groups[0]['lr'] = 0.1 * lr  # for backbone
                g_optimizer.param_groups[1]['lr'] = lr
                g_optimizer.momentum = momentum


            masks_pred = model(inputs)

            true_masks_indices = true_masks.permute(0, 2, 3, 1)
            true_masks_flat = true_masks_indices.reshape(-1, true_masks_indices.shape[-1])

            if net_name == "LANet":
                out2, out3, out4, out5, out5_ = masks_pred
                if out2 is not None:
                    masks_pred_transpose = out2.permute(0, 2, 3, 1)  
                    masks_pred_flat_out2 = masks_pred_transpose.reshape(-1, masks_pred_transpose.shape[-1])  
                    loss2 = criterion(masks_pred_flat_out2, true_masks_flat)
                else:
                    loss2 = torch.zeros(1)
                if out3 is not None:
                    masks_pred_transpose = out3.permute(0, 2, 3, 1)  
                    masks_pred_flat_out3 = masks_pred_transpose.reshape(-1, masks_pred_transpose.shape[-1])  
                    loss3 = criterion(masks_pred_flat_out3, true_masks_flat)
                else:
                    loss3 = torch.zeros(1)
                if out4 is not None:
                    masks_pred_transpose = out4.permute(0, 2, 3, 1)  
                    masks_pred_flat_out4 = masks_pred_transpose.reshape(-1, masks_pred_transpose.shape[-1])  
                    loss4 = criterion(masks_pred_flat_out4, true_masks_flat)
                else:
                    loss4 = torch.zeros(1)
                if out5 is not None:
                    masks_pred_transpose = out5.permute(0, 2, 3, 1)  
                    masks_pred_flat_out5 = masks_pred_transpose.reshape(-1, masks_pred_transpose.shape[-1])  
                    loss5 = criterion(masks_pred_flat_out5, true_masks_flat)
                else:
                    loss5 = torch.zeros(1)

                loss_ce = loss2 * 1
                if out3 is not None:
                    loss_ce = loss_ce + loss3 * 0.8
                if out4 is not None:
                    loss_ce = loss_ce + loss4 * 0.6
                if out5 is not None:
                    loss_ce = loss_ce + loss5 * 0.4
            else:
                masks_pred_transpose = masks_pred.permute(0, 2, 3, 1)  
                masks_pred_flat = masks_pred_transpose.reshape(-1, masks_pred_transpose.shape[-1]) 
                loss_ce = criterion(masks_pred_flat, true_masks_flat)

            ce_weight = 1.
            g_loss = loss_ce * ce_weight

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # tensorboard loss
            writer.add_scalar('g_loss', g_loss.item(), global_step=tot_step_count)
            # tensorboard lr
            if net_name == "LANet":
                writer.add_scalar('lr', g_optimizer.param_groups[1]['lr'], global_step=tot_step_count)
            else:
                writer.add_scalar('lr', next(iter(g_optimizer.param_groups))['lr'], global_step=tot_step_count)

            tot_step_count += 1

        if (epoch + 1) % 10 == 0:
            eval_re = eval_model(model, eval_loader, device=device, writer=writer, epoch=epoch)
            # tensorboard
            writer.add_scalar('eval-loss', eval_re["loss"], global_step=epoch)
            writer.add_scalars('eval', {'eval_map': eval_re["mAP"], 'eval_mdice': eval_re["mdice"]}, global_step=epoch)

            with open(log_txt_path, 'a') as f:
                f.write(str(epoch)+","+",".join([str(eval_re_i) for eval_re_i in list(eval_re.values())]) + ",")
            print("eval-{}: {}".format(epoch, "".join(["{}:{:.4f}, ".format(k, v) for k,v in eval_re.items()])))
            if eval_re["mAP"] > best_map :
                print("best map is {}".format(eval_re["mAP"]))
                with open(log_txt_path, 'a') as f:
                    f.write("best mAP, save state! \n")
                best_map = eval_re["mAP"]
                state = {'epoch': epoch,
                    'step': tot_step_count,
                    'state_dict': model.state_dict(),
                    'optimizer': g_optimizer.state_dict()}
                torch.save(state, os.path.join(dir_checkpoint, run_id + '.pth.tar'))
            elif eval_re["mdice"]>best_mdice:
                print("best dice is {}".format(eval_re["mdice"]))
                with open(log_txt_path, 'a') as f:
                    f.write("best mdice, save state! \n")
                best_mdice = eval_re["mdice"]
                state = {'epoch': epoch,
                    'step': tot_step_count,
                    'state_dict': model.state_dict(),
                    'optimizer': g_optimizer.state_dict()}
                torch.save(state, os.path.join(dir_checkpoint, run_id + '.pth.tar'))
            else:
                with open(log_txt_path, 'a') as f:
                    f.write("\n")
    writer.close()


def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1.,
                    annealing_decay=1e-2, momentums=[0.95, 0.85]):
    """ set lr """
    first = int(total_steps * ratio)
    last = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur / total_steps)
    x = np.abs(cur * 2.0 / total_steps - 2.0 * cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr) * cur + min_lr * first - base_lr * total_steps) / (first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1. - x)
        else:
            momentum = momentums[0]

    return lr, momentum

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
    parser.add_argument('--dataset', type=str, default="try_data",
                        help='IDRiD_seg, DDR_seg, FGADR_seg')
    parser.add_argument('--preprocess', type=str, default='7',
                        help='preprocessing type')
    parser.add_argument('--imagesize', type=int, default=512,
                        help='image size')
    parser.add_argument('--epochs', type=int, default=300,
                        help='num of epochs')
    parser.add_argument('--balanceSample', type=str2bool, default=False,
                        help='balanceSample')
    parser.add_argument('--net', type=str, default='UNet',
                        help='UNet, DeepLab, EADNet, LANet, HEDNet, DenseUNet, MTUNet, SambyalModel, RAUNet')
    parser.add_argument('--posw', type=int, default=10,
                        help='pos. weight')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='(init) learning rate')
    parser.add_argument('--lossfun', type=str, default='bce',
                        help='loss function; bce, bcedice')
    parser.add_argument('--numworkers', type=int, default=8,
                        help='num_workers')

    args = parser.parse_args()
    run_id = "{}_{}_{}_{}_{}-{}_{}_({})_{}".format(args.dataset, args.preprocess, args.imagesize, args.epochs, args.balanceSample,
                                                args.net, args.posw, args.lr, args.lossfun)
    print(run_id)

    rotation_angle = config.ROTATION_ANGEL
    batchsize = config.TRAIN_BATCH_SIZE

    if args.useGPU >= 0:
        device = torch.device("cuda:{}".format(args.useGPU))
    else:
        device = torch.device("cpu")

    lesions = config.LESIONS
    num_lesion = 0
    for key, value in lesions.items():
        if value:
            num_lesion = num_lesion + 1
    # Set random seed for Pytorch and Numpy for reproducibility
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
        raise ValueError("This model is not supported.")

    if net_name == "LANet":
        base, head = [], []
        for name, param in model.named_parameters():
            if 'bkbone' in name:
                base.append(param)
            else:
                head.append(param)
        g_optimizer = torch.optim.SGD([{'params': base, 'lr': args.lr*0.1}, {'params': head}], lr=args.lr, momentum=0.9,
                                         weight_decay=0.0005, nesterov=True)
    else:
        g_optimizer = optim.SGD(model.parameters(),
                                  lr=args.lr,
                                  momentum=0.9,
                                  weight_decay=0.0005)

    resume = False
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']+1
            start_step = checkpoint['step']
            model.load_state_dict(checkpoint['state_dict'])
            g_optimizer.load_state_dict(checkpoint['optimizer'])
            print('Model loaded from {}'.format(resume))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            raise ValueError(" ??? no checkpoint found at '{}'".format(resume))
    else:
        start_epoch = 0
        start_step = 0

    train_image_paths, train_mask_paths = get_images(image_dir, args.preprocess, phase='train')
    eval_image_paths, eval_mask_paths = get_images(image_dir, args.preprocess, phase='eval')

    train_dataset = SegDataset(train_image_paths, train_mask_paths, lesions=lesions, transform=
                            Compose([
                                RandomHorizontalFlip(),
                                RandomRotation(rotation_angle),
                                Resize(size=image_size)
                ]))
    eval_dataset = SegDataset(eval_image_paths, eval_mask_paths, lesions=lesions, transform=
                            Compose([
                                Resize(size=image_size)
                            ]))

    if args.balanceSample:
        train_loader = DataLoader(train_dataset, batchsize, shuffle=True, num_workers=args.numworkers)
    else:
        train_loader = DataLoader(train_dataset, batchsize, shuffle=True, num_workers=args.numworkers)

    eval_loader = DataLoader(eval_dataset, batchsize, shuffle=False, num_workers=args.numworkers)

    if net_name == "LANet":
        g_scheduler = None
        BASE_LR = 1e-3
        MAX_LR = 0.1
    else:
        g_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=20, gamma=0.9)

    if args.lossfun == "bce":
        criterion = BCEwithLogistic_loss(weight=torch.FloatTensor(config.CROSSENTROPY_WEIGHTS).to(device), pos_weight=args.posw)
    elif args.lossfun == "bcedice":
        # diceweight = config.LESION_DICE_WEIGHT
        diceweight = 1.0
        criterion = DiceBCELoss(diceweight=diceweight, pos_weight=args.posw)

    else:
        raise ValueError("--lossfun is not allowable??? Please Check!!!")


    train_model(model, lesions, args.preprocess, train_loader, eval_loader, criterion, g_optimizer, g_scheduler,
        batchsize, num_epochs=args.epochs, start_epoch=start_epoch, start_step=start_step, device=device)
