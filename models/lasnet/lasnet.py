# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Lesion-Aware Screening Network (LASNet) 
for stage 2 DR screening
"""


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out      = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out      = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out      = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out+residual, inplace=True)


class ResNet(nn.Module):
    def __init__(self, n_channels=3):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes*4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))

        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('./pre-trained/resnet50-19c8e357.pth'), strict=False)


class Head(nn.Module):
    """
    HAM (Head Attention Module).
    """
    def __init__(self, in_channel):
        super(Head, self).__init__()

        self.conv0 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel, 512, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(in_channel, 256, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.initialize_weights()

    def forward(self, input):
        left = F.relu(self.bn0(self.conv0(input)), inplace=True)
        wb = self.conv1(input)
        w, b = wb[:,:256,:,:], wb[:,256:,:,:]
        mid = F.relu(w * left + b, inplace=True)

        mid = F.relu(self.bn0(self.conv2(mid)), inplace=True)
        down = input.mean(dim=(2,3), keepdim=True)
        down = F.relu(self.conv3(down), inplace=True)
        down = torch.sigmoid(self.conv4(down))

        return mid * down

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


"""
FPM (Feature-Preserve Module)
contains Feature-Preserve Block (FPB) and Feature Fusion Block (FFB).
"""
class FPB(nn.Module):
    """
    Feature-Preserve Block (FPB).
    """
    def __init__(self, in_channel):
        super(FPB, self).__init__()
        self.conv0 = nn.Conv2d(in_channel, 512, kernel_size=1, stride=1, padding=0, bias=False)  # bias
        self.bn0   = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.initialize_weights()

    def forward(self, fea):
        fea = F.relu(self.bn0(self.conv0(fea)), inplace=True)
        down = fea.mean(dim=(2,3), keepdim=True)
        down = F.relu(self.conv1(down), inplace=True)
        down = torch.sigmoid(self.conv2(down))
        return down

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class FFB(nn.Module):
    """
    Feature Fusion Block (FFB).
    """
    def __init__(self, in_channel_left, in_channel_down, in_channel_right):
        super(FFB, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(256)

        self.conv_d1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_d2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_l  = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3   = nn.Conv2d(256*3, 256, kernel_size=3, stride=1, padding=1)
        self.bn3     = nn.BatchNorm2d(256)

        self.initialize_weights()

    def forward(self, f_enc, f_dec, f_p):
        f_enc = F.relu(self.bn0(self.conv0(f_enc)), inplace=True) #256 channels
        down_2 = self.conv_d2(f_enc)
        z1 = F.relu(down_2 * f_p, inplace=True)

        down = F.relu(self.bn1(self.conv1(f_dec)), inplace=True) #256 channels

        if down.size()[2:] != f_enc.size()[2:]:
            down = F.interpolate(down, size=f_enc.size()[2:], mode='bilinear')
            z2 = F.relu(f_p * down, inplace=True)
        else:
            z2 = F.relu(f_p * down, inplace=True)

        if f_dec.size()[2:] != f_enc.size()[2:]:
            down_1 = F.interpolate(f_dec, size=f_enc.size()[2:], mode='bilinear')
            z3 = F.relu(down_1 * f_p, inplace=True)
        else:
            z3 = F.relu(f_dec * f_p, inplace=True)

        out = torch.cat((z1, z2, z3), dim=1)
        out = F.relu(self.bn3(self.conv3(out)), inplace=True)  # f_f
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class LAM(nn.Module):
    """
    LAM (Lesion-Aware Module).
    """
    def __init__(self, in_channels):
        super(LAM, self).__init__()

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU(True))

        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU(True))
        self.conv1_2_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv1_2_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        inter_channels = in_channels
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU(True))

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,  bias=False),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU(True))

        self.initialize_weights()

    def forward(self, x):
        x1 = self.conv1_1(x)

        x1_1 = self.conv2_1(x1)  
        x1_2 = self.conv2_2(x1)  
        x1 = self.conv3(F.relu(x1_1 + x1_2, inplace=True)) 

        x2 = self.conv1_2(x)

        x2_ca = x2.mean(dim=(2,3), keepdim=True)
        x2_ca = F.relu(self.conv1_2_1(x2_ca), inplace=True)
        x2_ca = torch.sigmoid(self.conv1_2_2(x2_ca))  

        x = self.conv4(x1 * x2_ca)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

#
class LANet(nn.Module):
    """
    LANet (Lesion-Aware Network) (overall)
    """
    def __init__(self, n_channels=3, n_classes=4, snapshot=None):
        super(LANet, self).__init__()
        self.snapshot = snapshot
        self.bkbone  = ResNet(n_channels=n_channels)

        self.fpb45 = FPB(2048)
        self.fpb35 = FPB(2048)
        self.fpb25 = FPB(2048)

        self.head = Head(2048)

        self.ffb45   = FFB(1024,  256, 256)
        self.ffb34   = FFB( 512,  256, 256)
        self.ffb23   = FFB( 256,  256, 256)

        self.lam5    = LAM(256)
        self.lam4    = LAM(256)
        self.lam3    = LAM(256)
        self.lam2    = LAM(256)

        # self.linear5 = nn.Conv2d(256, n_classes, kernel_size=3, stride=1, padding=1)
        # self.linear4 = nn.Conv2d(256, n_classes, kernel_size=3, stride=1, padding=1)
        # self.linear3 = nn.Conv2d(256, n_classes, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(256, n_classes, kernel_size=3, stride=1, padding=1)
        self.initialize()

    def forward(self, x):
        out1, out2, out3, out4, out5_ = self.bkbone(x)

        out3_c = out3

        # FPB
        out4_a = self.fpb45(out5_)
        out3_a = self.fpb35(out5_)
        out2_a = self.fpb25(out5_)

        # HAM
        out5 = self.head(out5_) 

        # out
        out5 = self.lam5(out5) 
        out4 = self.lam4(self.ffb45(out4, out5, out4_a)) 
        out3 = self.lam3(self.ffb34(out3, out4, out3_a)) 
        out2 = self.lam2(self.ffb23(out2, out3, out2_a)) 

        # we use bilinear interpolation instead of transpose convolution
        # out5  = F.interpolate(self.linear5(out5), size=x.size()[2:], mode='bilinear')
        # out4  = F.interpolate(self.linear4(out4), size=x.size()[2:], mode='bilinear')
        # out3  = F.interpolate(self.linear3(out3), size=x.size()[2:], mode='bilinear')
        pred_mask  = torch.sigmoid(F.interpolate(self.linear2(out2), size=x.size()[2:], mode='bilinear'))
        return pred_mask, out2, out4, out5_, out3_c


    def initialize(self):
        if self.snapshot:
            try:
                miss, unexpect = self.load_state_dict(torch.load(self.snapshot, map_location='cuda:0')['state_dict'], strict=False)
                print("==> loaded pre-trained LANet from {}: \nmiss={}, unexpect={}".format(self.snapshot, miss, unexpect))
            except:
                print("Warning: please check the snapshot file:", self.snapshot)
                pass
        else:
            for n, m in self.named_children():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

class LASNet(nn.Module):
    """
    LASNet (Lesion-Aware Screening Network)
    """
    def __init__(self, n_channels=3, num_classes=5, output_ch=4, snapshot=None):
        super(LASNet, self).__init__()
        self.segNet = LANet(n_channels=n_channels, n_classes=output_ch, snapshot=None)

        self.head2 = Head(2048)
        self.conv_c3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

        self.maxpool4 = nn.MaxPool2d((2,2), stride=2)
        self.maxpool3 = nn.MaxPool2d((4, 4), stride=4)
        self.maxpool2 = nn.MaxPool2d((8, 8), stride=8)

        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.convout = nn.Conv2d(256 * 4, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnout = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)

        self.initialize()
        if snapshot:
            self.segNet.snapshot = snapshot
            self.segNet.initialize()

    def forward(self, x):
        pred_mask, out2, out4, out5_, out3_c = self.segNet(x)

        # only get out5_c， out4, out2， out3_c
        out3_c = self.conv_c3(out3_c) 
        out5_c = self.head2(out5_)

        out4 = self.maxpool4(out4)
        out2 = self.maxpool2(out2)
        out3_c = self.maxpool3(out3_c)

        out = torch.cat([out5_c, out4, out3_c, out2], dim=1)

        out = F.relu(self.bnout(self.convout(out)), inplace=True)

        out_m = self.gmp(out)
        out_a = self.gap(out)
        out = out_m + out_a

        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out, pred_mask

    def initialize(self):
        for n, m in self.named_children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



if __name__ == '__main__':
    """
    test
    """
    snapshot = r"[Absolute path to the weight pretraind with LANet]"
    net = LASNet(snapshot=snapshot)
    net.cuda()
    net.eval()
    input_ = torch.rand([2, 3, 512,512])
    out, pred_mask = net(input_.cuda())

    print(out.shape)
    print(pred_mask.shape)

    base, head = [], []
    for name, param in net.named_parameters():
        if 'segNet' in name:
            param.requires_grad = False  
            base.append(param)
        else:
            head.append(param)
    print()



