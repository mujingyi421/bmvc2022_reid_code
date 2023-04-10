
import torchvision
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import copy

class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNet50_se(nn.Module):
    def __init__(self, res4_stride=1, **kwargs):
        super().__init__()
            
        resnet50 = torchvision.models.resnet50(pretrained=True)
        if res4_stride == 1: 
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1) 
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layer2 = nn.Sequential(*list(resnet50.children())[:6])
        self.se = SEAttention(channel=512,reduction=8)

        self.base2 = nn.Sequential(*list(resnet50.children())[6:-2])
        self.bn = nn.BatchNorm1d(2048)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        
    def forward(self, xx):

        x = xx[:,:3,:,:] #[b,3,256,128]
        
        x = self.layer2(x)  #[b,512,32,16]
        b, c, h, w = x.shape #[b,512,32,16]
        
        w = self.se(x)

        w = self.base2(w)
        w = F.avg_pool2d(w, w.size()[2:])
        w = w.view(w.size(0), -1)
        f = self.bn(w)

        return f