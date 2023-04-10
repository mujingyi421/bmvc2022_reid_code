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

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

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
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)


class ResNet50_eca(nn.Module):
    def __init__(self, res4_stride=1, **kwargs):
        super().__init__()
            
        resnet50 = torchvision.models.resnet50(pretrained=True)
        if res4_stride == 1: 
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1) 
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layer2 = nn.Sequential(*list(resnet50.children())[:6])
        self.eca = ECAAttention(kernel_size=3)

        self.base2 = nn.Sequential(*list(resnet50.children())[6:-2])
        self.bn = nn.BatchNorm1d(2048)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        
    def forward(self, xx):

        x = xx[:,:3,:,:] #[b,3,256,128]
        
        x = self.layer2(x)  #[b,512,32,16]
        b, c, h, w = x.shape #[b,512,32,16]
        
        w = self.eca(x)

        w = self.base2(w)
        w = F.avg_pool2d(w, w.size()[2:])
        w = w.view(w.size(0), -1)
        f = self.bn(w)

        return f