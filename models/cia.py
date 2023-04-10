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


class ResNet50_cia(nn.Module):
    def __init__(self, res4_stride=1, **kwargs):
        super().__init__()
            
        resnet50 = torchvision.models.resnet50(pretrained=True)
        if res4_stride == 1: 
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1) 
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layer2 = nn.Sequential(*list(resnet50.children())[:6])

        self.base2 = nn.Sequential(*list(resnet50.children())[6:-2])
        self.bn = nn.BatchNorm1d(2048)
        self.bn1 = nn.BatchNorm2d(512)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        
    def forward(self, xx):

        x = xx[:,:3,:,:] #[b,3,256,128]
        
        x = self.layer2(x)  #[b,512,32,16]
        b1, c1, h1, w1 = x.shape #[b,512,32,16]

        g_x1 = x.view(b1, c1, -1) #[B, c, h*w]
        theta_x1 = g_x1 #[B, c, h*w]
        phi_x1 = g_x1.permute(0, 2, 1) #[B, h*w, c]
        f1 = torch.matmul(theta_x1, phi_x1) #[B, c, c]
        f1 = F.softmax(f1, dim=-1)
        y1 = torch.matmul(f1, g_x1)
        y1 = y1.view(b1, c1, *x.size()[2:])
        y1 = self.bn1(y1)
        w = y1 + x

        w = self.base2(w)
        w = F.avg_pool2d(w, w.size()[2:])
        w = w.view(w.size(0), -1)
        f = self.bn(w)

        return f