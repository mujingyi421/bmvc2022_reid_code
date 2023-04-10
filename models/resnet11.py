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

savepath=r'./vis/'
if not os.path.exists(savepath):
    os.mkdir(savepath)


def draw_features(x):
    tic=time.time()
    for i in range(1):

        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255
    
        img=img.astype(np.uint8)  #转成unit8
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map
        # img = img[:, :, ::-1]#注意cv2（BGR）和matplotlib(RGB)通道是相反的

        cv2.imwrite("{}/vis_{}.jpg".format(savepath,i),img)
        # print("{}/{}".format(i,1))

    # print("time:{}".format(time.time()-tic))

class ResNet50_11(nn.Module):
    def __init__(self, res4_stride=1, **kwargs):
        super().__init__()
            
        resnet50 = torchvision.models.resnet50(pretrained=True)
        if res4_stride == 1: 
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1) 
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

        self.layer2 = nn.Sequential(*list(resnet50.children())[:6])

        self.base2 = nn.Sequential(*list(resnet50.children())[6:-2])
        self.conv_1 = nn.Conv2d(512, 2, kernel_size=7, padding=3, bias=False)
        self.conv_2 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.conv_3 = nn.Conv2d(3, 1, kernel_size=7, padding=3, bias=False)
        # self.conv_2 = nn.Conv2d(512, 1, kernel_size=7, padding=3, bias=False)
        # self.conv_3 = nn.Conv2d(512, 1, kernel_size=7, padding=3, bias=False)
        # self.conv_4 = nn.Conv2d(512, 1, kernel_size=7, padding=3, bias=False)

        self.bn = nn.BatchNorm1d(2048)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        
    def forward(self, xx):

        x = xx[:,:3,:,:] #[b,3,256,128]
        m = xx[:,-1,:,:].unsqueeze(1) #[b,1,256,128]
        
        x = self.layer2(x)  #[b,512,32,16]
        b, c, h, w = x.shape #[b,512,32,16]
        m = torch.nn.functional.interpolate(m, size=(h, w), mode='bilinear', align_corners=True)

        mask1 = self.conv_2(m) #[b,1,32,16]
        mask2 = self.conv_1(x) #[b,3,32,16]
        mask11 = torch.sigmoid(mask1)
        mask = torch.cat([mask2,mask1],dim=1) #[b,4,32,16]
        y0 = self.conv_3(mask) #[b,1,32,16]

        # draw_features(y0.cpu().detach().numpy())
        
        y = y0 * mask1
        y = torch.sigmoid(y) 
        
        
        w = x * y
        w = self.base2(w)
        w = F.avg_pool2d(w, w.size()[2:])
        w = w.view(w.size(0), -1)
        f = self.bn(w)

        return f