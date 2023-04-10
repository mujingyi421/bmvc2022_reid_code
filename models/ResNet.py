import torchvision
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import random
from models import pooling
import copy
import math

class ConvBlock(nn.Module):
    """Basic convolutional block"""
    def __init__(self, in_c, out_c, k=1, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)

    def forward(self, x):
        return self.conv(x)

class ResNet50(nn.Module):
    def __init__(self, res4_stride=1, **kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if res4_stride == 1:
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1) 

        BN = nn.BatchNorm2d
        self.pre = nn.Sequential(*list(resnet50.children())[:3])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.base1 = nn.Sequential(*list(resnet50.children())[4:-2])
        
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        # self.a = nn.Parameter(torch.ones(1))
       
        # self.step1 = nn.Sequential(list(resnet50.children())[0])
        # self.in1 = nn.InstanceNorm2d(64, affine = True)
        # self.step2 = nn.Sequential(*list(resnet50.children())[1:-2])

        # self.layer2 = nn.Sequential(*list(resnet50.children())[:6])
        # self.layer3 = nn.Sequential(*list(resnet50.children())[6])
        # self.base2 = nn.Sequential(*list(resnet50.children())[7])
        # self.W1 = BN(512)
        # self.W2 = BN(1024)

        # init.constant_(self.W1.weight.data, 0.0)
        # init.constant_(self.W1.bias.data, 0.0)
        # init.constant_(self.W2.weight.data, 0.0)
        # init.constant_(self.W2.bias.data, 0.0)


        self.conv_1 = ConvBlock(2048, 896, 1, 1, 0)
        self.conv_2 = ConvBlock(2048, 128, 1, 1, 0)
        self.conv_3 = ConvBlock(2048, 128, 1, 1, 0)
        self.conv_4 = ConvBlock(2048, 896, 1, 1, 0)
        self.conv_5 = ConvBlock(4096, 2048, 1, 1, 0)

        self.bn = nn.BatchNorm1d(2048)   ######################
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)


    def weights(self, x, w):

        B, L = x.size()

        noise = torch.rand(B, L, device=x.device)
        shuffled = torch.argsort(noise, dim=1)
        len_keep = w
        keeps = shuffled[:, :len_keep]
        keeps , ins = torch.sort(keeps, dim=1)
            
        par = torch.gather(x, dim=1, index = keeps)
        return par

        
    def forward(self, x):

        # input: [64, 3, 256, 128]

        x = self.base(x)  #[64, 2048, 16, 8]
        #--------------------------------------------------------
        # x = self.step1(x)
        # x = self.in1(x)
        # x = self.step2(x)
        #---------------------------------------------------------
        # x = x[:,:3,:,:]
        # masked = x[:,-1,:,:].unsqueeze(1)
        # n = torch.mean(up.float(),dim=0)
        # n = int(torch.ceil(n).item())

        # x=self.layer2(x)
        # b, c, h, w = x.shape
        # up = h // n
        
        # mask  = torch.ones(b, 1, up, w).to(x.device)
        # # mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='bilinear', align_corners=True)
        # mask = mask * self.a
        
        # mask1 = torch.ones(b, 1, h-up, w).to(x.device)
        # mask2 = torch.cat([mask, mask1], dim=2)
        # mask2 = F.softmax(mask2, dim=1)
        # # mask2 = torch.sigmoid(mask2)
        # mask2= mask2.expand_as(x)
        # m = x * mask2
        # x = x + m

        # x=self.layer3(x)
        # x=self.base2(x)
        #-----------------------------------------
        # x = self.layer2(x)
        # b1, c1, w1, h1 = x.shape
        # g_x1 = x.view(b1, c1, -1) #[B, c, h*w]
        # theta_x1 = g_x1 #[B, c, h*w]
        # phi_x1 = g_x1.permute(0, 2, 1) #[B, h*w, c]
        # f1 = torch.matmul(theta_x1, phi_x1) #[B, c, c]
        # f1 = F.softmax(f1, dim=-1)
        # y1 = torch.matmul(f1, g_x1)
        # y1 = y1.view(b1, c1, *x.size()[2:])
        # y1 = self.W1(y1)
        # z = y1 + x
        
        # w = self.layer3(z)
        # b2, c2, w2, h2 = z.shape
        # g_x2 = z.view(b2, c2, -1) #[B, c, h*w]
        # theta_x2 = g_x2 #[B, c, h*w]
        # phi_x2 = g_x2.permute(0, 2, 1) #[B, h*w, c]
        # f2 = torch.matmul(theta_x2, phi_x2) #[B, c, c]
        # f2 = F.softmax(f2, dim=-1)
        # y2 = torch.matmul(f2, g_x2)
        # y2 = y2.view(b2, c2, *z.size()[2:])
        # y2 = self.W2(y2)
        # w = y2 + z
        
        # w = self.base2(w)
        # w = F.avg_pool2d(w, w.size()[2:])
        # w = w.view(w.size(0), -1)
        # f = self.bn(w)   #[128, 2048]
        #--------------------------------------------------------
        
        # part1 = x[:, :, :2, :]
        # part2 = x[:, :, 2:8, :]
        # part3 = x[:, :, 8:12, :]
        # part4 = x[:, :, 12:, :]

        

        # part1 = F.max_pool2d(part1, part1.size()[2:])
        # part2 = F.max_pool2d(part2, part2.size()[2:])
        # part3 = F.max_pool2d(part3, part3.size()[2:])
        # part4 = F.max_pool2d(part4, part4.size()[2:])


        # part1 = part1.view(part1.size(0), -1)
        # part2 = part2.view(part2.size(0), -1)
        # part3 = part3.view(part3.size(0), -1)
        # part4 = part4.view(part4.size(0), -1)

        # part1 = self.conv_1(part1.unsqueeze(2).unsqueeze(3))
        # part2 = self.conv_2(part2.unsqueeze(2).unsqueeze(3))
        # part3 = self.conv_3(part3.unsqueeze(2).unsqueeze(3))
        # part4 = self.conv_4(part4.unsqueeze(2).unsqueeze(3))

        # part1 = part1.view(part1.size(0), -1)
        # part2 = part2.view(part2.size(0), -1)
        # part3 = part3.view(part3.size(0), -1)
        # part4 = part4.view(part4.size(0), -1)

        #-------------------------------------------------------
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        f = self.bn(x)   #[64, 2048]
        
        # xc = torch.cat((x,parthead), 1)
        # xc = xc.view(xc.size(0), -1)
        # f= self.bn(xc)
        
        return f
