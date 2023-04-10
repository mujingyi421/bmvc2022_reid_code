import torch
from torch import nn
from torch.nn import MSELoss
import numpy as np
import torch.nn.functional as F


class FeatLoss(nn.Module):
    def __init__(self, ):
        super(FeatLoss, self).__init__()

    def forward(self, feat1, feat2):    # [64, 2048], [64, 2048]
        B, C = feat1.shape

        dist = torch.pow(torch.abs(feat1 - feat2), 2).sum(dim=-1)

        loss = (1. / (1. + torch.exp(-dist))).mean()

        # loss = dist.mean()

        return loss