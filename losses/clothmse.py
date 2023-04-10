import math
import torch
import torch.nn.functional as F
from torch import nn


class clothmse(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, distance='euclidean'):
        super(clothmse, self).__init__()
        if distance not in ['euclidean', 'cosine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance


    def forward(self, inputs, targets, cams):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)  #[64,2048]

        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'euclidean':
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(1, -2, inputs, inputs.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability     [n, n] n个样本之间的距离矩阵
        elif self.distance == 'cosine':
            inputs = F.normalize(inputs, p=2, dim=1)
            dist = - torch.mm(inputs, inputs.t())

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_1 = targets.expand(n, n).eq(targets.expand(n, n).t()).float()
        cam_mask = cams.expand(n, n) * cams.expand(n, n).t().float()
        device = torch.device('cuda:0')
        cam_mask = cam_mask.to(device)
        
        cam_mask = cam_mask * mask_1
        ti = torch.triu(torch.ones(n,n),diagonal=1)
        ti = ti.to(device)
        cam_mask = cam_mask * ti

        loss1 = dist[cam_mask==3].sum()
        loss2 = dist[cam_mask==6].sum()
        loss = loss1+loss2
        

        return loss