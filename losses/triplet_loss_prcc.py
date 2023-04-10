import math
import torch
import torch.nn.functional as F
from torch import nn


class TripletLossPRCC(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, distance='euclidean'):
        super(TripletLossPRCC, self).__init__()
        if distance not in ['euclidean', 'cosine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

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
        
        dist_ap, dist_an = [], []
        for i in range(n):
            # if 6 not in cam_mask[i] and 3 not in cam_mask[i]:
            #     dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))   # hard positive(可以理解为不同衣服的同样的人)
            #     dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))   #negative 可以理解为相似衣服的不同的人
            for j in range(len(cam_mask[i])):
                if cam_mask[i][j] == 6 or cam_mask[i][j] == 3:
                   dist_ap.append(dist[i][j].unsqueeze(0))
                   dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)  # L = MAX(0, -y * (x1 - x2) + margin)

        return loss