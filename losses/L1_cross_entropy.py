import math
import torch
import torch.nn.functional as F
from torch import nn


class l1(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self):
        super().__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, x_ba):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """

        
        _, num_classes = inputs.size() #[128,150]
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda() #[128,150]
        
        x_ba = torch.abs(x_ba) #[128,4,32,16]
        x_ba1 = torch.sum(x_ba, dim = 1) #[128,32,16]
        x_ba2 = torch.sum(x_ba1, dim = 1) #[128,16]
        x_ba3 = torch.sum(x_ba2, dim = 1) #[128, ]
        
        loss =  (- targets * log_probs).mean(0).sum() + torch.sum(x_ba3, dim = 0)
        
        return loss