import torch
from torch import nn

from losses.cross_entropy_label_smooth import CrossEntropyLabelSmooth
from losses.triplet_loss import TripletLoss
from losses.contrastive_loss import ContrastiveLoss
from losses.arcface_loss import ArcFaceLoss
from losses.cosface_loss import CosFaceLoss, PairwiseCosFaceLoss
from losses.circle_loss import CircleLoss, PairwiseCircleLoss
from losses.triplet_loss_prcc import TripletLossPRCC
from losses.WTL import WTL
from losses.L1_cross_entropy import l1
from losses.triplet_loss_cloth import TripletLossCloth
from losses.featloss import FeatLoss

def build_losses(config):
    # Build classification loss
    if config.LOSS.CLA_LOSS == 'crossentropy':
        criterion_cla = nn.CrossEntropyLoss()
        # criterion_cla = l1()
    elif config.LOSS.CLA_LOSS == 'crossentropylabelsmooth':
        criterion_cla = CrossEntropyLabelSmooth()
    elif config.LOSS.CLA_LOSS == 'arcface':
        criterion_cla = ArcFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'cosface':
        criterion_cla = CosFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'circle':
        criterion_cla = CircleLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    else:
        raise KeyError("Invalid classification loss: '{}'".format(config.LOSS.CLA_LOSS))

    # Build pairwise loss
    if config.LOSS.PAIR_LOSS == 'triplet':
        criterion_pair = TripletLoss(margin=config.LOSS.PAIR_M, distance=config.TEST.DISTANCE)
    elif config.LOSS.PAIR_LOSS == 'contrastive':
        criterion_pair = ContrastiveLoss(scale=config.LOSS.PAIR_S)
    elif config.LOSS.PAIR_LOSS == 'cosface':
        criterion_pair = PairwiseCosFaceLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'circle':
        criterion_pair = PairwiseCircleLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'triplet_prcc':
        criterion_pair = TripletLossPRCC(margin=config.LOSS.PAIR_M, distance=config.TEST.DISTANCE)
    elif config.LOSS.PAIR_LOSS == 'triplet_cloth':
        criterion_pair = TripletLossCloth(margin=config.LOSS.PAIR_M, distance=config.TEST.DISTANCE)
    elif config.LOSS.PAIR_LOSS == 'WTL':
        criterion_pair = WTL(margin=config.LOSS.PAIR_M, distance=config.TEST.DISTANCE)
    else:
        raise KeyError("Invalid pairwise loss: '{}'".format(config.LOSS.PAIR_LOSS))

    criterion_mse =  torch.nn.MSELoss(reduction='mean')

    return criterion_cla, criterion_pair, criterion_mse


def DeepSupervision(criterion, xs, y):
    """
    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    # loss /= len(xs)
    return loss

