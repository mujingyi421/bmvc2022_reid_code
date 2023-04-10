import os
import sys
import time
import datetime
import random
import copy
import argparse
import os.path as osp
import numpy as np
import math

import os.path as osp
from PIL import Image
import cv2
import data.transforms as T

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from configs.default import get_config
from data import build_dataloader
from models import build_model
from losses import build_losses
from tools.eval_metrics import evaluate, evaluate_with_clothes
from tools.utils import AverageMeter, Logger, save_checkpoint, set_seed

# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image, \
#                                          deprocess_image, \
#                                          preprocess_image


def parse_option():    # 参数设置
    parser = argparse.ArgumentParser(description='Train image-based re-id model')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, help="market1501, cuhk03, dukemtmcreid, msmt17")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU

    if not config.EVAL_MODE:
        sys.stdout = Logger(osp.join(config.OUTPUT, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(config.OUTPUT, 'log_test.txt'))
    print("==========\nConfig:{}\n==========".format(config))
    print("Currently using GPU {}".format(config.GPU))
    # Set random seed
    set_seed(config.SEED)

    # Build dataloader
    trainloader, queryloader, galleryloader, num_classes = build_dataloader(config)
    # Build model
    model, classifier = build_model(config, num_classes) #######################
    # Build classification and pairwise loss
    criterion_cla, criterion_pair, criterion_mse = build_losses(config)
    # criterion_cla, criterion_pair, _ = build_losses(config)
    # Build optimizer
    parameters = list(model.parameters()) + list(classifier.parameters())  #################
    if config.TRAIN.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'adamw':
        optimizer = optim.AdamW(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, 
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
    else:
        raise KeyError("Unknown optimizer: {}".format(config.TRAIN.OPTIMIZER.NAME))
    # Build lr_scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, 
                                         gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)

    start_epoch = config.TRAIN.START_EPOCH
    if config.MODEL.RESUME:
        print("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    model = nn.DataParallel(model).cuda()
    classifier = nn.DataParallel(classifier).cuda()

    if config.EVAL_MODE:
        print("Evaluate only")
        test(model, queryloader, galleryloader)
        # visul(model)
        # heat_nkup(model)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")
    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
        start_train_time = time.time()
        # train_mask_vc(epoch, model, classifier, criterion_cla, criterion_pair, optimizer, trainloader)   ##############
        train_mask_vc1(epoch, model, classifier, criterion_cla, criterion_pair, criterion_mse, optimizer, trainloader)
        train_time += round(time.time() - start_train_time)        
        
        if (epoch+1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 and \
            (epoch+1) % config.TEST.EVAL_STEP == 0 or (epoch+1) == config.TRAIN.MAX_EPOCH:
            print("==> Test")
            rank1 = test_vc(model, queryloader, galleryloader) #######################
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            state_dict = model.module.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(config.OUTPUT, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
        scheduler.step()

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, classifier, clothclassifier, criterion_cla, criterion_pair, criterion_cloth, optimizer, trainloader): #######
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    # batch_cloth_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    # clothclassifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids) in enumerate(trainloader):
        imgs, pids = imgs.cuda(), pids.cuda()
        # clothids = clothids.cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        # cloth_outputs = clothclassifier(features)
        # _, cloth_preds = torch.max(cloth_outputs.data, 1)
        _, preds = torch.max(outputs.data, 1)
        
        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        # cloth_loss = criterion_cloth(cloth_outputs, clothids)   ############
        pair_loss = criterion_pair(features, pids, camids)    #######################

        
        loss = cla_loss + pair_loss
        # Backward + Optimize
        loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        # batch_cloth_loss.update(cloth_loss, clothids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'ClaLoss:{cla_loss.avg:.4f} '
          'PairLoss:{pair_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
           epoch+1, batch_time=batch_time, data_time=data_time, 
           cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, acc=corrects))

def train_mask1(epoch, model, classifier, criterion_cla, criterion_pair, optimizer, trainloader):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    use_cuda = torch.cuda.is_available()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, imgg, pids, cams) in enumerate(trainloader):  #####################
        imgs, imgg, pids = imgs.cuda(), imgg.cuda(), pids.cuda()
        
        b, c, h, w = imgs.shape
        
        img_c = torch.cat([imgs, imgg], dim=0)
        pid_c = torch.cat([pids, pids], dim=0)
        cam_c = torch.cat([cams, cams], dim=0)
        # cloth_c = torch.cat([clothes, clothes], dim=0)


        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward
        features = model(img_c)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)
        # Compute loss

        cla_loss = criterion_cla(outputs, pid_c) 
        pair_loss = criterion_pair(features, pid_c, cam_c) #####################
        loss = cla_loss + pair_loss
        # Backward + Optimize
        loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == pid_c.data).float()/pid_c.size(0), pid_c.size(0))
        batch_cla_loss.update(cla_loss.item(), pid_c.size(0))
        batch_pair_loss.update(pair_loss.item(), pid_c.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'ClaLoss:{cla_loss.avg:.4f} '
          'PairLoss:{pair_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
           epoch+1, batch_time=batch_time, data_time=data_time, 
           cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, acc=corrects))  ##################



def train_mask_vc(epoch, model, classifier, criterion_cla, criterion_pair, optimizer, trainloader):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    use_cuda = torch.cuda.is_available()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, imgg, pids, cams, clothes) in enumerate(trainloader):  #####################
        imgs, imgg, pids = imgs.cuda(), imgg.cuda(), pids.cuda()
        
        b, c, h, w = imgs.shape

        # data_time.update(time.time() - end)
        # print(64/(time.time()-end),"imgs/s")
        # end.update(time.time())
        
        img_c = torch.cat([imgs, imgg], dim=0)
        pid_c = torch.cat([pids, pids], dim=0)
        cam_c = torch.cat([cams, cams], dim=0)
        cloth_c = torch.cat([clothes, clothes], dim=0)


        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward
        features = model(img_c)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)
        # Compute loss

        cla_loss = criterion_cla(outputs, pid_c) 
        pair_loss = criterion_pair(features, pid_c) #####################
        loss = cla_loss + pair_loss
        # Backward + Optimize
        loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == pid_c.data).float()/pid_c.size(0), pid_c.size(0))
        batch_cla_loss.update(cla_loss.item(), pid_c.size(0))
        batch_pair_loss.update(pair_loss.item(), pid_c.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'ClaLoss:{cla_loss.avg:.4f} '
          'PairLoss:{pair_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
           epoch+1, batch_time=batch_time, data_time=data_time, 
           cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, acc=corrects))  ##################

def train_mask_vc1(epoch, model, classifier, criterion_cla, criterion_pair,  criterion_mse, optimizer, trainloader):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_mse_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    use_cuda = torch.cuda.is_available()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, cams, clothes, mask) in enumerate(trainloader):  #####################
        imgs, pids = imgs.cuda(), pids.cuda()
        
        b, c, h, w = imgs.shape
        
        mask = mask.cuda()
        # print(mask.shape)
        mask_i = mask.expand_as(imgs)
        img_a = copy.deepcopy(imgs)

            # upper clothes sampling
        index = np.random.permutation(b)
        img_r = imgs[index]                                  # [64, 3, 256, 128]
        msk_r = mask_i[index]                               # [64, 6, 256, 128]
        img_a[mask_i == 2] = img_r[msk_r == 2]

            # pant sampling
        index = np.random.permutation(b)
        img_r = imgs[index]  # [64, 3, 256, 128]
        msk_r = mask_i[index]  # [64, 6, 256, 128]
        img_a[mask_i == 3] = img_r[msk_r == 3]

        # Measure data loading time
        data_time.update(time.time() - end)
        # print(64/(time.time()-end),"imgs/s")
        # end1 = time.time()

        img_c = torch.cat([imgs, img_a], dim=0)
        pid_c = torch.cat([pids, pids], dim=0)
        cam_c = torch.cat([cams, cams], dim=0)
        # cloth_c = torch.cat([clothes, clothes], dim=0)



        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward
        features = model(img_c)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)
        # Compute loss

        cla_loss = criterion_cla(outputs, pid_c) 
        pair_loss = criterion_pair(features, pid_c) #####################
        mse = criterion_mse(features[:b,:], features[b:,:])
        loss = cla_loss + pair_loss + mse
        # Backward + Optimize
        loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == pid_c.data).float()/pid_c.size(0), pid_c.size(0))
        batch_cla_loss.update(cla_loss.item(), pid_c.size(0))
        batch_pair_loss.update(pair_loss.item(), pid_c.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'ClaLoss:{cla_loss.avg:.4f} '
          'PairLoss:{pair_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
           epoch+1, batch_time=batch_time, data_time=data_time, 
           cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, acc=corrects))  ##################

def train_mask_prcc1(epoch, model, classifier, criterion_cla, criterion_pair,  criterion_mse, optimizer, trainloader):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_mse_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    use_cuda = torch.cuda.is_available()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, mask, pids, cams) in enumerate(trainloader):  #####################
        imgs, pids = imgs.cuda(), pids.cuda()
        
        b, c, h, w = imgs.shape
        mask = mask.cuda()
        # print(mask.shape)
        mask_i = mask.expand_as(imgs)
        img_a = copy.deepcopy(imgs)

            # upper clothes sampling
        index = np.random.permutation(b)
        img_r = imgs[index]                                  # [64, 3, 256, 128]
        msk_r = mask_i[index]                               # [64, 6, 256, 128]
        img_a[mask_i == 2] = img_r[msk_r == 2]

            # pant sampling
        index = np.random.permutation(b)
        img_r = imgs[index]  # [64, 3, 256, 128]
        msk_r = mask_i[index]  # [64, 6, 256, 128]
        img_a[mask_i == 3] = img_r[msk_r == 3]

        
        img_c = torch.cat([imgs, img_a], dim=0)
        pid_c = torch.cat([pids, pids], dim=0)
        cam_c = torch.cat([cams, cams], dim=0)
        # cloth_c = torch.cat([clothes, clothes], dim=0)


        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward
        features = model(img_c)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)
        # Compute loss

        cla_loss = criterion_cla(outputs, pid_c) 
        pair_loss = criterion_pair(features, pid_c) #####################
        mse = criterion_mse(features[:b,:], features[b:,:])
        loss = cla_loss + pair_loss + mse
        # Backward + Optimize
        loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == pid_c.data).float()/pid_c.size(0), pid_c.size(0))
        batch_cla_loss.update(cla_loss.item(), pid_c.size(0))
        batch_pair_loss.update(pair_loss.item(), pid_c.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'ClaLoss:{cla_loss.avg:.4f} '
          'PairLoss:{pair_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
           epoch+1, batch_time=batch_time, data_time=data_time, 
           cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, acc=corrects))  ##################


def train_mask_pro(epoch, model, classifier, criterion_cla, criterion_pair, optimizer, trainloader):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    use_cuda = torch.cuda.is_available()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, imgg, pids, cams, imggg) in enumerate(trainloader):  #####################
        imgs, imgg, pids = imgs.cuda(), imgg.cuda(), pids.cuda()
        imggg = imggg.cuda()
        b, c, h, w = imgs.shape
        
        img_c = torch.cat([imgs, imgg], dim=0)
        pid_c = torch.cat([pids, pids], dim=0)
        cam_c = torch.cat([cams, cams], dim=0)

        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward
        features = model(img_c, imggg)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)
        # Compute loss

        cla_loss = criterion_cla(outputs, pid_c) 
        pair_loss = criterion_pair(features, pid_c, cam_c) #####################
        loss = cla_loss + pair_loss
        # Backward + Optimize
        loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == pid_c.data).float()/pid_c.size(0), pid_c.size(0))
        batch_cla_loss.update(cla_loss.item(), pid_c.size(0))
        batch_pair_loss.update(pair_loss.item(), pid_c.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'ClaLoss:{cla_loss.avg:.4f} '
          'PairLoss:{pair_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
           epoch+1, batch_time=batch_time, data_time=data_time, 
           cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, acc=corrects))  ##################
           
def train_mask_vc2(epoch, model, classifier, criterion_cla, criterion_pair, optimizer, trainloader):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    use_cuda = torch.cuda.is_available()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, imgg, pids, cams, clothes) in enumerate(trainloader):  #####################
        imgs, imgg, pids = imgs.cuda(), imgg.cuda(), pids.cuda()
        
        b, c, h, w = imgs.shape
        
        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)
        # Compute loss

        cla_loss = criterion_cla(outputs, pids) 
        pair_loss = criterion_pair(features, pids, clothes) #####################
        loss = cla_loss + pair_loss
        # Backward + Optimize
        loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'ClaLoss:{cla_loss.avg:.4f} '
          'PairLoss:{pair_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
           epoch+1, batch_time=batch_time, data_time=data_time, 
           cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, acc=corrects))  ##################

def train_mask_2(epoch, model, classifier, criterion_cla, criterion_pair, optimizer, trainloader):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    use_cuda = torch.cuda.is_available()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, imgg, pids, cams) in enumerate(trainloader):  #####################
        imgs, imgg, pids = imgs.cuda(), imgg.cuda(), pids.cuda()
        
        b, c, h, w = imgs.shape
        
        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)
        # Compute loss

        cla_loss = criterion_cla(outputs, pids) 
        pair_loss = criterion_pair(features, pids) #####################
        loss = cla_loss + pair_loss
        # Backward + Optimize
        loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'ClaLoss:{cla_loss.avg:.4f} '
          'PairLoss:{pair_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
           epoch+1, batch_time=batch_time, data_time=data_time, 
           cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, acc=corrects))  ##################


def train_mask(epoch, model, classifier, criterion_cla, criterion_pair, optimizer, trainloader):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    use_cuda = torch.cuda.is_available()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, msks, pids, cams) in enumerate(trainloader):
        imgs, msks, pids = imgs.cuda(), msks.cuda(), pids.cuda()
        b, c, h, w = imgs.shape
        
        msks = msks.squeeze(dim=1)
        mask_o = msks.argmax(dim=1).unsqueeze(dim=1)
        mask_w = mask_o.squeeze(dim=1)
        mask_i = mask_o.expand_as(imgs)
        img_a = copy.deepcopy(imgs)

        for i in range(len(img_a)):
            for chanel in img_a[i]:
                e = chanel[mask_w[i]==2].mean()
                v = chanel[mask_w[i]==2].std()
                chanel[mask_w[i]==2] = (chanel[mask_w[i]==2] - e) / v
                for x in chanel[mask_w[i]==2]:
                    if x<-1:
                        x=-1
                    if x>1:
                        x=1
                    
        for i in range(len(img_a)):
            for chanel in img_a[i]:
                e = chanel[mask_w[i]==3].mean()
                v = chanel[mask_w[i]==3].std()
                chanel[mask_w[i]==3] = (chanel[mask_w[i]==3] - e) / v


        # img_a[mask_i==2] = 0
        # img_a[mask_i==3] = 0

        img_c = torch.cat([imgs, img_a], dim=0)
        pid_c = torch.cat([pids, pids], dim=0)
        cam_c = torch.cat([cams, cams], dim=0)

        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward
        features= model(img_c)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)
        # Compute loss
        cla_loss = criterion_cla(outputs, pid_c) 
        pair_loss = criterion_pair(features, pid_c, cam_c) #####################
        loss = cla_loss + pair_loss    
        # Backward + Optimize
        loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == pid_c.data).float()/pid_c.size(0), pid_c.size(0))
        batch_cla_loss.update(cla_loss.item(), pid_c.size(0))
        batch_pair_loss.update(pair_loss.item(), pid_c.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'ClaLoss:{cla_loss.avg:.4f} '
          'PairLoss:{pair_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
           epoch+1, batch_time=batch_time, data_time=data_time, 
           cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, acc=corrects))  ##################

def train_mask_cloth(epoch, model, classifier, clothclassifier, criterion_cla, criterion_pair, criterion_cloth, optimizer, trainloader):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_cloth_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    use_cuda = torch.cuda.is_available()

    model.train()
    classifier.train()
    clothclassifier.train()

    end = time.time()
    for batch_idx, (imgs, msks, pids, cams, clothids) in enumerate(trainloader):
        imgs, msks, pids = imgs.cuda(), msks.cuda(), pids.cuda()
        clothids = clothids.cuda()
        b, c, h, w = imgs.shape
        
        msks = msks.squeeze(dim=1)
        mask_i = msks.argmax(dim=1).unsqueeze(dim=1)
        mask_i = mask_i.expand_as(imgs)
        img_a = copy.deepcopy(imgs)

        img_a[mask_i==2] = 0
        img_a[mask_i==3] = 0

        img_c = torch.cat([imgs, img_a], dim=0)
        pid_c = torch.cat([pids, pids], dim=0)
        cam_c = torch.cat([cams, cams], dim=0)

        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward
        features = model(img_c)
        features_1 = model(imgs)
        outputs = classifier(features)
        cloth_outputs = clothclassifier(features_1)
        _, preds = torch.max(outputs.data, 1)
        # Compute loss
        cla_loss = criterion_cla(outputs, pid_c) 
        cloth_loss = criterion_cloth(cloth_outputs, clothids)
        pair_loss = criterion_pair(features, pid_c, cam_c)
        gap = math.exp(-(cloth_loss - cla_loss)) 
        
        loss = cla_loss + pair_loss + gap     
        # Backward + Optimize
        loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == pid_c.data).float()/pid_c.size(0), pid_c.size(0))
        batch_cla_loss.update(cla_loss.item(), pid_c.size(0))
        batch_pair_loss.update(pair_loss.item(), pid_c.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'ClaLoss:{cla_loss.avg:.4f} '
          'PairLoss:{pair_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
           epoch+1, batch_time=batch_time, data_time=data_time, 
           cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, acc=corrects))


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)

    return img_flip


@torch.no_grad()
def extract_feature(model, dataloader):
    features, pids, camids = [], [], []
    for batch_idx, (imgs, imggs, batch_pids, batch_camids) in enumerate(dataloader): ############################
        flip_imgs = fliplr(imgs)
        flip_imggs = fliplr(imggs) ######
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        imggs, flip_imggs = imggs.cuda(), flip_imggs.cuda() #############
        
        batch_features = model(imgs).data.cpu()
        batch_features_flip = model(flip_imgs).data.cpu()
        batch_features += batch_features_flip

        features.append(batch_features)
        pids.append(batch_pids)
        camids.append(batch_camids)
    features = torch.cat(features, 0)
    pids = torch.cat(pids, 0).numpy()
    camids = torch.cat(camids, 0).numpy()

    return features, pids, camids

def extract_feature_ltcc(model, dataloader):
    features, pids, camids, clothids = [], [], [], []
    for batch_idx, (imgs, _, batch_pids, batch_camids, batch_clothids) in enumerate(dataloader): ############################
        flip_imgs = fliplr(imgs)
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        batch_features = model(imgs).data.cpu()
        batch_features_flip = model(flip_imgs).data.cpu()
        batch_features += batch_features_flip

        features.append(batch_features)
        pids.append(batch_pids)
        camids.append(batch_camids)
        clothids.append(batch_clothids)
    features = torch.cat(features, 0)
    pids = torch.cat(pids, 0).numpy()
    camids = torch.cat(camids, 0).numpy()
    clothids = torch.cat(clothids, 0).numpy()

    return features, pids, camids, clothids

def extract_feature_p(model, dataloader):
    features, pids, camids, clothids = [], [], [], []
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothids, _) in enumerate(dataloader): ############################
        flip_imgs = fliplr(imgs)
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        batch_features = model(imgs).data.cpu()
        batch_features_flip = model(flip_imgs).data.cpu()
        batch_features += batch_features_flip

        features.append(batch_features)
        pids.append(batch_pids)
        camids.append(batch_camids)
        clothids.append(batch_clothids)
    features = torch.cat(features, 0)
    pids = torch.cat(pids, 0).numpy()
    camids = torch.cat(camids, 0).numpy()
    clothids = torch.cat(clothids, 0).numpy()

    return features, pids, camids, clothids

def extract_feature_pro(model, dataloader):
    features, pids, camids, masks = [], [], [], []
    for batch_idx, (imgs, _, batch_pids, batch_camids, masks) in enumerate(dataloader): ############################
        flip_imgs = fliplr(imgs)
        flip_masks = fliplr(masks)
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        masks, flip_masks = masks.cuda(), flip_masks.cuda()

        batch_features = model(imgs, masks).data.cpu()
        batch_features_flip = model(flip_imgs, flip_masks).data.cpu()
        batch_features += batch_features_flip

        features.append(batch_features)
        pids.append(batch_pids)
        camids.append(batch_camids)
        
    features = torch.cat(features, 0)
    pids = torch.cat(pids, 0).numpy()
    camids = torch.cat(camids, 0).numpy()

    return features, pids, camids

def visul(model):
    model.eval()
    persons = os.listdir('../data/prcc/rgb/val')
    for pid_s in persons:
        path_p = os.path.join('../data/prcc/rgb/val', pid_s)
        files = os.listdir(path_p)
        path = os.path.join('./mask_att_2', pid_s)
        # if os.path.exists(path) == True:
        #     continue
        os.mkdir(path)
        for file in files:
            if file[0] == 'T':
                    continue
            img_path = os.path.join('../data/prcc/rgb/val', pid_s, file)
            img = Image.open(img_path).convert('RGB')
            imgcv =  cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    # img = Image.open('../data/prcc/rgb/val/110/A_cropped_rgb001.jpg').convert('RGB')
    # img = Image.open('../data/prcc/rgb/val/122/A_cropped_rgb001.jpg').convert('RGB')
    # img = Image.open('../data/prcc/rgb/val/158/A_cropped_rgb001.jpg').convert('RGB')
    # img = Image.open('../data/prcc/rgb/val/240/B_cropped_rgb004.jpg').convert('RGB')
    # img = Image.open('../data/prcc/rgb/val/250/C_cropped_rgb064.jpg').convert('RGB')
    # img = Image.open('../data/prcc/rgb/val/234/C_cropped_rgb091.jpg').convert('RGB')
    # imgcv =  cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR) 
    
            trans = T.Compose([
                T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
            imgg=trans(img)
            imgg=imgg.unsqueeze(0)
            imgg = model(imgg).data.cpu()
    
            for i in range(1):
                mask = cv2.imread('./vis/vis_{}.jpg'.format(i))
                mask = cv2.resize(mask, (imgcv.shape[1], imgcv.shape[0]))
                
                cam = 0.5 * mask + (1 - 0.5) * np.float32(imgcv)  # [256, 128, 3]
                cam = cam / np.max(cam)
                cam = np.uint8(255 * cam)  
                cam = np.clip(cam,0,255)  
            save_path = os.path.join('./mask_att_2', pid_s, file)
            cv2.imwrite(save_path, mask)

            # cv2.imwrite("./vis/heatmap_{}.jpg".format(i),cam)  
    # cv2.imwrite("./vis/origin.jpg",imgcv)    
    return

def heat(model):
    target_layer = [model.base]

    persons = os.listdir('../data/prcc/rgb/val')
    for pid_s in persons:
        path_p = os.path.join('../data/prcc/rgb/val', pid_s)
        files = os.listdir(path_p)
        path = os.path.join('./prcc_hm_base', pid_s)
        # if os.path.exists(path) == True:
        #     continue
        # if path != './vis4/111':
        #     continue
        os.mkdir(path)
        for file in files:
            if file[0] == 'T':
                    continue
            img_path = os.path.join('../data/prcc/rgb/val', pid_s, file)
            rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            cam = GradCAM(model=model, target_layers=target_layer, use_cuda=False)
            target_category = None
            grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
            grayscale_cam = grayscale_cam[0]
            visualization = show_cam_on_image(rgb_img, grayscale_cam)  # (224, 224, 3)
            # name = file.split('.')[0]
            save_path = os.path.join('./prcc_hm_base', pid_s, file)
            cv2.imwrite(save_path, visualization)

def heat_vc(model):
    target_layer = [model.base]

    persons = os.listdir('../data/vc_cloth/query')
    for file in persons:
        if file[0] == 'T':
                continue
        img_path = os.path.join('../data/vc_cloth/query', file)
        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        cam = GradCAM(model=model, target_layers=target_layer, use_cuda=False)
        target_category = None
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam)  # (224, 224, 3)
        # name = file.split('.')[0]
        save_path = os.path.join('./vc_hm_base', file)
        cv2.imwrite(save_path, visualization)

def heat_nkup(model):
    target_layer = [model.base2]

    persons = os.listdir('../data/nkup/bounding_box_train')
    for file in persons:
        if file[0] == 'T':
                continue
        img_path = os.path.join('../data/nkup/bounding_box_train', file)
        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        cam = GradCAM(model=model, target_layers=target_layer, use_cuda=False)
        target_category = None
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam)  # (224, 224, 3)
        # name = file.split('.')[0]
        save_path = os.path.join('./nkup_hm2', file)
        cv2.imwrite(save_path, visualization)



def test(model, queryloader, galleryloader):
    since = time.time()
    model.eval()
    # Extract features for query set
    qf, q_pids, q_camids = extract_feature(model, queryloader) ##########################
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))
    # Extract features for gallery set
    gf, g_pids, g_camids = extract_feature(model, galleryloader) #############################
    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))
    if config.TEST.DISTANCE == 'euclidean':
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        for i in range(m):
            distmat[i:i+1].addmm_(1, -2, qf[i:i+1], gf.t())
    else:
        # Cosine similarity
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)
        for i in range(m):
            distmat[i] = - torch.mm(qf[i:i+1], gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], mAP))
    print("------------------------------------------------")

    return cmc[0]

def test_ltcc(model, queryloader, galleryloader):
    since = time.time()
    model.eval()
    # Extract features for query set
    qf, q_pids, q_camids, q_clothids = extract_feature_ltcc(model, queryloader)
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))
    # Extract features for gallery set
    gf, g_pids, g_camids, g_clothids = extract_feature_ltcc(model, galleryloader)
    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))
    if config.TEST.DISTANCE == 'euclidean':
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        for i in range(m):
            distmat[i:i+1].addmm_(1, -2, qf[i:i+1], gf.t())
    else:
        # Cosine similarity
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)
        for i in range(m):
            distmat[i] = - torch.mm(qf[i:i+1], gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids, mode='CC')

    print("Results ----------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], mAP))
    print("------------------------------------------------")

    return cmc[0]

def test_vc(model, queryloader, galleryloader):
    since = time.time()
    model.eval()
    # Extract features for query set
    qf, q_pids, q_camids, q_clothids = extract_feature_p(model, queryloader)
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))
    # Extract features for gallery set
    gf, g_pids, g_camids, g_clothids = extract_feature_p(model, galleryloader)
    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))
    if config.TEST.DISTANCE == 'euclidean':
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        for i in range(m):
            distmat[i:i+1].addmm_(1, -2, qf[i:i+1], gf.t())
    else:
        # Cosine similarity
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)
        for i in range(m):
            distmat[i] = - torch.mm(qf[i:i+1], gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], mAP))
    print("------------------------------------------------")

    return cmc[0]




if __name__ == '__main__':
    config = parse_option()
    main(config)