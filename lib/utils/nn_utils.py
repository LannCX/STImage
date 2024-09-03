import os
import gc
import time
import math
# import pynvml
import inspect
import datetime
import numpy as np
from collections import namedtuple


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.radam import RAdam




def save_checkpoint(checkpoint_file, net, epoch, optim, gs, is_parallel=True):
    checkpoint_dict = {
        'epoch': epoch,
        'global_step': gs,
        'optimizer': optim.state_dict(),
        'state_dict': net.module.state_dict() if is_parallel else net.state_dict()
    }
    torch.save(checkpoint_dict, checkpoint_file)


def load_checkpoint(checkpoint_file, is_parallel=True):
    checkpoint = torch.load(checkpoint_file)
    if is_parallel:
        w_dict = checkpoint['state_dict']
        w_dict = {'module.' + k: v for k, v in w_dict.items()}
    else:
        w_dict = checkpoint['state_dict']
        # w_dict = {k.replace('module.',''):v for k,v in mdl.items()}
    return w_dict, checkpoint


def get_optimizer(cfg, model, policies=None):
    if policies is None:
        if cfg.TRAIN.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(
                [{'params': filter(lambda p: p.requires_grad, model.parameters()),
                  'initial_lr': cfg.TRAIN.LR}],
                lr=cfg.TRAIN.LR,
                momentum=cfg.TRAIN.MOMENTUM,
                weight_decay=cfg.TRAIN.WD,
                nesterov=cfg.TRAIN.NESTEROV
            )
        elif cfg.TRAIN.OPTIMIZER == 'adam':
            optimizer = optim.Adam(
                [{'params': filter(lambda p: p.requires_grad, model.parameters()),
                  'initial_lr': cfg.TRAIN.LR}],
                lr=cfg.TRAIN.LR
            )
        else:
            raise(KeyError, '%s not supported yet...'%cfg.TRAIN.OPTIMIZER)
    else:
        for group in policies:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

        if cfg.TRAIN.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(
                policies,
                lr=cfg.TRAIN.LR,
                momentum=cfg.TRAIN.MOMENTUM,
                weight_decay=cfg.TRAIN.WD,
                nesterov=cfg.TRAIN.NESTEROV
            )
        elif cfg.TRAIN.OPTIMIZER == 'adam':
            optimizer = optim.Adam(
                policies,
                lr=cfg.TRAIN.LR
            )
        elif cfg.TRAIN.OPTIMIZER == 'radam':
            optimizer = RAdam(
                policies,
                lr=cfg.TRAIN.LR
            )
        else:
            raise(KeyError, '%s not supported yet...'%cfg.TRAIN.OPTIMIZER)

    return optimizer


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def video_to_stimg(x, t_size):
    t, c, h, w = x.shape
    assert h==w
    inv = h

    # image as pixel
    pixels = x.as_strided(
        (c, t_size, inv, t_size, inv),
        (inv ** 2, c * h * w * t_size, inv, c * h * w, 1)
    ).contiguous()
    pixels = pixels.permute(0, 2, 1, 4, 3).contiguous()
    out_img = pixels.as_strided(
        (c, inv * t_size, inv * t_size),
        (t_size ** 2 * inv ** 2, t_size * inv, 1)
    ).contiguous()

    return out_img


def stimg_to_video(x, t_size):
    c, h, w = x.shape
    assert h == w
    inv = h // t_size
    grids = x.as_strided(
        (t_size, t_size, c, inv, inv),
        (w, 1, h * w, t_size * w, t_size)
    ).contiguous()
    vid_data = grids.as_strided(
        (t_size ** 2, c, inv, inv),
        (c * inv ** 2, inv ** 2, inv, 1)
    ).contiguous()

    return vid_data


class MEModule(nn.Module):
    """ Motion exciation module

    :param reduction=16
    :param n_segment=8/16
    """

    def __init__(self, channel, reduction=16, n_segment=8):
        super(MEModule, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.n_segment = n_segment
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel // self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel // self.reduction)

        self.conv2 = nn.Conv2d(
            in_channels=self.channel // self.reduction,
            out_channels=self.channel // self.reduction,
            kernel_size=3,
            padding=1,
            groups=channel // self.reduction,
            bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)

        self.conv3 = nn.Conv2d(
            in_channels=self.channel // self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        self.identity = nn.Identity()

    def forward(self, x):
        nt, c, h, w = x.size()
        bottleneck = self.conv1(x)  # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck)  # nt, c//r, h, w

        # t feature
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w
        t_fea, __ = reshape_bottleneck.split([self.n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w

        # apply transformation conv to t+1 feature
        conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
        # reshape fea: n, t, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:])
        __, tPlusone_fea = reshape_conv_bottleneck.split([1, self.n_segment - 1], dim=1)  # n, t-1, c//r, h, w

        # motion fea = t+1_fea - t_fea
        # pad the last timestamp
        diff_fea = tPlusone_fea - t_fea  # n, t-1, c//r, h, w
        # pad = (0,0,0,0,0,0,0,1)
        diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, t, c//r, h, w
        diff_fea_pluszero = diff_fea_pluszero.view((-1,) + diff_fea_pluszero.size()[2:])  # nt, c//r, h, w
        y = self.avg_pool(diff_fea_pluszero)  # nt, c//r, 1, 1
        y = self.conv3(y)  # nt, c, 1, 1
        y = self.bn3(y)  # nt, c, 1, 1
        y = self.sigmoid(y)  # nt, c, 1, 1
        y = y - 0.5
        output = x + x * y.expand_as(x)
        return output


