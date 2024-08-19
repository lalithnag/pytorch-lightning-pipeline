import os
import sys
import torch
import numpy as np


def dice_coeff(pred, target):
    smooth = 1.

    p_flat = pred.view(-1) # p is of N,C,H,W
    t_flat = target.view(-1) # t is of N, C, H, W
    intersection = (p_flat * t_flat).sum()
    return ((2. * intersection + smooth) / (p_flat.sum() + t_flat.sum() + smooth)).mean()


def dice_coeff_loss(pred, target):
    return 1 - dice_coeff(pred=pred, target=target)