''' Add all the necessary metrics here. '''

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict as edict

def cross_entropy() -> Any:
    return torch.nn.CrossEntropyLoss()

def binary_cross_entropy() -> Any:
    return torch.nn.BCEWithLogitsLoss()

def mse_loss() -> Any:
    return torch.nn.MSELoss()

def l1_loss() -> Any:
    return torch.nn.L1Loss()

def smooth_l1_loss() -> Any:
    return torch.nn.SmoothL1Loss()

def get_loss(config: edict) -> Any:
    f = globals().get(config.loss.name)
    return f()
