''' All available optimizers are defined here. '''

import json
from typing import Any
import torch.optim as optim


def adam(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0,
         amsgrad=False, **_):
    if isinstance(betas, str):
        betas = json.loads(betas)

    return optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay,
                      amsgrad=amsgrad)

def sgd(parameters, lr=0.001, momentum=0.9, weight_decay=0, nesterov=True, **_):
    return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay,
                     nesterov=nesterov)

def get_optimizer(config, parameters):
    f = globals().get(config.optimizer.name)
    return f(parameters, **config.optimizer.params)

def set_lr(optimizer: Any, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['initial_lr'] = lr

def get_lr(optimizer: Any) -> float:
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        return lr

    assert False
