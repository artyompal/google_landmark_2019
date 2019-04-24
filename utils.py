import logging, os
import inspect, re
from typing import Any

import numpy as np
import torch
from debug import dprint


def create_logger(filename: str) -> Any:
    logger_name = 'logger'
    file_fmt_str = '%(asctime)s %(message)s'
    console_fmt_str = '%(message)s'
    file_level = logging.DEBUG
    console_level = logging.DEBUG

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    file_fmt = logging.Formatter(file_fmt_str, '%m-%d %H:%M:%S')
    log_file = logging.FileHandler(filename)
    log_file.setLevel(file_level)
    log_file.setFormatter(file_fmt)
    logger.addHandler(log_file)

    console_fmt = logging.Formatter(console_fmt_str)
    log_console = logging.StreamHandler()
    log_console.setLevel(logging.DEBUG)
    log_console.setFormatter(console_fmt)
    logger.addHandler(log_console)

    return logger

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def GAP(predicts: torch.tensor, confs: torch.tensor, targets: torch.tensor,
        threshold: float) -> torch.tensor:
    ''' Computes GAP@1 '''
    if len(predicts.shape) != 1:
        dprint(predicts.shape)
        assert False

    if len(confs.shape) != 1:
        dprint(confs.shape)
        assert False

    if len(targets.shape) != 1:
        dprint(targets.shape)
        assert False

    assert predicts.shape == confs.shape and confs.shape == targets.shape

    sorted_confs, indices = torch.sort(confs, dim=1, descending=True)

    confs = confs.numpy().cpu()
    predicts = predicts[indices].numpy().cpu()
    targets = targets[indices].numpy().cpu()

    res, true_pos = 0.0, 0

    for i, (c, p, t) in enumerate(zip(confs, predicts, targets)):
        rel = int(p == t)
        true_pos += rel

        res += true_pos / i * rel

    res /= targets.shape[0] # FIXME: incorrect, not all test images depict landmarks
    return res
