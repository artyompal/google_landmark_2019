''' Reads config file and merges settings with default ones. '''

import multiprocessing
import os
import yaml

import torch
from easydict import EasyDict as edict


def _get_default_config(filename: str) -> edict:
    cfg = edict()
    cfg.in_kernel = False
    cfg.version = os.path.splitext(os.path.basename(filename))[0]
    cfg.experiment_dir = f'../models/{cfg.version}'
    cfg.num_workers = min(12, multiprocessing.cpu_count())

    cfg.data = edict()
    cfg.data.data_dir = '../data/'
    cfg.data.train_dir = '../data/train/'
    cfg.data.test_dir = '../data/test/'
    cfg.data.train_filename = None
    cfg.data.params = edict()

    cfg.model = edict()
    cfg.model.arch = 'resnet50'
    cfg.model.image_size = 256
    cfg.model.input_size = 224
    cfg.model.num_classes = None
    cfg.model.dropout = 0

    cfg.train = edict()
    cfg.train.batch_size = 32 * torch.cuda.device_count()
    cfg.train.num_epochs = 1000
    cfg.train.num_folds = 5
    cfg.train.shuffle = True
    cfg.train.images_per_class = None
    cfg.train.max_steps_per_epoch = None
    cfg.train.log_freq = 100
    cfg.train.min_lr = 3e-7
    cfg.train.lr_scheduler = None
    cfg.train.use_balancing_sampler = False

    cfg.val = edict()
    cfg.val.batch_size = 64 * torch.cuda.device_count()
    cfg.val.images_per_class = 10

    cfg.test = edict()
    cfg.test.batch_size = 64 * torch.cuda.device_count()
    cfg.test.num_ttas = 1

    cfg.optimizer = edict()
    cfg.optimizer.name = 'adam'
    cfg.optimizer.params = edict()

    cfg.scheduler = edict()
    cfg.scheduler.name = 'none'
    cfg.scheduler.params = edict()

    cfg.loss = edict()
    cfg.loss.name = 'none'

    return cfg

def _merge_config(src: edict, dst: edict) -> edict:
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v

def load(config_path: str) -> edict:
    with open(config_path) as f:
        yaml_config = edict(yaml.load(f, Loader=yaml.SafeLoader))

    config = _get_default_config(config_path)
    _merge_config(yaml_config, config)

    return config
