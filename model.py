#!/usr/bin/python3.6
''' Trains a model or infers predictions. '''

import argparse
import hashlib
import logging
import math
import multiprocessing
import os
import pickle
import pprint
import sys
import time

from typing import *
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.multiprocessing
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchsummary

import albumentations as albu

from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
from easydict import EasyDict as edict

import parse_config

from data_loader import ImageDataset
from balancing_sampler import BalancingSampler
from utils import create_logger, AverageMeter
from debug import dprint

from losses import get_loss
from schedulers import get_scheduler, is_scheduler_continuous
from optimizers import get_optimizer, get_lr, set_lr
from metrics import GAP
from random_rect_crop import RandomRectCrop


def load_data(fold: int) -> Any:
    torch.multiprocessing.set_sharing_strategy('file_system')
    cudnn.benchmark = True

    logger.info('config:')
    logger.info(pprint.pformat(config))

    fname = f'{config.data.train_filename}_fold_{fold}_'
    train_df = pd.read_csv(os.path.join(config.data.data_dir, fname + 'train.csv'))
    val_df = pd.read_csv(os.path.join(config.data.data_dir, fname + 'val.csv'))
    print('train_df', train_df.shape, 'val_df', val_df.shape)

    val_df = pd.concat([c[1].iloc[:config.val.images_per_class]
                        for c in val_df.groupby('landmark_id')])
    print('val_df after class filtering', val_df.shape)

    test_df = pd.read_csv('../data/test.csv', dtype=str)
    # test_df.drop(columns='url', inplace=True)
    print('test_df', test_df.shape)

    test_df = test_df.loc[test_df.id.apply(lambda img: os.path.exists(os.path.join(
        config.data.test_dir, img + '.jpg')))]
    print('test_df after filtering', test_df.shape)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df.landmark_id.values)
    print('found classes', len(label_encoder.classes_))
    assert len(label_encoder.classes_) == config.model.num_classes

    train_df.landmark_id = label_encoder.transform(train_df.landmark_id)
    val_df.landmark_id = label_encoder.transform(val_df.landmark_id)

    augs = []

    if config.data.use_rect_crop:
        augs.append(RandomRectCrop(rect_min_area=config.data.rect_min_area,
                                   rect_min_ratio=config.data.rect_min_ratio,
                                   image_size=config.model.image_size,
                                   input_size=config.model.input_size))

    if config.augmentations.blur != 0:
        augs.append(albu.OneOf([
            albu.MotionBlur(p=.2),
            albu.MedianBlur(blur_limit=3, p=0.1),
            albu.Blur(blur_limit=3, p=0.1),
        ], p=config.augmentations.blur))

    if config.augmentations.color != 0:
        augs.append(albu.OneOf([
            albu.CLAHE(clip_limit=2),
            albu.IAASharpen(),
            albu.IAAEmboss(),
            albu.RandomBrightnessContrast(),
        ], p=config.augmentations.color))

    augs.append(albu.HorizontalFlip(0.5))
    transform_train = albu.Compose(augs)

    if config.test.num_ttas > 1:
        transform_test = albu.Compose([
            albu.RandomCrop(height=config.model.input_size, width=config.model.input_size),
            albu.HorizontalFlip(),
        ])
    else:
        transform_test = albu.Compose([
            albu.CenterCrop(height=config.model.input_size, width=config.model.input_size),
        ])


    train_dataset = ImageDataset(train_df, path=config.data.train_dir, mode='train',
                                 image_size=config.model.image_size,
                                 num_classes=config.model.num_classes,
                                 images_per_class=config.train.images_per_class,
                                 aug_type='albu', augmentor=transform_train)

    val_dataset = ImageDataset(val_df, path=config.data.train_dir, mode='val',
                               image_size=config.model.image_size,
                               num_classes=config.model.num_classes)

    if args.dataset == 'test':
        test_dataset = ImageDataset(test_df, path=config.data.test_dir, mode='test',
                                    image_size=config.model.image_size,
                                    input_size=config.model.input_size,
                                    num_ttas=config.test.num_ttas,
                                    num_classes=config.model.num_classes)
    else:
        train_df = pd.read_csv('../data/train.csv')
        train_df.drop(columns=['url', 'landmark_id'], inplace=True)

        test_dataset = ImageDataset(train_df, path=config.data.train_dir, mode='test',
                                    image_size=config.model.image_size,
                                    num_classes=config.model.num_classes)

    if config.train.use_balancing_sampler:
        sampler = BalancingSampler(train_df, num_classes=config.model.num_classes,
                                   images_per_class=config.train.images_per_class)
        shuffle = False
    else:
        sampler = None
        shuffle = config.train.shuffle

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size,
                              sampler=sampler, shuffle=shuffle,
                              num_workers=config.num_workers, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=config.val.batch_size,
                            shuffle=False, num_workers=config.num_workers)

    test_loader = DataLoader(test_dataset, batch_size=config.test.batch_size,
                             shuffle=False, num_workers=config.num_workers)


    return train_loader, val_loader, test_loader, label_encoder

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = get_model(config.model.arch,
                               pretrained=not (args.gen_predict or args.gen_features))
        assert config.model.input_size % 32 == 0

        self.model.features[-1] = nn.AdaptiveAvgPool2d(1)
        in_features = self.model.output.in_features

        if config.model.bottleneck_fc is not None:
            self.model.output = nn.Sequential(
                nn.Linear(in_features, config.model.bottleneck_fc),
                nn.Linear(config.model.bottleneck_fc, config.model.num_classes))
        else:
            self.model.output = nn.Linear(in_features, config.model.num_classes)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.features(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.forward(images)

def create_model() -> Any:
    logger.info(f'creating a model {config.model.arch}')

    model = Model()
    model = torch.nn.DataParallel(model).cuda()
    model.cuda()

    if args.summary:
        torchsummary.summary(model, (3, config.model.input_size, config.model.input_size))

    return model

def save_checkpoint(state: Dict[str, Any], filename: str, model_dir: str) -> None:
    torch.save(state, os.path.join(model_dir, filename))
    logger.info(f'A snapshot was saved to {filename}')

def train(train_loader: Any, model: Any, criterion: Any, optimizer: Any,
          epoch: int, lr_scheduler: Any, lr_scheduler2: Any) -> None:
    logger.info(f'epoch {epoch}')
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()

    num_steps = len(train_loader)
    if config.train.max_steps_per_epoch is not None:
        num_steps = min(len(train_loader), config.train.max_steps_per_epoch)

    logger.info(f'total batches: {num_steps}')

    end = time.time()
    lr_str = ''

    for i, (input_, target) in enumerate(train_loader):
        if i >= num_steps:
            break

        # compute output
        output = model(input_.cuda())
        loss = criterion(output, target.cuda())

        # get metric
        confs, predicts = torch.max(output.detach(), dim=1)
        avg_score.update(GAP(predicts, confs, target))

        # compute gradient and do SGD step
        losses.update(loss.data.item(), input_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_updated = False

        if is_scheduler_continuous(config.scheduler.name):
            lr_scheduler.step()
            lr_updated = True

        if is_scheduler_continuous(config.scheduler2.name):
            lr_scheduler2.step()
            lr_updated = True

        if lr_updated:
            lr_str = f'\tlr {get_lr(optimizer):.08f}'

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.train.log_freq == 0:
            logger.info(f'{epoch} [{i}/{num_steps}]\t'
                        f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'GAP {avg_score.val:.4f} ({avg_score.avg:.4f})'
                        + lr_str)

    logger.info(f' * average GAP on train {avg_score.avg:.4f}')

def inference(data_loader: Any, model: Any) -> Tuple[torch.Tensor, torch.Tensor,
                                                     Optional[torch.Tensor]]:
    ''' Returns predictions and targets, if any. '''
    model.eval()

    activation = nn.Softmax(dim=1)
    all_predicts, all_confs, all_targets = [], [], []

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, disable=config.in_kernel)):
            if data_loader.dataset.mode != 'test':
                input_, target = data
            else:
                input_, target = data, None

            if config.test.num_ttas != 1 and data_loader.dataset.mode == 'test':
                bs, num_ttas, c, h, w = input_.size()
                input_ = input_.view(-1, c, h, w)

                output = model(input_.cuda())
                output = activation(output)

                if config.test.tta_combine_func == 'max':
                    output = output.view(bs, num_ttas, -1).max(dim=1)[0]
                elif config.test.tta_combine_func == 'mean':
                    output = output.view(bs, num_ttas, -1).mean(dim=1)
                else:
                    assert False
            else:
                output = model(input_.cuda())
                output = activation(output)

            confs, predicts = torch.topk(output, config.test.num_predicts)
            all_confs.append(confs)
            all_predicts.append(predicts)

            if target is not None:
                all_targets.append(target)

    predicts = torch.cat(all_predicts)
    confs = torch.cat(all_confs)
    targets = torch.cat(all_targets) if len(all_targets) else None

    return predicts, confs, targets

def validate(val_loader: Any, model: Any, epoch: int) -> float:
    ''' Infers predictions and calculates validation score. '''
    logger.info('validate()')

    predicts, confs, targets = inference(val_loader, model)
    predicts, confs = predicts[:, 0], confs[:, 0] # FIXME: use config.test.num_predicts?
    assert targets is not None
    score = GAP(predicts, confs, targets)

    logger.info(f'{epoch} GAP {score:.4f}')
    logger.info(f' * GAP on validation {score:.4f}')
    return score

def generate_features(test_loader: Any, model: Any, model_path: Any) -> None:
    model.eval()

    all_features = []
    max_num_of_samples = 500000
    max_batches = max_num_of_samples // config.test.batch_size
    part_idx = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, disable=config.in_kernel)):
            input_, target = data, None

            if config.test.num_ttas != 1 and test_loader.dataset.mode == 'test':
                bs, num_ttas, c, h, w = input_.size()
                input_ = input_.view(-1, c, h, w)

                features = model.module.features(input_.cuda())
                features = features.view(bs, num_ttas, -1).mean(dim=1)
            else:
                features = model.module.features(input_.cuda())

            all_features.append(features)

            if len(all_features) == max_batches or i == len(test_loader) - 1:
                all_features = torch.cat(all_features).cpu().numpy()
                model_name = os.path.basename(model_path)[:-4]
                np.save(f'../features/{args.dataset}_{model_name}_part{part_idx:02d}.npy', all_features)

                part_idx += 1
                all_features = []

def generate_submission(val_loader: Any, test_loader: Any, model: Any,
                        label_encoder: Any, epoch: int, model_path: Any) -> np.ndarray:
    sample_sub = pd.read_csv('../data/recognition_sample_submission.csv')

    predicts_gpu, confs_gpu, _ = inference(test_loader, model)
    predicts, confs = predicts_gpu.cpu().numpy(), confs_gpu.cpu().numpy()

    labels = [label_encoder.inverse_transform(pred) for pred in predicts]
    print('labels')
    print(np.array(labels))
    print('confs')
    print(np.array(confs))

    sub = test_loader.dataset.df
    def concat(label, conf):
        return ' '.join([f'{L} {c}' for L, c in zip(label, conf)])
    sub['landmarks'] = [concat(label, conf) for label, conf in zip(labels, confs)]

    sample_sub = sample_sub.set_index('id')
    sub = sub.set_index('id')
    sample_sub.update(sub)

    sample_sub.to_csv(f'../submissions/{os.path.basename(model_path)[:-4]}.csv')

def run() -> float:
    np.random.seed(0)
    model_dir = config.experiment_dir

    logger.info('=' * 50)
    # logger.info(f'hyperparameters: {params}')

    train_loader, val_loader, test_loader, label_encoder = load_data(args.fold)
    model = create_model()

    optimizer = get_optimizer(config, model.parameters())
    lr_scheduler = get_scheduler(config, optimizer)
    lr_scheduler2 = get_scheduler(config, optimizer) if config.scheduler2.name else None
    criterion = get_loss(config)

    if args.weights is None:
        last_epoch = 0
        logger.info(f'training will start from epoch {last_epoch+1}')
    else:
        last_checkpoint = torch.load(args.weights)
        assert last_checkpoint['arch'] == config.model.arch
        model.load_state_dict(last_checkpoint['state_dict'])
        optimizer.load_state_dict(last_checkpoint['optimizer'])
        logger.info(f'checkpoint {args.weights} was loaded.')

        last_epoch = last_checkpoint['epoch']
        logger.info(f'loaded the model from epoch {last_epoch}')

        if args.lr_override != 0:
            set_lr(optimizer, float(args.lr_override))
        elif 'lr' in config.scheduler.params:
            set_lr(optimizer, config.scheduler.params.lr)

    if args.gen_predict:
        print('inference mode')
        generate_submission(val_loader, test_loader, model, label_encoder,
                            last_epoch, args.weights)
        sys.exit(0)

    if args.gen_features:
        print('inference mode')
        generate_features(test_loader, model, args.weights)
        sys.exit(0)

    best_score = 0.0
    best_epoch = 0

    last_lr = get_lr(optimizer)
    best_model_path = args.weights

    for epoch in range(last_epoch + 1, config.train.num_epochs + 1):
        logger.info('-' * 50)

        # if not is_scheduler_continuous(config.scheduler.name):
        #     # if we have just reduced LR, reload the best saved model
        #     lr = get_lr(optimizer)
        #     logger.info(f'learning rate {lr}')
        #
        #     if lr < last_lr - 1e-10 and best_model_path is not None:
        #         last_checkpoint = torch.load(os.path.join(model_dir, best_model_path))
        #         assert(last_checkpoint['arch']==config.model.arch)
        #         model.load_state_dict(last_checkpoint['state_dict'])
        #         optimizer.load_state_dict(last_checkpoint['optimizer'])
        #         logger.info(f'checkpoint {best_model_path} was loaded.')
        #         set_lr(optimizer, lr)
        #         last_lr = lr
        #
        #     if lr < config.train.min_lr * 1.01:
        #         logger.info('reached minimum LR, stopping')
        #         break

        get_lr(optimizer)

        train(train_loader, model, criterion, optimizer, epoch, lr_scheduler,
              lr_scheduler2)
        score = validate(val_loader, model, epoch)

        if not is_scheduler_continuous(config.scheduler.name):
            lr_scheduler.step(score)
        if lr_scheduler2 and not is_scheduler_continuous(config.scheduler.name):
            lr_scheduler2.step(score)

        is_best = score > best_score
        best_score = max(score, best_score)
        if is_best:
            best_epoch = epoch

        data_to_save = {
            'epoch': epoch,
            'arch': config.model.arch,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'score': score,
            'optimizer': optimizer.state_dict(),
            'options': config
        }

        filename = config.version
        if is_best:
            best_model_path = f'{filename}_f{args.fold}_e{epoch:02d}_{score:.04f}.pth'
            save_checkpoint(data_to_save, best_model_path, model_dir)

    logger.info(f'best score: {best_score:.04f}')
    return -best_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='model configuration file (YAML)', type=str, required=True)
    parser.add_argument('--weights', help='model to resume training', type=str)
    parser.add_argument('--dataset', help='dataset for prediction, train/test',
                        type=str, default='test')
    parser.add_argument('--fold', help='fold number', type=int, default=0)
    parser.add_argument('--gen_predict', help='make predictions for the testset and return', action='store_true')
    parser.add_argument('--gen_features', help='calculate features for the given set', action='store_true')
    parser.add_argument('--summary', help='show model summary', action='store_true')
    parser.add_argument('--lr_override', help='override learning rate', type=float, default=0)
    parser.add_argument('--num_ttas', help='override number of TTAs', type=int, default=0)
    args = parser.parse_args()

    config = parse_config.load(args.config, args)

    if args.num_ttas:
        config.test.num_ttas = args.num_ttas

    if not os.path.exists(config.experiment_dir):
        os.makedirs(config.experiment_dir)

    log_filename = 'log_training.txt' if not args.gen_predict and \
                   not args.gen_features else 'log_predict.txt'
    logger = create_logger(os.path.join(config.experiment_dir, log_filename))
    run()
