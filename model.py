#!/usr/bin/python3.6
''' Trains a model. '''

import argparse, hashlib, logging, math, multiprocessing
import os, pickle, pprint, sys, time
from typing import *
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models

from easydict import EasyDict as edict
import PIL
import torchsummary
# import pretrainedmodels
from pytorchcv.model_provider import get_model

from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

from data_loader_v1_balanced import Dataset
from utils import create_logger, AverageMeter, GAP
from debug import dprint, assert_eq, assert_ne
from cosine_scheduler import CosineLRWithRestarts
import albumentations as albu

IN_KERNEL = False

opt = edict()

opt.MODEL = edict()
opt.MODEL.ARCH = 'resnet50'
opt.MODEL.IMAGE_SIZE = 64
opt.MODEL.INPUT_SIZE = 64
opt.MODEL.VERSION = os.path.splitext(os.path.basename(__file__))[0][6:]
opt.MODEL.DROPOUT = 0.5
opt.MODEL.NUM_CLASSES = 203094

opt.EXPERIMENT_DIR = f'../models/{opt.MODEL.VERSION}'

opt.TRAIN = edict()
opt.TRAIN.VAL_SAMPLES = 100000
opt.TRAIN.IMAGES_PER_CLASS = 10
opt.TRAIN.STEPS_PER_EPOCH = 30000

opt.TRAIN.SPLITS_FILE = '../cache/splits.pkl'
opt.TRAIN.NUM_FOLDS = 5
opt.TRAIN.BATCH_SIZE = 256 * torch.cuda.device_count()
opt.TRAIN.LOSS = 'CE'
opt.TRAIN.SHUFFLE = True
opt.TRAIN.WORKERS = min(12, multiprocessing.cpu_count())
opt.TRAIN.PRINT_FREQ = 100
opt.TRAIN.LEARNING_RATE = 1e-4
opt.TRAIN.PATIENCE = 4
opt.TRAIN.LR_REDUCE_FACTOR = 0.2
opt.TRAIN.MIN_LR = 1e-7
opt.TRAIN.EPOCHS = 1000
opt.TRAIN.PATH = f'../data/train_{opt.MODEL.INPUT_SIZE}'
opt.TRAIN.OPTIMIZER = 'Adam'
opt.TRAIN.MIN_IMPROVEMENT = 0.001

opt.TRAIN.COSINE = edict()
opt.TRAIN.COSINE.ENABLE = False
opt.TRAIN.COSINE.LR = 1e-4
opt.TRAIN.COSINE.PERIOD = 10
opt.TRAIN.COSINE.COEFF = 1.2

opt.TEST = edict()
opt.TEST.PATH = f'../data/test_{opt.MODEL.INPUT_SIZE}'
opt.TEST.NUM_TTAS = 1


def train_val_split() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(opt.TRAIN.SPLITS_FILE):
        full_df = pd.read_csv('../data/train.csv')
        print('full_df', full_df.shape)

        value_counts = full_df.landmark_id.value_counts()
        more_than_one_sample = value_counts[value_counts > 1].index
        print('classes with more than 1 sample', more_than_one_sample.shape)

        val_candidates_df = full_df.loc[full_df.landmark_id.isin(more_than_one_sample)]
        val_candidates = val_candidates_df.groupby('landmark_id').first().id
        val_ids = val_candidates.sample(opt.TRAIN.VAL_SAMPLES, random_state=0)

        train_df = full_df.loc[~full_df.id.isin(val_ids)]
        val_df = full_df.loc[full_df.id.isin(val_ids)]

        dprint(train_df.shape)
        dprint(val_df.shape)

        with open(opt.TRAIN.SPLITS_FILE, 'wb') as f:
            pickle.dump((train_df, val_df), f)
    else:
        with open(opt.TRAIN.SPLITS_FILE, 'rb') as f:
            train_df, val_df = pickle.load(f)

    return train_df, val_df

def load_data(fold: int, params: Dict[str, Any]) -> Any:
    torch.multiprocessing.set_sharing_strategy('file_system')
    cudnn.benchmark = True

    logger.info('Options:')
    logger.info(pprint.pformat(opt))

    train_df, val_df = train_val_split()
    print('train_df', train_df.shape, 'val_df', val_df.shape)
    test_df = pd.read_csv('../data/test.csv')

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df.landmark_id.values)
    print('found classes', len(label_encoder.classes_))
    assert len(label_encoder.classes_) == opt.MODEL.NUM_CLASSES

    train_df.landmark_id = label_encoder.transform(train_df.landmark_id)
    val_df.landmark_id = label_encoder.transform(val_df.landmark_id)

    transform_train = albu.Compose([
        albu.HorizontalFlip(.5),
        albu.OneOf([
            albu.IAAAdditiveGaussianNoise(),
            albu.GaussNoise(),
        ], p=0.2),
        albu.OneOf([
            albu.MotionBlur(p=.2),
            albu.MedianBlur(blur_limit=3, p=0.1),
            albu.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        albu.OneOf([
            albu.OpticalDistortion(p=0.3),
            albu.GridDistortion(p=.1),
            albu.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        albu.OneOf([
            albu.CLAHE(clip_limit=2),
            albu.IAASharpen(),
            albu.IAAEmboss(),
            albu.RandomBrightnessContrast(),
        ], p=0.3),
        albu.HueSaturationValue(p=0.3),
    ])

    if opt.TEST.NUM_TTAS > 1:
        transform_test = albu.Compose([
            albu.PadIfNeeded(opt.MODEL.INPUT_SIZE, opt.MODEL.INPUT_SIZE),
            albu.RandomCrop(height=opt.MODEL.INPUT_SIZE, width=opt.MODEL.INPUT_SIZE),
            albu.HorizontalFlip(),
        ])
    else:
        transform_test = albu.Compose([
            albu.PadIfNeeded(opt.MODEL.INPUT_SIZE, opt.MODEL.INPUT_SIZE),
            albu.CenterCrop(height=opt.MODEL.INPUT_SIZE, width=opt.MODEL.INPUT_SIZE),
        ])


    train_dataset = Dataset(train_df, path=opt.TRAIN.PATH, mode='train',
                            image_size=opt.MODEL.IMAGE_SIZE,
                            num_classes=opt.MODEL.NUM_CLASSES,
                            images_per_class=opt.TRAIN.IMAGES_PER_CLASS,
                            aug_type='albu', augmentor=transform_train)

    val_dataset = Dataset(val_df, path=opt.TRAIN.PATH, mode='val',
                          image_size=opt.MODEL.IMAGE_SIZE,
                          num_classes=opt.MODEL.NUM_CLASSES)
    test_dataset = Dataset(test_df, path=opt.TEST.PATH, mode='test',
                           image_size=opt.MODEL.IMAGE_SIZE,
                           num_classes=opt.MODEL.NUM_CLASSES)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.TRAIN.BATCH_SIZE, shuffle=True,
        num_workers=opt.TRAIN.WORKERS)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.TRAIN.BATCH_SIZE, shuffle=False, num_workers=opt.TRAIN.WORKERS)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.TRAIN.BATCH_SIZE, shuffle=False, num_workers=opt.TRAIN.WORKERS)

    return train_loader, val_loader, test_loader, label_encoder

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = get_model(opt.MODEL.ARCH, pretrained=True)
        assert(opt.MODEL.INPUT_SIZE % 32 == 0)

        self.model.features[-1] = nn.AdaptiveAvgPool2d(1)

        self.model.output = nn.Sequential(
            nn.Dropout(opt.MODEL.DROPOUT),
            nn.Linear(2048, 512),
            nn.Dropout(opt.MODEL.DROPOUT),
            nn.Linear(512, opt.MODEL.NUM_CLASSES),
            )

        # self.dropout = nn.Dropout2d(opt.MODEL.DROPOUT, inplace=True)
        # self.head = Head(channel_size, num_outputs)

    def forward(self, images: torch.tensor, labels: torch.tensor=None) -> torch.tensor:
        return self.model.forward(images)

        # features = self.model.features(images)
        # features = self.bn1(features)
        # features = self.dropout(features)
        # features = features.view(features.size(0), -1)
        # features = self.fc1(features)
        # features = self.bn2(features)
        #
        # features = self.head(features)
        # features = F.normalize(features)
        # return features

def create_model() -> Any:
    logger.info(f'creating a model {opt.MODEL.ARCH}')
    assert(opt.MODEL.INPUT_SIZE % 32 == 0)

    model = Net()
    model = torch.nn.DataParallel(model).cuda()
    model.cuda()

    # if torch.cuda.device_count() == 1:
    #     torchsummary.summary(model, (3, opt.MODEL.INPUT_SIZE, opt.MODEL.INPUT_SIZE))

    return model

def save_checkpoint(state: Dict[str, Any], filename: str, model_dir: str) -> None:
    torch.save(state, os.path.join(model_dir, filename))
    logger.info(f'A snapshot was saved to {filename}')

def train(train_loader: Any, model: Any, criterion: Any, optimizer: Any,
          epoch: int, lr_scheduler: Any) -> None:
    logger.info(f'epoch {epoch}')
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()

    num_steps = min(len(train_loader), opt.TRAIN.STEPS_PER_EPOCH)
    print('total batches:', num_steps)

    threshold = 0.1
    end = time.time()

    for i, (input_, target) in enumerate(train_loader):
        if i >= opt.TRAIN.STEPS_PER_EPOCH:
            break

        # compute output
        output = model(input_.cuda())
        loss = criterion(output, target.cuda())

        # get metric
        confs, predicts = torch.max(output.detach(), dim=1)
        avg_score.update(GAP(predicts, confs, target, threshold))

        # compute gradient and do SGD step
        losses.update(loss.data.item(), input_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if hasattr(lr_scheduler, 'batch_step'):
            lr_scheduler.batch_step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.TRAIN.PRINT_FREQ == 0:
            logger.info(f'{epoch} [{i}/{num_steps}]\t'
                        f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'GAP {avg_score.val:.4f} ({avg_score.avg:.4f})')

    logger.info(f' * average GAP on train {avg_score.avg:.4f}')

def inference(data_loader: Any, model: Any) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    ''' Returns predictions and targets, if any. '''
    model.eval()

    activation = nn.Softmax(dim=1)
    all_predicts, all_confs, all_targets = [], [], []

    with torch.no_grad():
        for i, (input_, target) in enumerate(tqdm(data_loader, disable=IN_KERNEL)):
            if opt.TEST.NUM_TTAS != 1 and data_loader.dataset.mode == 'test':
                bs, ncrops, c, h, w = input_.size()
                input_ = input_.view(-1, c, h, w) # fuse batch size and ncrops

                output = model(input_)
                output = activation(output)

                if opt.TEST.TTA_COMBINE_FUNC == 'max':
                    output = output.view(bs, ncrops, -1).max(1)[0]
                elif opt.TEST.TTA_COMBINE_FUNC == 'mean':
                    output = output.view(bs, ncrops, -1).mean(1)
                else:
                    assert False
            else:
                output = model(input_.cuda())
                output = activation(output)

            confs, predicts = torch.max(output, dim=1)
            all_confs.append(confs)
            all_predicts.append(predicts)

            if target is not None:
                all_targets.append(target)

    predicts = torch.cat(all_predicts)
    confs = torch.cat(all_confs)
    targets = torch.cat(all_targets)

    return predicts, confs, targets

def validate(val_loader: Any, model: Any, epoch: int) -> float:
    ''' Calculates validation score.
    1. Infers predictions
    2. Finds optimal threshold
    3. Returns the best score and a threshold. '''
    logger.info('validate()')

    predicts, confs, targets = inference(val_loader, model)
    threshold = 0.1
    score = GAP(predicts, confs, targets, threshold)

    logger.info(f'{epoch} GAP {score:.4f}')
    logger.info(f' * GAP on validation {score:.4f}')
    return score

def generate_submission(val_loader: Any, test_loader: Any, model: Any,
                        label_encoder: Any, epoch: int, model_path: Any) -> np.ndarray:
    predicts, confs, _ = inference(test_loader, model)

    labels = [label_encoder.inverse_transform(pred) for pred in predicts]
    print('labels')
    print(np.array(labels))

    sub = test_loader.dataset.df
    sub.landmark_id = labels
    sub.to_csv(f'../submissions/{os.path.basename(model_path)[:-4]}.csv', index=False)

def set_lr(optimizer: Any, lr: float) -> None:
    for param_group in optimizer.param_groups:
       param_group['lr'] = lr
       param_group['initial_lr'] = lr

def read_lr(optimizer: Any) -> float:
    for param_group in optimizer.param_groups:
       lr = float(param_group['lr'])
       logger.info(f'learning rate: {lr}')
       return lr

    assert False

def train_model(params: Dict[str, Any]) -> float:
    np.random.seed(0)
    model_dir = opt.EXPERIMENT_DIR

    logger.info('=' * 50)
    logger.info(f'hyperparameters: {params}')

    train_loader, val_loader, test_loader, label_encoder = load_data(args.fold, params)
    model = create_model() # float(params['dropout']))
    # freeze_layers(model)

    # if torch.cuda.device_count() == 1:
    #     torchsummary.summary(model, (3, 224, 224))

    if opt.TRAIN.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(), opt.TRAIN.LEARNING_RATE)
    elif opt.TRAIN.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), opt.TRAIN.LEARNING_RATE,
                              momentum=0.9, nesterov=True)
    else:
        assert False

    if opt.TRAIN.COSINE.ENABLE:
        set_lr(optimizer, opt.TRAIN.COSINE.LR)
        lr_scheduler = CosineLRWithRestarts(optimizer, opt.TRAIN.BATCH_SIZE,
            opt.TRAIN.BATCH_SIZE * opt.TRAIN.STEPS_PER_EPOCH,
            restart_period=opt.TRAIN.COSINE.PERIOD, t_mult=opt.TRAIN.COSINE.COEFF)
    else:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                           patience=opt.TRAIN.PATIENCE, factor=opt.TRAIN.LR_REDUCE_FACTOR,
                           verbose=True, min_lr=opt.TRAIN.MIN_LR,
                           threshold=opt.TRAIN.MIN_IMPROVEMENT, threshold_mode='abs')

    if args.weights is None:
        last_epoch = 0
        logger.info(f'training will start from epoch {last_epoch+1}')
    else:
        last_checkpoint = torch.load(args.weights)
        assert(last_checkpoint['arch']==opt.MODEL.ARCH)
        model.load_state_dict(last_checkpoint['state_dict'])
        optimizer.load_state_dict(last_checkpoint['optimizer'])
        logger.info(f'checkpoint {args.weights} was loaded.')

        last_epoch = last_checkpoint['epoch']
        logger.info(f'loaded the model from epoch {last_epoch}')
        set_lr(optimizer, opt.TRAIN.LEARNING_RATE)


    if args.predict:
        print('inference mode')
        generate_submission(val_loader, test_loader, model, label_encoder,
                            last_epoch, args.pretrained)
        sys.exit(0)

    if opt.TRAIN.LOSS == 'CE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise RuntimeError('unknown loss specified')

    best_score = 0.0
    best_epoch = 0

    last_lr = read_lr(optimizer)
    best_model_path = None

    for epoch in range(last_epoch + 1, opt.TRAIN.EPOCHS + 1):
        logger.info('-' * 50)

        if not opt.TRAIN.COSINE.ENABLE:
            lr = read_lr(optimizer)
            if lr < last_lr - 1e-10 and best_model_path is not None:
                # reload the best model
                last_checkpoint = torch.load(os.path.join(model_dir, best_model_path))
                assert(last_checkpoint['arch']==opt.MODEL.ARCH)
                model.load_state_dict(last_checkpoint['state_dict'])
                optimizer.load_state_dict(last_checkpoint['optimizer'])
                logger.info(f'checkpoint {best_model_path} was loaded.')
                set_lr(optimizer, lr)
                last_lr = lr

            if lr < opt.TRAIN.MIN_LR * 1.01:
                logger.info('reached minimum LR, stopping')
                break

                # logger.info(f'lr={lr}, start cosine annealing!')
                # set_lr(optimizer, opt.TRAIN.COSINE.LR)
                # opt.TRAIN.COSINE.ENABLE = True
                #
                # lr_scheduler = CosineLRWithRestarts(optimizer, opt.TRAIN.BATCH_SIZE,
                #     opt.TRAIN.BATCH_SIZE * opt.TRAIN.STEPS_PER_EPOCH,
                #     restart_period=opt.TRAIN.COSINE.PERIOD, t_mult=opt.TRAIN.COSINE.COEFF)

        if opt.TRAIN.COSINE.ENABLE:
            lr_scheduler.step()

        read_lr(optimizer)

        train(train_loader, model, criterion, optimizer, epoch, lr_scheduler)
        score = validate(val_loader, model, epoch)

        if not opt.TRAIN.COSINE.ENABLE:
            lr_scheduler.step(score)    # type: ignore

        is_best = score > best_score
        best_score = max(score, best_score)
        if is_best:
            best_epoch = epoch

        data_to_save = {
            'epoch': epoch,
            'arch': opt.MODEL.ARCH,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'score': score,
            'optimizer': optimizer.state_dict(),
            'options': opt
        }

        filename = opt.MODEL.VERSION
        if is_best:
            best_model_path = f'{filename}_f{args.fold}_e{epoch:02d}_{score:.04f}.pth'
            save_checkpoint(data_to_save, best_model_path, model_dir)

    logger.info(f'best score: {best_score:.04f}')
    return -best_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', help='model to resume training', type=str)
    parser.add_argument('--fold', help='fold number', type=int, default=0)
    parser.add_argument('--predict', help='model to resume training', action='store_true')
    parser.add_argument('--num_tta', help='number of TTAs', type=int, default=opt.TEST.NUM_TTAS)
    args = parser.parse_args()

    params = {'affine': 'medium',
              'aug_global_prob': 0.5346290229823514,
              'blur': 0.1663552826866818,
              'color': 0.112355821364934,
              'distortion': 0.12486453027371469,
              'dropout': 0.3,
              'noise': 0.29392632695458587,
              'rotate90': 0,
              'vflip': 0}

    opt.EXPERIMENT_DIR = os.path.join(opt.EXPERIMENT_DIR, f'fold_{args.fold}')
    opt.TEST.NUM_TTAS = args.num_tta

    if not os.path.exists(opt.EXPERIMENT_DIR):
        os.makedirs(opt.EXPERIMENT_DIR)

    logger = create_logger(os.path.join(opt.EXPERIMENT_DIR, 'log_training.txt'))
    train_model(params)
