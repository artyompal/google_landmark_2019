#!/usr/bin/python3.6

import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, path: str, mode: str,
                 image_size: int) -> None:
        print(f'creating data loader - {mode}')
        assert mode in ['train', 'val', 'test']

        self.df = dataframe
        self.path = path
        self.mode = mode
        self.image_size = image_size

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index: int) -> Any:
        ''' Returns: tuple (sample, target) '''
        filename = self.df.id.values[index]

        sample = Image.open(os.path.join(self.path, filename + '.jpg'))
        assert sample.mode == 'RGB'
        image = np.array(sample)

        assert image.dtype == np.uint8
        assert image.shape == (self.image_size, self.image_size, 3)

        image = self.transforms(image)

        if self.mode == 'test':
            return image
        else:
            return image, self.df.landmark_id.values[index]

    def __len__(self) -> int:
        return self.df.shape[0]

class AverageMeter(object):
    ''' Computes and stores the average and current value '''
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



def load_data(fold: int) -> Any:
    torch.multiprocessing.set_sharing_strategy('file_system')
    cudnn.benchmark = True

    print('config:')
    print(pprint.pformat(config))

    fname = f'{config.data.train_filename}_fold_{fold}_'
    train_df = pd.read_csv(os.path.join(config.data.data_dir, fname + 'train.csv'))
    val_df = pd.read_csv(os.path.join(config.data.data_dir, fname + 'val.csv'))
    print('train_df', train_df.shape, 'val_df', val_df.shape)

    val_df = pd.concat([c[1].iloc[:config.val.images_per_class]
                        for c in val_df.groupby('landmark_id')])
    print('val_df after class filtering', val_df.shape)

    test_df = pd.read_csv('../data/test.csv', dtype=str)
    test_df.drop(columns='url', inplace=True)
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



    train_dataset = ImageDataset(train_df, path=config.data.train_dir, mode='train',
                                 image_size=config.model.image_size,
                                 num_classes=config.model.num_classes,
                                 images_per_class=config.train.images_per_class,
                                 aug_type='albu', augmentor=transform_train)

    val_dataset = ImageDataset(val_df, path=config.data.train_dir, mode='val',
                               image_size=config.model.image_size,
                               num_classes=config.model.num_classes)

    test_dataset = ImageDataset(test_df, path=config.data.test_dir, mode='test',
                                image_size=config.model.image_size,
                                num_classes=config.model.num_classes)

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

def train(train_loader: Any, model: Any, criterion: Any, optimizer: Any,
          epoch: int, lr_scheduler: Any, lr_scheduler2: Any) -> None:
    print(f'epoch {epoch}')
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()

    num_steps = len(train_loader)
    if config.train.max_steps_per_epoch is not None:
        num_steps = min(len(train_loader), config.train.max_steps_per_epoch)

    print(f'total batches: {num_steps}')

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
            print(f'{epoch} [{i}/{num_steps}]\t'
                        f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'GAP {avg_score.val:.4f} ({avg_score.avg:.4f})'
                        + lr_str)

    print(f' * average GAP on train {avg_score.avg:.4f}')

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

    sample_sub.to_csv('submission.csv')

if __name__ == '__main__':

    model_conv = torchvision.models.resnet50(pretrained=True)
    model_conv.avg_pool = nn.AvgPool2d((5,10))
    model_conv.last_linear = nn.Linear(model_conv.last_linear.in_features, 5005)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.SGD(model_conv.parameters(), lr=0.01)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

