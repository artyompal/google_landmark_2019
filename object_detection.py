#!/usr/bin/python3.6

import multiprocessing
import os
import pickle
import time

import numpy as np
import pandas as pd

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.utils.data import TensorDataset, DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
BATCH_SIZE = 16
NUM_WORKERS = multiprocessing.cpu_count()


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, mode: str) -> None:
        print(f'creating data loader - {mode}')
        assert mode in ['train', 'val', 'test']

        self.df = dataframe
        self.mode = mode

        transforms_list = [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                       std=[0.229, 0.224, 0.225]),
        ]
        self.transforms = transforms.Compose(transforms_list)

    def __getitem__(self, index: int) -> Any:
        ''' Returns: tuple (sample, target) '''
        filename = self.df.id.values[index]
        sample = Image.open('../data/test/' + filename + '.jpg')
        sample = sample.resize((800, 800), Image.LANCZOS)
        assert sample.mode == 'RGB'

        image = self.transforms(sample)

        if self.mode == 'test':
            return image
        else:
            return image, self.df.landmark_id.values[index]

    def __len__(self) -> int:
        return self.df.shape[0]

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    cudnn.benchmark = True

    test_df = pd.read_csv('../data/test.csv', dtype=str)
    print('test_df', test_df.shape)

    test_dataset = ImageDataset(test_df, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.cuda()

    model.eval()
    boxes, labels, scores = [], [], []

    with torch.no_grad():
        for i, input_ in enumerate(tqdm(test_loader, disable=IN_KERNEL)):
            output = model(input_.cuda())

            for item in output:
                boxes.append(item['boxes'].cpu().numpy())
                labels.append(item['labels'].cpu().numpy())
                scores.append(item['scores'].cpu().numpy())

    with open('obj_det.pkl', 'wb') as f:
        pickle.dump((boxes, labels, scores), f)
