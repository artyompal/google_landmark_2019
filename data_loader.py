''' Data loaders for training & validation. '''

import math
import os
import pickle

from collections import defaultdict
from glob import glob
from typing import *

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from debug import dprint

SAVE_DEBUG_IMAGES = False
VERSION = os.path.splitext(os.path.basename(__file__))[0]


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, path: str, mode: str,
                 image_size: int, num_classes: int, images_per_class: int = 0,
                 aug_type: str = "albu", augmentor: Any = None) -> None:
        print(f'creating data loader {VERSION} - {mode}')
        assert mode in ['train', 'val', 'test']
        assert aug_type in ['albu', 'imgaug'] or augmentor is None

        self.df = dataframe
        self.path = path
        self.mode = mode
        self.image_size = image_size
        self.images_per_class = images_per_class
        self.num_classes = num_classes
        self.aug_type = aug_type
        self.augmentor = augmentor

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

        if self.augmentor is not None:
            if self.aug_type == 'albu':
                image = self.augmentor(image=image)['image']
            elif self.aug_type == 'imgaug':
                image = self.augmentor.augment_image(image)

        if SAVE_DEBUG_IMAGES:
            os.makedirs(f'../debug_images_{VERSION}/', exist_ok=True)
            Image.fromarray(image).save(f'../debug_images_{VERSION}/{index}.png')

        image = self.transforms(image)

        if self.mode == 'test':
            return image, ''
        else:
            return image, self.df.landmark_id.values[index]

    def __len__(self) -> int:
        return self.df.shape[0]
