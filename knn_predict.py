#!/usr/bin/python3.6
""" For every image from test set, searches for the top-100 closes landmarks. """

import os
import sys

from glob import glob
from typing import *

import numpy as np
import pandas as pd

# from sklearn.model_selection import train_test_split
from scipy.stats import describe
from tqdm import tqdm
from debug import dprint

K = 20
DIMS = 2048


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} <distances.npz>')
        sys.exit()

    ''' Algorithm:
    1. define level-2 train and validation sets
    2. for every sample from validation set, find K nearest samples from the train set
    3. make a prediction about classes
    4. calculate the metric
    5. take full train set
    6. for every sample from the test set, find K nearest samples from the full train set
    7. make a prediction
    8. generate submission
    '''

    # 1. for every sample from validation set, find K nearest samples from the train set
    dist_file = np.load(sys.argv[1], allow_pickle=True)
    distances, indices = dist_file['distances'], dist_file['indices']
    dprint(distances.shape)
    dprint(indices.shape)

    if distances.shape[1] == K:
        distances = np.delete(distances, 0, axis=1)
        dprint(distances.shape)

    dprint(np.max(indices.flatten()))

    # 2. define level-2 train and validation sets
    full_train_df = pd.read_csv('../data/train.csv')
    train_df = pd.read_csv('../data/splits/50_samples_18425_classes_fold_0_train.csv')
    train_mask = ~full_train_df.id.isin(train_df.id)
    knn_train_df = full_train_df.loc[train_mask]
    dprint(knn_train_df.shape)

    # 3. make a prediction about classes

    # 4. calculate the metric

    # 5. take full train set

    # 6. for every sample from the test set, find K nearest samples from the full train set

    # 7. make a prediction

    # 8. generate submission
