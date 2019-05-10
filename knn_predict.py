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


def GAP(predicts: np.ndarray, confs: np.ndarray, targets: np.ndarray) -> float:
    ''' Computes GAP@1 '''

    # FIXME: handle more than one prediction
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
    indices = np.argsort(-confs)

    res, true_pos = 0.0, 0

    for i, (c, p, t) in enumerate(zip(confs, predicts, targets)):
        rel = int(p == t)
        true_pos += rel

        res += true_pos / (i + 1) * rel

    res /= targets.shape[0] # FIXME: incorrect, not all test images depict landmarks
    return res

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ['--val', '--test']:
        print(f'usage: {sys.argv[0]} --val|--test <distances.npz>')
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

    predict_test = sys.argv[1] == '--test'

    # 1. for every sample from validation set, find K nearest samples from the train set
    dist_file = np.load(sys.argv[2], allow_pickle=True)
    distances, indices = dist_file['distances'], dist_file['indices']
    dprint(distances.shape)
    dprint(indices.shape)
    dprint(np.max(indices.flatten()))

    # 2. define level-2 train and validation sets
    full_train_df = pd.read_csv('../data/train.csv')
    full_train_df.drop(columns='url', inplace=True)
    train_df = pd.read_csv('../data/splits/50_samples_18425_classes_fold_0_train.csv')
    train_mask = ~full_train_df.id.isin(train_df.id)
    knn_train_df = full_train_df.loc[train_mask]
    # knn_train_df = knn_train_df.iloc[:282363]
    dprint(knn_train_df.shape)

    # 3. make a prediction about classes
    predicts = []
    for i, (_, id, landmark_id) in enumerate(tqdm(knn_train_df.itertuples(),
                                                  total=knn_train_df.shape[0])):
        closest_id = indices[i, 0]
        predict = knn_train_df.iloc[closest_id, 1]
        predicts.append(predict)

    # 4. calculate the metric
    predicts = np.array(predicts)
    gap = GAP(predicts, np.ones_like(predicts), knn_train_df.landmark_id)
    dprint(gap)

    # 8. generate submission
