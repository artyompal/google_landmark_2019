#!/usr/bin/python3.6
""" Makes predictions basing on distances, calculated by KNN. """

import os
import sys

from glob import glob
from typing import DefaultDict, List, Set
from collections import defaultdict

import numpy as np
import pandas as pd

# from sklearn.model_selection import train_test_split
from scipy.stats import describe
from tqdm import tqdm
from debug import dprint


def GAP(predicts: np.ndarray, confs: np.ndarray, targets: np.ndarray) -> float:
    ''' Computes GAP@1 '''
    if len(targets.shape) != 1:
        dprint(targets.shape)
        assert False

    assert predicts.shape == confs.shape
    indices = np.argsort(-confs)

    res, true_pos = 0.0, 0
    num_predicts_per_sample = confs.shape[1]
    num_targets = len(targets)

    predicts = predicts.flatten()
    confs = confs.flatten()
    targets = np.repeat(targets.reshape(-1, 1), num_predicts_per_sample,  axis=1)
    targets = targets.flatten()

    dprint(predicts)
    dprint(targets)

    sorting_idx = np.argsort(-confs)
    predicts = predicts[sorting_idx]
    targets = targets[sorting_idx]

    for i, (p, t) in enumerate(zip(predicts, targets)):
        rel = int(p == t)
        true_pos += rel

        res += true_pos / (i + 1) * rel

    res /= num_targets # TODO: incorrect, not all test images depict landmarks
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

    # load distances info
    dist_file = np.load(sys.argv[2], allow_pickle=True)
    distances, indices = dist_file['distances'], dist_file['indices']
    dprint(distances.shape)
    dprint(indices.shape)
    dprint(np.max(indices.flatten()))

    if not predict_test:
        distances = np.delete(distances, 0, axis=1)
        indices = np.delete(indices, 0, axis=1)

    # load dataframe
    full_train_df = pd.read_csv('../data/train.csv')
    full_train_df.drop(columns='url', inplace=True)
    train_df = pd.read_csv('../data/splits/10_samples_92740_classes_fold_0_train.csv')
    val_dataset_mask = ~full_train_df.id.isin(train_df.id)
    knn_train_df = full_train_df.loc[val_dataset_mask]
    dprint(knn_train_df.shape)

    if predict_test:
        df = pd.read_csv('../data/test.csv')
        df = df.loc[df.id.apply(lambda img: os.path.exists(os.path.join(
            f'../data/test/{img}.jpg')))]
        print('test df after filtering', df.shape)
    else:
        df = knn_train_df

    # make a prediction about classes
    num_predicts = 1 # indices.shape[1]
    predicts = np.zeros((len(df), num_predicts), dtype=int)
    confs = np.zeros((len(df), num_predicts))

    for i in range(num_predicts):
        idx = indices[:, i]
        dprint(describe(idx))
        dprint(idx)
        # dprint(np.sort(idx))

        if predict_test:
            predicts[:, i] = full_train_df.iloc[idx, 1]
        else:
            # WTF? Why are we not using same indices?
            predicts[:, i] = knn_train_df.iloc[idx, 1]

        confs[:, i] = distances[:, i]

    # print('calculating class centroids')
    # for i in tqdm(range(confs.shape[0])):
    #     preds: DefaultDict[int, List[float]] = defaultdict(list)
    #
    #     for pred, conf in zip(predicts[i], confs[i]):
    #         preds[pred].append(conf)
    #
    #     for j, (pred, conf) in enumerate(preds.items()):
    #         predicts[i, j] = pred
    #         confs[i, j] = np.mean(conf)
    #
    #     predicts[i, len(preds):] = -1
    #     confs[i, len(preds):] = 0

    if not predict_test:
        print('calculating GAP')
        gap = GAP(predicts, confs, knn_train_df.landmark_id.values)
        dprint(gap)
    else:
        print('generating submission')
        sub = df
        combine = lambda t: " ".join(f'{lm} {conf}' for lm, conf in zip(t[0], t[1]) if conf)
        sub['landmarks'] = list(map(combine, zip(predicts, confs)))

        sample_sub = pd.read_csv('../data/recognition_sample_submission.csv')
        sample_sub = sample_sub.set_index('id')
        sub = sub.set_index('id')
        sample_sub.update(sub)

        name = os.path.splitext(os.path.basename(sys.argv[2]))[0]
        sample_sub.to_csv(f'{name}.csv')
