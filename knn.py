#!/usr/bin/python3.6
""" For every image from test set, searches for the top-100 closes landmarks. """

import os
import sys

from glob import glob
from typing import *

import numpy as np
import pandas as pd
import faiss

from sklearn.model_selection import train_test_split
from scipy.stats import describe
from tqdm import tqdm
from debug import dprint

K = 20
KNN_PREDICTS_DIR = '../predicts'
USE_GPU = True
USE_COSINE_DIST = False
DIMS = 2048

def search_against_fragment(train_features: np.ndarray, test_features: np.ndarray) \
    -> Tuple[np.ndarray, np.ndarray]:
    if USE_GPU:
        # build a flat index (CPU)
        if USE_COSINE_DIST:
            index_flat = faiss.IndexFlat(DIMS, faiss.METRIC_INNER_PRODUCT)
        else:
            index_flat = faiss.IndexFlatL2(DIMS)

        # make it into a GPU index
        index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    else:
        index_flat = faiss.IndexFlatIP(DIMS)

    index_flat.add(train_features)
    print("total size of index:", index_flat.ntotal)

    # print("sanity search...")
    # distances, index = index_flat.search(train_features[:10], K)  # actual search
    # print(index[:10])
    # print(distances[:10])

    print("searching")
    distances, index = index_flat.search(test_features, K)  # actual search
    dprint(index)
    dprint(distances)
    dprint(describe(distances.flatten()))
    return index, distances

def merge_results(index1: np.ndarray, distances1: np.ndarray, index2: np.ndarray,
                  distances2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns top-K of two sets. """
    print("merging results")
    assert index1.shape == distances1.shape and index2.shape == distances2.shape
    assert index1.shape[1] == index2.shape[1]

    joint_index = np.hstack((index1, index2))
    joint_distances = np.hstack((distances1, distances2))
    print("joint_index", joint_index.shape, "joint_distances", joint_distances.shape)
    assert joint_index.shape == joint_distances.shape

    best_indices = np.zeros((index1.shape[0], K), dtype=object)
    best_distances = np.zeros((index1.shape[0], K), dtype=np.float32)

    for sample in range(joint_index.shape[0]):
        closest_indices = np.argsort(joint_distances[sample])[:K]
        best_indices[sample] = joint_index[sample, closest_indices]
        best_distances[sample] = joint_distances[sample, closest_indices]

    print("best_indices", best_indices.shape, "best_distances", best_distances.shape)
    dprint(best_indices)
    dprint(best_distances)
    dprint(describe(best_distances))
    return best_indices, best_distances

'''
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f'usage: {sys.argv[0]} test_features.npy train_features_1.npy ...')
        sys.exit()

    test_fname = sys.argv[1]
    train_fnames = sys.argv[2:]

    if not os.path.exists(KNN_PREDICTS_DIR):
        os.makedirs(KNN_PREDICTS_DIR)

    model_name = os.path.splitext(os.path.basename(test_fname))[0]
    model_name = model_name[5:] if model_name.startswith('test_') else model_name
    model_name = model_name[:-7] if model_name.endswith('_part00') else model_name
    result_fname = os.path.join(KNN_PREDICTS_DIR, f'dist_{model_name}.npz')
    print("will save results to", result_fname)

    test_features = np.squeeze(np.load(test_fname))
    if USE_COSINE_DIST:
        test_features /= (np.linalg.norm(test_features, axis=1, keepdims=True) + 1e-8)

    print(test_features.shape)
    print(test_features)
    print("first vector:")
    print("shape", test_features[0].shape, "non-zeros", np.count_nonzero(test_features[0]))

    if USE_GPU:
        print("initializing CUDA")
        res = faiss.StandardGpuResources()

    best_index, best_distance = None, None
    offset = 0
    for fragment in tqdm(train_fnames):
        train_features = np.load(fragment)
        train_features = train_features.reshape(train_features.shape[0], -1)

        if USE_COSINE_DIST:
            train_features /= (np.linalg.norm(train_features, axis=1, keepdims=True) + 1e-8)

        print("features shape", train_features.shape)
        idx, dist = search_against_fragment(train_features, test_features)
        idx += offset
        offset += train_features.shape[0]

        if best_index is None:
            best_index, best_distances = idx, dist
        else:
            best_index, best_distances = merge_results(best_index, best_distances, idx, dist)

    print('best_index')
    print(best_index.shape)
    print(best_index)
    print('best_distances')
    print(best_distances.shape)
    print(best_distances)

    print("writing results to", result_fname)
    np.savez(result_fname, indices=best_index, distances=best_distances)
'''

def load_features(filename: str) -> np.ndarray:
    features = np.load(filename)
    features = features.reshape(features.shape[0], -1)

    if USE_COSINE_DIST:
        features /= (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    return features

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f'usage: {sys.argv[0]} test_features.npy train_features_1.npy ...')
        sys.exit()

    test_fname = sys.argv[1]
    train_fnames = sys.argv[2:]

    if not os.path.exists(KNN_PREDICTS_DIR):
        os.makedirs(KNN_PREDICTS_DIR)

    model_name = os.path.splitext(os.path.basename(test_fname))[0]
    model_name = model_name[5:] if model_name.startswith('test_') else model_name
    model_name = model_name[:-7] if model_name.endswith('_part00') else model_name
    result_fname = os.path.join(KNN_PREDICTS_DIR, f'dist_{model_name}.npz')
    print("will save results to", result_fname)


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

    # 1. define level-2 train and validation sets

    if len(sys.argv) < 3:
        print(f'usage: {sys.argv[0]} test_features.npy train_features_1.npy ...')
        sys.exit()

    test_fname = sys.argv[1]
    train_fnames = sys.argv[2:]

    if not os.path.exists(KNN_PREDICTS_DIR):
        os.makedirs(KNN_PREDICTS_DIR)

    full_train_df = pd.read_csv('../data/train.csv')
    # level1_val = pd.read_csv('../data/splits/50_samples_18425_classes_fold_0_val.csv')
    # val_df = full_train_df.loc[~train_df.id.isin(train_df.id)]

    level2_train = pd.read_csv('../data/splits/under_50_samples_fold_0_train.csv')
    level2_val = pd.read_csv('../data/splits/under_50_samples_fold_0_val.csv')

    # level2_full_df = full_train_df.loc[~full_train_df.id.isin(level1_val)]
    # dprint(level2_full_df.shape)
    # train_df, val_df  = train_test_split(level2_full_df, shuffle=True, random_state=0)

    # train_df = full_train_df.loc[full_train_df.id.isin(level2_train.id)]
    # val_df = full_train_df.loc[full_train_df.id.isin(level2_val.id)]

    dprint(level2_train.shape)
    dprint(level2_val.shape)

    train_bitmask = full_train_df.id.isin(level2_train.id)
    val_bitmask = full_train_df.id.isin(level2_val.id)
    dprint(sum(train_bitmask))
    dprint(sum(val_bitmask))


    # 2. for every sample from validation set, find K nearest samples from the train set

    # if USE_GPU:
    #     print("initializing CUDA")
    #     res = faiss.StandardGpuResources()

    val_offset = 0
    total_val_samples = 0

    for val_frag_idx, val_fragment in enumerate(train_fnames):
        val_features = load_features(val_fragment)
        fragment_size = val_features.shape[0]
        mask = val_bitmask[val_offset : val_offset + fragment_size]
        dprint(val_features.shape)
        dprint(sum(mask))

        val_features = val_features[mask]
        # dprint(val_features.shape)

        total_val_samples += val_features.shape[0]
        val_offset += fragment_size

        # best_index, best_distance = None, None
        # train_offset = 0
        #
        # for train_fragment in tqdm(train_fnames):
        #     train_features = load_features(train_fragment)
        #
        #     idx, dist = search_against_fragment(train_features, val_features)
        #     idx += train_offset
        #     train_offset += train_features.shape[0]
        #
        #     if best_index is None:
        #         best_index, best_distances = idx, dist
        #     else:
        #         best_index, best_distances = merge_results(best_index, best_distances, id)

    dprint(total_val_samples)

    # 3. make a prediction about classes

    # 4. calculate the metric

    # 5. take full train set

    # 6. for every sample from the test set, find K nearest samples from the full train set

    # 7. make a prediction

    # 8. generate submission
