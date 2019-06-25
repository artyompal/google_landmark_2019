#!/usr/bin/python3.6
""" For every image from the test set, searches for the top-20 closest images from the train set. """

import os
import sys

from glob import glob
from typing import Iterator, Iterable, List, Optional, Tuple

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
USE_COSINE_DIST = True
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
    print("total size of the database:", index_flat.ntotal)

    # print("sanity search...")
    # distances, index = index_flat.search(train_features[:10], K)  # actual search
    # print(index[:10])
    # print(distances[:10])

    print("searching")
    distances, index = index_flat.search(test_features, K)  # actual search
    dprint(index)
    dprint(distances)
    dprint(describe(index.flatten()))
    dprint(describe(distances.flatten()))
    return index, distances

def merge_results(index1: np.ndarray, distances1: np.ndarray, index2: np.ndarray,
                  distances2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns top-K of two sets. """
    print("merging results")
    assert index1.shape == distances1.shape and index2.shape == distances2.shape
    assert index1.shape[1] == index2.shape[1]

    joint_indices = np.hstack((index1, index2))
    joint_distances = np.hstack((distances1, distances2))
    print("joint_indices", joint_indices.shape, "joint_distances", joint_distances.shape)
    assert joint_indices.shape == joint_distances.shape

    best_indices = np.zeros((index1.shape[0], K), dtype=int)
    best_distances = np.zeros((index1.shape[0], K), dtype=np.float32)

    for sample in range(joint_indices.shape[0]):
        if not USE_COSINE_DIST:
            closest_indices = np.argsort(joint_distances[sample])
        else:
            closest_indices = np.argsort(-joint_distances[sample])

        closest_indices = closest_indices[:K]
        best_indices[sample] = joint_indices[sample, closest_indices]
        best_distances[sample] = joint_distances[sample, closest_indices]

    print("best_indices", best_indices.shape, "best_distances", best_distances.shape)
    dprint(best_indices)
    dprint(best_distances)
    dprint(describe(best_indices.flatten()))
    return best_indices, best_distances

def load_features(filename: str) -> np.ndarray:
    features = np.load(filename)
    features = features.reshape(features.shape[0], -1)

    if USE_COSINE_DIST:
        features /= (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    return features


class DatasetIter(Iterator[np.ndarray]):
    ''' Iterator which remembers previous position. '''
    def __init__(self, dataset_parts: 'DatasetParts') -> None:
        self.files = dataset_parts.files
        self.train_files = iter(dataset_parts.files)
        # self.subset_mask = dataset_parts.subset_mask
        self.base_offset = 0

    def __next__(self) -> np.ndarray:
        features = load_features(next(self.train_files))
        fragment_size = features.shape[0]

        # if self.subset_mask is not None:
        #     part_mask = self.subset_mask[self.base_offset : self.base_offset + fragment_size]
        #     features = features[part_mask]

        self.base_offset += fragment_size
        return features

class DatasetParts(Iterable[DatasetIter]):
    ''' A collection that reads features by parts. '''
    def __init__(self, df: pd.DataFrame, # subset_mask: Optional[pd.Series],
                 files: List[str]) -> None:
        self.df = df
        self.files = files
        # self.subset_mask = subset_mask

    def __iter__(self) -> DatasetIter:
        return DatasetIter(self)

    def __len__(self) -> int:
        return len(self.files)


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] not in ['--train', '--test']:
        print(f'usage: {sys.argv[0]} --train train_features_1.npy ...')
        print(f'or {sys.argv[0]} --test test_features.npy train_features_1.npy ...')
        sys.exit()

    predict_test = sys.argv[1] == '--test'
    test_fname = sys.argv[2] if predict_test else ''
    train_fnames = sys.argv[3:] if predict_test else sys.argv[2:]

    if not os.path.exists(KNN_PREDICTS_DIR):
        os.makedirs(KNN_PREDICTS_DIR)

    model_name = os.path.splitext(os.path.basename(train_fnames[0]))[0]
    assert model_name.startswith('train_')
    assert model_name.endswith('_part00')
    model_name = model_name[6:-7]

    dataset = 'test' if predict_test else 'train'
    type = 'cosine' if USE_COSINE_DIST else 'euclidean'
    result_fname = os.path.join(KNN_PREDICTS_DIR,
                                f'dist_{model_name}_{dataset}_{type}.npz')
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

    if not os.path.exists(KNN_PREDICTS_DIR):
        os.makedirs(KNN_PREDICTS_DIR)

    full_train_df = pd.read_csv('../data/train.csv')

    # # train_df = pd.read_csv('../data/splits/50_samples_18425_classes_fold_0_train.csv')
    # train_df = pd.read_csv('../data/splits/10_samples_92740_classes_fold_0_train.csv')
    # train_mask = ~full_train_df.id.isin(train_df.id)
    # dprint(train_mask.shape)
    # dprint(sum(train_mask))

    if predict_test:
        test_df = pd.read_csv('../data/test.csv')


    # 2. for every sample from validation set, find K nearest samples from the train set

    if USE_GPU:
        print("initializing CUDA")
        res = faiss.StandardGpuResources()

    train_dataset_parts = DatasetParts(full_train_df, train_fnames)
    test_dataset_parts = DatasetParts(test_df, [test_fname]) \
                            if predict_test else train_dataset_parts
    total_best_indices, total_best_distances = None, None

    for i, val_features in enumerate(tqdm(test_dataset_parts, disable=predict_test)):
        print('=' * 100)
        print('iteration', i)

        best_indices, best_distances = None, None
        base_index = 0

        for train_features in tqdm(train_dataset_parts, disable=not predict_test):
            print('-' * 100)
            idx, dist = search_against_fragment(train_features, val_features)
            idx += base_index
            dprint(idx.shape)
            dprint(dist.shape)

            if best_indices is None:
                best_indices, best_distances = idx, dist
            else:
                best_indices, best_distances = merge_results(best_indices, best_distances, idx, dist)

            base_index += train_features.shape[0]

        # best_indices = np.delete(best_indices, 0, axis=1)
        # best_distances = np.delete(best_distances, 0, axis=1)

        dprint(best_indices.shape)
        dprint(best_indices)
        dprint(best_distances.shape)
        dprint(best_distances)

        if total_best_indices is None:
            total_best_indices, total_best_distances = best_indices, best_distances
        else:
            total_best_indices = np.vstack((total_best_indices, best_indices))
            total_best_distances = np.vstack((total_best_distances, best_distances))

    print("writing results to", result_fname)
    dprint(total_best_indices)
    dprint(total_best_distances)
    np.savez(result_fname, indices=total_best_indices, distances=total_best_distances)
