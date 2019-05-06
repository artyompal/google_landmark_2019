#!/usr/bin/python3.6
""" For every image from test set, searches for the top-100 closes landmarks. """

import os
import sys

from glob import glob
from typing import *

import numpy as np
import faiss

from scipy.stats import describe
from tqdm import tqdm
from debug import dprint

K = 20
RESULTS_DIR = '../predicts'
USE_GPU = True
USE_COSINE_DIST = False

def search_against_fragment(train_features: np.ndarray, test_features: np.ndarray) \
    -> Tuple[np.ndarray, np.ndarray]:
    if USE_GPU:
        # build a flat index (CPU)
        if USE_COSINE_DIST:
            index_flat = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
        else:
            index_flat = faiss.IndexFlatL2(d)

        # make it into a GPU index
        index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    else:
        index_flat = faiss.IndexFlatIP(d)

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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f'usage: {sys.argv[0]} test_features.npy train_features_1.npy ...')
        sys.exit()

    test_fname = sys.argv[1]
    train_fnames = sys.argv[2:]

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    model_name = os.path.splitext(os.path.basename(test_fname))[0]
    model_name = model_name[5:] if model_name.startswith('test_') else model_name
    model_name = model_name[:-7] if model_name.endswith('_part00') else model_name
    result_fname = os.path.join(RESULTS_DIR, f'dist_{model_name}.npz')
    print("will save results to", result_fname)

    test_features = np.squeeze(np.load(test_fname))
    if USE_COSINE_DIST:
        test_features /= (np.linalg.norm(test_features, axis=1, keepdims=True) + 1e-8)

    print(test_features.shape)
    print(test_features)
    print("first vector:")
    print("shape", test_features[0].shape, "non-zeros", np.count_nonzero(test_features[0]))
    d = test_features[0].shape[0]

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
