#!/usr/bin/python3.6
''' Patches the submission. '''

import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm


sub = pd.read_csv('best.csv')

with open('obj_det.pkl', 'rb') as f:
    boxes, labels, confs = pickle.load(f)

categories = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic', 'light', 'fire', 'hydrant', 'N/A', 'stop',
    'sign', 'parking', 'meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports', 'ball',
    'kite', 'baseball', 'bat', 'baseball', 'glove', 'skateboard', 'surfboard', 'tennis',
    'racket', 'bottle', 'N/A', 'wine', 'glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot', 'dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted', 'plant', 'bed', 'N/A', 'dining', 'table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell',
    'phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy', 'bear', 'hair', 'drier', 'toothbrush'
]


for i in tqdm(range(sub.shape[0])):
    index = sub.index.values[i]

    PERSON_MIN_CONF = 0.5
    PERSON_MIN_AREA = 0.4
    CAR_MIN_CONF = 0.5
    CAR_MIN_AREA = 0.4

    total_persons_area = 0.0
    persons_count = 0

    for L, conf, box in zip(labels[index], confs[index], boxes[index]):
        box /= 800.0
        area = (box[2] - box[0]) * (box[3] - box[1])
        class_ = categories[L]

        if class_ == 'person' and conf > PERSON_MIN_CONF:
            total_persons_area += area
            persons_count += 1

        if class_ == 'car' and conf > CAR_MIN_CONF and area > CAR_MIN_AREA:
            sub.landmarks.iloc[i] = ''

        # if class_ == 'airplane' and conf > CAR_MIN_CONF and area > CAR_MIN_AREA:
        #     sub.landmarks.iloc[i] = ''

    if total_persons_area > PERSON_MIN_AREA:
        sub.landmarks.iloc[i] = ''


sub.to_csv('filtered.csv', index=False)
