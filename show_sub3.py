#!/usr/bin/python3.6

import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from debug import dprint


with open('imagenet1000.json') as f:
    imagenet = json.load(f)

categories = list(imagenet.keys())

with open('image.pkl', 'rb') as f:
    boxes, labels, confs = pickle.load(f)

assert len(sys.argv) == 2
sub = pd.read_csv(sys.argv[1])

fig = plt.figure(figsize = (16, 16))
N = 9
landmarks = sub.loc[~sub.landmarks.isnull()].sample(N)

print(landmarks)
print(landmarks.index)
print(landmarks.id)


for i in range(landmarks.shape[0]):
    print('-' * 80)

    index = landmarks.index.values[i]
    img = landmarks.id.values[i]

    dprint(index)
    dprint(img)

    # MIN_CONF = 0.5
    # for L, conf, box in zip(labels[index], confs[index], boxes[index]):
    #     if conf > MIN_CONF:
    #         box /= 800.0
    #         area = (box[2] - box[0]) * (box[3] - box[1])
    #         print(categories[L], conf, area)

    PERSON_MIN_CONF = 0.5
    PERSON_MIN_AREA = 0.5
    CAR_MIN_CONF = 0.5
    CAR_MIN_AREA = 0.5

    total_persons_area = 0.0
    persons_count = 0
    total_cars_area = 0.0
    cars_count = 0

    objects = np.unique([f'{categories[L]} ({conf:.02})' for L, conf, box in
                         zip(labels[index], confs[index], boxes[index])
                         if conf > 0.5])
    print('objects', objects)

    for L, conf, box in zip(labels[index], confs[index], boxes[index]):
        box /= 800.0
        area = (box[2] - box[0]) * (box[3] - box[1])
        class_ = categories[L]

        if class_ == 'person' and conf > PERSON_MIN_CONF:
            total_persons_area += area
            persons_count += 1

        if class_ == 'car' and conf > CAR_MIN_CONF:
            total_cars_area += area
            cars_count += 1

    print('total_persons_area', total_persons_area)
    print('persons_count', persons_count)
    print('total_cars_area', total_cars_area)
    print('cars_count', cars_count)
    print('prediction', landmarks.landmarks.values[i])

    fig.add_subplot(3, 3, i + 1)
    plt.title(img)
    plt.imshow(Image.open('../data/test/' + img + '.jpg'))

plt.show()
