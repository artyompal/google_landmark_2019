#!/usr/bin/python3.6

import json
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from scipy.stats import describe

from debug import dprint


with open('imagenet1000.txt') as f:
    imagenet = eval(f.read())

categories = list(imagenet.values())

with open('imagenet_classes.pkl', 'rb') as ff:
    predicts = pickle.load(ff)
    predicts = np.vstack(predicts)

dprint(predicts.shape)
dprint(describe(predicts.flatten()))

classes = np.argmax(predicts, axis=1)
# confs = predicts[:, classes] # this hangs my PC
dprint(classes)


imagenet_classes = [categories[classes[i]] for i in tqdm(range(predicts.shape[0]))]
confs = [predicts[i, classes[i]] for i in tqdm(range(predicts.shape[0]))]


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

    MIN_CONF = 0.4
    if confs[index] > MIN_CONF:
        print(index, img, imagenet_classes[index], confs[index])

    fig.add_subplot(3, 3, i + 1)
    plt.title(f'{imagenet_classes[index]} ({confs[index]:.1})')
    plt.imshow(Image.open('../data/test/' + img + '.jpg'))

plt.show()
