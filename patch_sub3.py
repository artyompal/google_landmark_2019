#!/usr/bin/python3.6
''' Patches the submission. '''

import json
import pickle
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm
from debug import dprint


if len(sys.argv) != 3:
    print(f'usage: {sys.argv[0]} dest.csv source.csv')
    sys.exit()

sub = pd.read_csv(sys.argv[2])

with open('imagenet1000.txt') as f:
    imagenet = eval(f.read())

categories = list(imagenet.values())

with open('imagenet_classes.pkl', 'rb') as ff:
    predicts = pickle.load(ff)
    predicts = np.vstack(predicts)

classes = np.argmax(predicts, axis=1)

imagenet_classes = [categories[classes[i]] for i in range(predicts.shape[0])]
imagenet_indices = [classes[i] for i in range(predicts.shape[0])]
confs = [predicts[i, classes[i]] for i in range(predicts.shape[0])]

MIN_CONF = 0.7

for i in tqdm(range(sub.shape[0])):
    index = sub.index.values[i]

    class_, conf = imagenet_classes[index], confs[index]

    if conf < MIN_CONF:
        continue

    for c in ['warplane', 'coil', 'missile', 'conch', 'gar', 'tank',
              'schooner', 'book jacket', 'scabbard', 'aircraft carrier',
              'school bus', 'space shuttle', 'cannon',
              'trilobite', 'tow truck', 'submarine', 'pickup', 'amphibian',
              'marmot', 'mushroom', 'shield', 'French loaf',
              'poncho', 'warthog']:
        if class_.startswith(c + ','):
            sub.landmarks.iloc[i] = ''

    # for c in ['warplane', 'coil', 'missile', 'conch', 'gar', 'tank',
    #           'schooner', 'book jacket', 'scabbard', 'aircraft carrier',
    #           'school bus', 'trolley bus', 'space shuttle', 'cannon',
    #           'trilobite', 'tow truck', 'submarine', 'pickup', 'amphibian',
    #           'marmot', 'mushroom', 'passenger car', 'shield', 'French loaf',
    #           'poncho', 'warthog']:
    #     if class_.startswith(c + ','):
    #         sub.landmarks.iloc[i] = ''

    if imagenet_indices[index] < 400:
        sub.landmarks.iloc[i] = ''

    if imagenet_indices[index] >= 985:
        sub.landmarks.iloc[i] = ''


sub.to_csv(sys.argv[1], index=False)
