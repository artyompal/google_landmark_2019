#!/usr/bin/python3.6

import sys
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from debug import dprint


assert len(sys.argv) == 2
sub = pd.read_csv(sys.argv[1])

fig = plt.figure(figsize = (16, 16))
N = 5
samples = sub.loc[~sub.landmarks.isnull()].sample(N)

train_df = pd.read_csv('../data/train.csv')
originals = train_df.groupby('landmark_id').first()
originals = originals.id.values
dprint(originals.shape)
dprint(originals)

for i in range(samples.shape[0]):
    img, predict = samples.iloc[i]
    dprint(img)
    dprint(predict)

    cls_ = int(predict.split()[0])
    orig = originals[cls_]
    dprint(orig)

    fig.add_subplot(2, N, i + 1)
    plt.title(img)
    plt.imshow(Image.open('../data/test/' + img + '.jpg'))

    fig.add_subplot(2, N, i + 1 + N)
    plt.title(orig)
    plt.imshow(Image.open('../data/train/' + orig + '.jpg'))

plt.show()
