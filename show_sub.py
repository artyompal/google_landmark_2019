#!/usr/bin/python3.6

import sys
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image


assert len(sys.argv) == 2
sub = pd.read_csv(sys.argv[1])

fig = plt.figure(figsize = (16, 16))
N = 9
landmarks = sub.loc[sub.landmarks != ''].sample(N).id

for i, im in enumerate(landmarks):
    print(im)
    fig.add_subplot(3, 3, i + 1)
    plt.title(im)
    plt.imshow(Image.open('../data/test/' + im + '.jpg'))
plt.show()
