#!/usr/bin/python3.6
''' Detects non-landmark images. '''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from cv2 import resize
from tqdm import tqdm
from vgg16_places_365 import VGG16_Places365

# Placeholders for predictions
p0, p1, p2 = [], [], []

# Places365 Model
model = VGG16_Places365(weights='places')
topn = 5

all_images = os.listdir('../data/test/')

# Loop through all images
for filename in tqdm(all_images):
    # Predict Top N Image Classes
    image = np.array(Image.open('../data/test/' + filename))
    image = image[16:-16, 16:-16]
    image = np.expand_dims(image, 0)
    topn_preds = np.argsort(model.predict(image)[0])[::-1][0:topn]

    p0.append(topn_preds[0])
    p1.append(topn_preds[1])
    p2.append(topn_preds[2])

# Create dataframe for later usage
topn_df = pd.DataFrame()
topn_df['filename'] = np.array(all_images)
topn_df['p0'] = np.array(p0)
topn_df['p1'] = np.array(p1)
topn_df['p2'] = np.array(p2)
topn_df.to_csv('topn_class_numbers.csv', index = False)

# Summary
print(topn_df.head())

# Read Class number, class name and class indoor/outdoor marker
class_information = pd.read_csv('../input/keras-vgg16-places365/categories_places365_extended.csv')
class_information.head()

# Set Class Labels
for col in ['p0', 'p1', 'p2']:
    topn_df[col + '_label'] = topn_df[col].map(class_information.set_index('class')['label'])
    topn_df[col + '_landmark'] = topn_df[col].map(class_information.set_index('class')['io'].replace({1:'non-landmark', 2:'landmark'}))
topn_df.to_csv('topn_all_info.csv', index = False)

# Summary
print(topn_df.head())

# Get 'landmark' images
n = 9
landmark_images =  topn_df[topn_df['p0_landmark'] == 'landmark']['filename']
landmark_indexes = landmark_images[:n].index.values

# Plot image examples
fig = plt.figure(figsize = (16, 16))
for index, im in zip(range(1, n+1), [ all_images[i] for i in landmark_indexes]):
    fig.add_subplot(3, 3, index)
    plt.title(im)
    plt.imshow(im)

# Get 'non-landmark' images
n = 9
landmark_images =  topn_df[topn_df['p0_landmark'] == 'non-landmark']['filename']
landmark_indexes = landmark_images[:n].index.values

# Plot image examples
fig = plt.figure(figsize = (16, 16))
for index, im in zip(range(1, n+1), [ all_images[i] for i in landmark_indexes]):
    fig.add_subplot(3, 3, index)
    plt.title(im)
    plt.imshow(im)
