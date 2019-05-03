#!/usr/bin/python3.6

import pandas as pd
from scipy.stats import describe
from debug import dprint

df = pd.read_csv('../data/train.csv')
dprint(df.columns)
dprint(df.shape)
dprint(df.head())
dprint(df.landmark_id.unique().shape)

counts = df.landmark_id.value_counts()
dprint(describe(counts))

for val in range(1, 10):
    print(f'val={val}')
    dprint(sum(counts == val))
    dprint(sum(counts == val) / counts.shape[0])

for val in [5, 10, 20, 50, 100]:
    print(f'val={val}')
    dprint(sum(counts >= val))
    dprint(sum(counts >= val) / counts.shape[0])

