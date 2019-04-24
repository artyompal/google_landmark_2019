>>> import pandas as pd
>>> from scipy.stats import describe

>>> df = pd.read_csv('train.csv')
>>> df.columns
Index(['id', 'url', 'landmark_id'], dtype='object')
>>> df.shape
(4132914, 3)
>>> df.head()
                 id                                                url  landmark_id
0  6e158a47eb2ca3f6  https://upload.wikimedia.org/wikipedia/commons...       142820
1  202cd79556f30760  http://upload.wikimedia.org/wikipedia/commons/...       104169
2  3ad87684c99c06e1  http://upload.wikimedia.org/wikipedia/commons/...        37914
3  e7f70e9c61e66af3  https://upload.wikimedia.org/wikipedia/commons...       102140
4  4072182eddd0100e  https://upload.wikimedia.org/wikipedia/commons...         2474
>>>
>>>

>>> df.shape
(4132914, 3)

>>> df.landmark_id.unique().shape
(203094,)

>>> describe(counts)
DescribeResult(nobs=203094, minmax=(1, 10247), mean=20.34975922479246, variance=2742.1996287573975, skewness=50.407900932571934, kurtosis=7753.8331819825535)
>>> sum(counts > 10)
86804
>>> sum(counts > 10) / counts.shape[0]
0.4274079982668124
>>>

>>>
>>> sum(counts >= 9) / counts.shape[0]
0.48914295843304084
>>> sum(counts >= 8) / counts.shape[0]
0.526829940815583
>>> sum(counts >= 7) / counts.shape[0]
0.5699577535525422
>>> sum(counts >= 6) / counts.shape[0]
0.6190778654219229


>>> sum(counts > 4) / counts.shape[0]
0.6752193565541079
>>> sum(counts > 3) / counts.shape[0]
0.7418190591548741
>>> sum(counts > 2) / counts.shape[0]
0.8196352427939771
>>> sum(counts > 1) / counts.shape[0]
0.906969186682029
>>>

