''' Sampler object for PyTorch. Returns fixed number of samples per class. '''

from typing import Any, List

import numpy as np
import pandas as pd
import torch.utils.data
import numpy.random

class BalancingSampler(torch.utils.data.Sampler):
    def __init__(self, df: pd.DataFrame, images_per_class: int,
                 num_classes: int) -> None:
        self.df = df
        self.images_per_class = images_per_class
        self.num_classes = num_classes

    def __iter__(self) -> Any:
        ''' Returns: iterator '''
        indices: List[int] = []

        for class_ in self.df.groupby('landmark_id'):
            df = class_[1]

            if df.shape[0] >= self.images_per_class:
                indices.extend(numpy.random.choice(df.index, self.images_per_class,
                                                   replace=False))
            else:
                indices.extend(df.index)
                indices.extend(numpy.random.choice(df.index, self.images_per_class
                                                   - df.shape[0], replace=True))

        assert len(indices) == len(self)
        return iter(np.random.permutation(indices))

    def __len__(self) -> int:
        return self.images_per_class * self.num_classes
