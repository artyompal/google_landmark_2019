''' Extracts N most frequent classes and splits data set for K-fold cross-validation. '''
import argparse
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from debug import dprint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_folds', help='number of folds', type=int, default=5)
    parser.add_argument('--min_samples', help='minimum number of samples per class',
                        type=int, required=True)
    args = parser.parse_args()

    num_folds = int(args.num_folds)
    min_samples = int(args.min_samples)

    df = pd.read_csv('../data/train.csv')
    df.drop(columns='url', inplace=True)
    counts = df.landmark_id.value_counts()

    selected_classes = counts[counts >= min_samples].index
    print('classes with at least N samples:', selected_classes.shape[0])

    df = df.loc[df.landmark_id.isin(selected_classes)]
    dprint(df.shape)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=7)

    for i, (train_index, val_index) in enumerate(tqdm(skf.split(df.id, df.landmark_id),
                                                      total=num_folds)):
        df.iloc[train_index].to_csv(f'f{i}_smp_{min_samples}_train.csv', index=False)
        df.iloc[val_index].to_csv(f'f{i}_smp_{min_samples}_val.csv', index=False)
