''' Extracts N most frequent classes and splits data set for K-fold cross-validation. '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_folds', help='number of folds', type=int, default=5)
    parser.add_argument('--min_samples', help='minimum number of samples per class',
                        type=int, required=True)
