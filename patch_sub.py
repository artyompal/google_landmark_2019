#!/usr/bin/python3.6
''' Patches the submission. '''

import sys
import pandas as pd


if len(sys.argv) != 3:
    print(f'usage: {sys.argv[0]} dest.csv source.csv')
    sys.exit()

sub = pd.read_csv(sys.argv[2])
info = pd.read_csv('topn_all_info.csv')

sub['landmarks'][(info.p0_landmark == 'non-landmark') |
                 (info.p1_landmark == 'non-landmark') |
                 (info.p2_landmark == 'non-landmark')] = ''
sub.to_csv(sys.argv[1], index=False)
