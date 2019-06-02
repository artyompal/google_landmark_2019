#!/usr/bin/python3.6
''' Patches the submission. '''

import pandas as pd


sub = pd.read_csv('best.csv')
info = pd.read_csv('topn_all_info.csv')

sub['landmarks'][(info.p0_landmark == 'non-landmark') |
                 (info.p1_landmark == 'non-landmark') |
                 (info.p2_landmark == 'non-landmark')] = ''
sub.to_csv('filtered.csv', index=False)
