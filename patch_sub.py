#!/usr/bin/python3.6
''' Patches the submission. '''

import pandas as pd


sub = pd.read_csv('best.csv')

# sub['landmarks'] = sub.landmarks.apply(lambda s: ' '.join(s.split()[:2]))
# sub.to_csv('best2.csv', index=False)


info = pd.read_csv('topn_all_info.csv')
# print(info.head())

sub['landmarks'][info.p0_landmark == 'non-landmark'] = ''
sub.to_csv('filtered.csv', index=False)
