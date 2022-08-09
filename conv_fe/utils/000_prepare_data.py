# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import os
import pandas as pd
import numpy as np

data = np.load('../xydata/raw/phase1_gdata.npz')

train_test_df = pd.DataFrame(data['x'])
train_test_df = train_test_df.add_prefix('feature_')
label_df = pd.DataFrame(data['y'], columns=['target'])
train_test_df = pd.concat([train_test_df, label_df], axis=1)
train_test_df['user_id'] = range(len(train_test_df))

train_df = train_test_df.loc[data['train_mask']].sample(frac=1.0).reset_index(drop=True)
#train_df always need to shuffle?
test_df = train_test_df.loc[data['test_mask']].reset_index(drop=True)
os.makedirs('../xydata/proc/', exist_ok=True)
train_test_df.to_pickle('../xydata/proc/all_df.pkl')

pd.to_pickle(train_df, '../xydata/proc/train_df.pkl')
pd.to_pickle(test_df, '../xydata/proc/test_df.pkl')

edge_df = pd.DataFrame(data['edge_index'], columns=['from_id', 'to_id'])
edge_df['edge_type'] = data['edge_type']
edge_df['edge_timestamp'] = data['edge_timestamp']
print(edge_df)
pd.to_pickle(edge_df, '../xydata/proc/edge_df.pkl')

