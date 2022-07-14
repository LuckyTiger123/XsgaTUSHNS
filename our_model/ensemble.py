import gc
import os
import sys
import torch
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from our_model.modified_xygraph import XYGraphP1
from our_model.load_data import fold_timestamp, to_undirected, degree_frequency
from our_model.faeture_propagation import feature_propagation
from our_model.modified_GAT import modified_GAT

file_id = 'final'

# load data
dataset = XYGraphP1(root='/home/luckytiger/xinye_data_1', name='xydata')
data = dataset[0]

series3_valid = np.load('../other/pred/84.8_t_FLAG_valid.npy')
series3_test = np.load('../other/pred/84.8_t_FLAG_test.npy')

series1_valid = np.load('/home/luckytiger/2022_finvcup_baseline/submit/series1_valid_7.npy')
series1_test = np.load('/home/luckytiger/2022_finvcup_baseline/submit/series1_test_7.npy')

series2_valid = np.load('/home/luckytiger/2022_finvcup_baseline/submit/series2_valid_0.npy')
series2_test = np.load('/home/luckytiger/2022_finvcup_baseline/submit/series2_test_0.npy')

series4_valid_0 = np.load('../submit/series4_valid_0.npy')
series4_test_0 = np.load('../submit/series4_test_0.npy')
series4_test_0 = series4_test_0.reshape(-1, 1)
label_0 = 1 - series4_test_0
series4_test_0 = np.hstack((label_0, series4_test_0))

series4_valid_1 = np.load('../submit/series4_valid_1.npy')
series4_test_1 = np.load('../submit/series4_test_1.npy')
series4_test_1 = series4_test_1.reshape(-1, 1)
label_0 = 1 - series4_test_1
series4_test_1 = np.hstack((label_0, series4_test_1))

series4_valid_2 = np.load('../submit/series4_valid_2.npy')
series4_test_2 = np.load('../submit/series4_test_2.npy')
series4_test_2 = series4_test_2.reshape(-1, 1)
label_0 = 1 - series4_test_2
series4_test_2 = np.hstack((label_0, series4_test_2))

series4_valid_3 = np.load('../submit/series4_valid_3.npy')
series4_test_3 = np.load('../submit/series4_test_3.npy')
series4_test_3 = series4_test_3.reshape(-1, 1)
label_0 = 1 - series4_test_3
series4_test_3 = np.hstack((label_0, series4_test_3))

series4_valid_4 = np.load('../submit/series4_valid_4.npy')
series4_test_4 = np.load('../submit/series4_test_4.npy')
series4_test_4 = series4_test_4.reshape(-1, 1)
label_0 = 1 - series4_test_4
series4_test_4 = np.hstack((label_0, series4_test_4))

series4_valid_5 = np.load('../submit/series4_valid_5.npy')
series4_test_5 = np.load('../submit/series4_test_5.npy')
series4_test_5 = series4_test_5.reshape(-1, 1)
label_0 = 1 - series4_test_5
series4_test_5 = np.hstack((label_0, series4_test_5))

series4_valid_6 = np.load('../submit/series4_valid_6.npy')
series4_test_6 = np.load('../submit/series4_test_6.npy')
series4_test_6 = series4_test_6.reshape(-1, 1)
label_0 = 1 - series4_test_6
series4_test_6 = np.hstack((label_0, series4_test_6))

series4_valid_7 = np.load('../submit/series4_valid_7.npy')
series4_test_7 = np.load('../submit/series4_test_7.npy')
series4_test_7 = series4_test_7.reshape(-1, 1)
label_0 = 1 - series4_test_7
series4_test_7 = np.hstack((label_0, series4_test_7))

series4_valid_8 = np.load('../submit/series4_valid_8.npy')
series4_test_8 = np.load('../submit/series4_test_8.npy')
series4_test_8 = series4_test_8.reshape(-1, 1)
label_0 = 1 - series4_test_8
series4_test_8 = np.hstack((label_0, series4_test_8))

series4_valid_9 = np.load('../submit/series4_valid_9.npy')
series4_test_9 = np.load('../submit/series4_test_9.npy')
series4_test_9 = series4_test_9.reshape(-1, 1)
label_0 = 1 - series4_test_9
series4_test_9 = np.hstack((label_0, series4_test_9))

series4_valid_10 = np.load('../submit/series4_valid_10.npy')
series4_test_10 = np.load('../submit/series4_test_10.npy')
series4_test_10 = series4_test_10.reshape(-1, 1)
label_0 = 1 - series4_test_10
series4_test_10 = np.hstack((label_0, series4_test_10))

series4_valid_11 = np.load('../submit/series4_valid_11.npy')
series4_test_11 = np.load('../submit/series4_test_11.npy')
series4_test_11 = series4_test_11.reshape(-1, 1)
label_0 = 1 - series4_test_11
series4_test_11 = np.hstack((label_0, series4_test_11))

series4_valid_12 = np.load('../submit/series4_valid_12.npy')
series4_test_12 = np.load('../submit/series4_test_12.npy')
series4_test_12 = series4_test_12.reshape(-1, 1)
label_0 = 1 - series4_test_12
series4_test_12 = np.hstack((label_0, series4_test_12))

series4_valid_13 = np.load('../submit/series4_valid_13.npy')
series4_test_13 = np.load('../submit/series4_test_13.npy')
series4_test_13 = series4_test_13.reshape(-1, 1)
label_0 = 1 - series4_test_13
series4_test_13 = np.hstack((label_0, series4_test_13))

series4_valid_14 = np.load('../submit/series4_valid_14.npy')
series4_test_14 = np.load('../submit/series4_test_14.npy')
series4_test_14 = series4_test_14.reshape(-1, 1)
label_0 = 1 - series4_test_14
series4_test_14 = np.hstack((label_0, series4_test_14))

series6_valid_0 = np.load('../submit/series6_valid_0.npy')
series6_test_0 = np.load('../submit/series6_test_0.npy')
series6_test_0 = series6_test_0.reshape(-1, 1)
label_0 = 1 - series6_test_0
series6_test_0 = np.hstack((label_0, series6_test_0))

series6_valid_1 = np.load('../submit/series6_valid_1.npy')
series6_test_1 = np.load('../submit/series6_test_1.npy')
series6_test_1 = series6_test_1.reshape(-1, 1)
label_0 = 1 - series6_test_1
series6_test_1 = np.hstack((label_0, series6_test_1))

series6_valid_2 = np.load('../submit/series6_valid_2.npy')
series6_test_2 = np.load('../submit/series6_test_2.npy')
series6_test_2 = series6_test_2.reshape(-1, 1)
label_0 = 1 - series6_test_2
series6_test_2 = np.hstack((label_0, series6_test_2))

series6_valid_3 = np.load('../submit/series6_valid_3.npy')
series6_test_3 = np.load('../submit/series6_test_3.npy')
series6_test_3 = series6_test_3.reshape(-1, 1)
label_0 = 1 - series6_test_3
series6_test_3 = np.hstack((label_0, series6_test_3))

series6_valid_4 = np.load('../submit/series6_valid_4.npy')
series6_test_4 = np.load('../submit/series6_test_4.npy')
series6_test_4 = series6_test_4.reshape(-1, 1)
label_0 = 1 - series6_test_4
series6_test_4 = np.hstack((label_0, series6_test_4))

series6_valid_5 = np.load('../submit/series6_valid_5.npy')
series6_test_5 = np.load('../submit/series6_test_5.npy')
series6_test_5 = series6_test_5.reshape(-1, 1)
label_0 = 1 - series6_test_5
series6_test_5 = np.hstack((label_0, series6_test_5))

series6_valid_6 = np.load('../submit/series6_valid_6.npy')
series6_test_6 = np.load('../submit/series6_test_6.npy')
series6_test_6 = series6_test_6.reshape(-1, 1)
label_0 = 1 - series6_test_6
series6_test_6 = np.hstack((label_0, series6_test_6))

series6_valid_7 = np.load('../submit/series6_valid_7.npy')
series6_test_7 = np.load('../submit/series6_test_7.npy')
series6_test_7 = series6_test_7.reshape(-1, 1)
label_0 = 1 - series6_test_7
series6_test_7 = np.hstack((label_0, series6_test_7))

series6_valid_8 = np.load('../submit/series6_valid_8.npy')
series6_test_8 = np.load('../submit/series6_test_8.npy')
series6_test_8 = series6_test_8.reshape(-1, 1)
label_0 = 1 - series6_test_8
series6_test_8 = np.hstack((label_0, series6_test_8))

series6_valid_9 = np.load('../submit/series6_valid_9.npy')
series6_test_9 = np.load('../submit/series6_test_9.npy')
series6_test_9 = series6_test_9.reshape(-1, 1)
label_0 = 1 - series6_test_9
series6_test_9 = np.hstack((label_0, series6_test_9))

series6_valid_10 = np.load('../submit/series6_valid_10.npy')
series6_test_10 = np.load('../submit/series6_test_10.npy')
series6_test_10 = series6_test_10.reshape(-1, 1)
label_0 = 1 - series6_test_10
series6_test_10 = np.hstack((label_0, series6_test_10))

series6_valid_11 = np.load('../submit/series6_valid_11.npy')
series6_test_11 = np.load('../submit/series6_test_11.npy')
series6_test_11 = series6_test_11.reshape(-1, 1)
label_0 = 1 - series6_test_11
series6_test_11 = np.hstack((label_0, series6_test_11))

series6_valid_12 = np.load('../submit/series6_valid_12.npy')
series6_test_12 = np.load('../submit/series6_test_12.npy')
series6_test_12 = series6_test_12.reshape(-1, 1)
label_0 = 1 - series6_test_12
series6_test_12 = np.hstack((label_0, series6_test_12))

series6_valid_13 = np.load('../submit/series6_valid_13.npy')
series6_test_13 = np.load('../submit/series6_test_13.npy')
series6_test_13 = series6_test_13.reshape(-1, 1)
label_0 = 1 - series6_test_13
series6_test_13 = np.hstack((label_0, series6_test_13))

series6_valid_14 = np.load('../submit/series6_valid_14.npy')
series6_test_14 = np.load('../submit/series6_test_14.npy')
series6_test_14 = series6_test_14.reshape(-1, 1)
label_0 = 1 - series6_test_14
series6_test_14 = np.hstack((label_0, series6_test_14))

series4_valid = np.array(
    [series4_valid_0, series4_valid_1, series4_valid_2, series4_valid_3, series4_valid_4, series4_valid_5,
     series4_valid_6, series4_valid_7, series4_valid_8, series4_valid_9, series4_valid_10, series4_valid_11,
     series4_valid_12, series4_valid_13, series4_valid_14])
series4_test = np.array(
    [series4_test_0, series4_test_1, series4_test_2, series4_test_3, series4_test_4, series4_test_5, series4_test_6,
     series4_test_7, series4_test_8, series4_test_9, series4_test_10, series4_test_11, series4_test_12, series4_test_13,
     series4_test_14])

series6_valid = np.array(
    [series6_valid_0, series6_valid_1, series6_valid_2, series6_valid_3, series6_valid_4, series6_valid_5,
     series6_valid_6, series6_valid_7, series6_valid_8, series6_valid_9, series6_valid_10, series6_valid_11,
     series6_valid_12, series6_valid_13, series6_valid_14])
series6_test = np.array(
    [series6_test_0, series6_test_1, series6_test_2, series6_test_3, series6_test_4, series6_test_5, series6_test_6,
     series6_test_7, series6_test_8, series6_test_9, series6_test_10, series6_test_11, series6_test_12, series6_test_13,
     series6_test_14])

ensemble_valid_agg = np.mean(
    np.vstack((series1_valid, series3_valid, series3_valid, series6_valid, series6_valid)), axis=0)

print('The auc score is {}.'.format(roc_auc_score(data.y[data.valid_mask].numpy(), ensemble_valid_agg)))

ensemble_test_agg = np.mean(np.vstack((series1_test, series3_test, series3_test, series6_test, series6_test)), axis=0)
np.save('../submit/ensemble_{}.npy'.format(file_id), ensemble_test_agg)
