import gc
import os
import sys
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from our_model.modified_xygraph import XYGraphP1
from our_model.load_data import fold_timestamp, to_undirected, degree_frequency, cal_current_state, sharpen_value
from our_model.faeture_propagation import feature_propagation
from our_model.modified_GAT import modified_GAT

y = np.load('/home/luckytiger/xinye_data_1/y.npy')
pred = np.load('/home/luckytiger/xinye_data_1/pred.npy')

cal_current_state(y, pred)
print('Origin auc is {}.'.format(roc_auc_score(y, pred)))

auc_result = list()

for i in range(1000):
    print('----------------------------------------')
    pred_fix = sharpen_value(pred, right_th=1 - 1e-3 * i)
    cal_current_state(y, pred_fix)
    auc_current = roc_auc_score(y, pred_fix)
    print('For the {} round, the auc score is {}.'.format(i, auc_current))
    auc_result.append(auc_current)
    print('----------------------------------------')

print(max(auc_result))
