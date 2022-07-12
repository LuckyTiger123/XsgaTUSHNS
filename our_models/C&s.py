import os
import sys
import torch
from tqdm import tqdm
import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader
import argparse

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from our_models.modified_xygraph_yh import XYGraphP1
from our_models.load_data import fold_timestamp, to_undirected
from our_models.faeture_propagation import feature_propagation
from our_models.modified_GAT_yh import modified_GAT
from utils.utils import prepare_tune_folder

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, default=0)
parser.add_argument('-t', '--train_round', type=int, default=1)
parser.add_argument('-f', '--file_id', type=int, default=0)
parser.add_argument('-e', '--epoch', type=int, default=30)
parser.add_argument('-x', '--x_type', type=int, default=5)
# parser.add_argument('-hs', '--hidden_size', type=int, default=256)
parser.add_argument('-r', '--rand_seed', type=int, default=0)
args = parser.parse_args()

# parameters
epoch_num = args.epoch
# round_number = args.train_round
device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

# load data
dataset = XYGraphP1(root='/data/shangyihao/ppd', name='xydata')
data = dataset[0]

# edge_index, edge_attr = to_undirected(data.edge_index, data.edge_attr)
edge_index, edge_attr = data.edge_index, data.edge_attr
data.edge_index = edge_index
data.edge_attr = edge_attr

# hyperparameter list
lr = 5e-4
weight_decay = 5e-4
hidden_size = 64
head = 1
key_type = 0

# edge_index = torch.from_numpy(edge_index)
adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                     sparse_sizes=(data.size(0), data.size(0)),
                     is_sorted=True)
adj_t = adj_t.to_symmetric()
adj_t = gcn_norm(adj_t, add_self_loops=True)

train_mask = data.train_mask
valid_mask = data.valid_mask
test_mask = data.test_mask
y_pred = torch.from_numpy(np.load('/home/shangyihao/xinye/our_models/tune_results'
                                  '/XYGraphP1/gat_cspre_c&s/20220708_200446/pred.npy'))

y_train = data.y[train_mask]
y_valid = data.y[valid_mask]

num_correction_layers = 3
correction_alpha = 1.0
num_smoothing_layers = 2
smoothing_alpha = 0.8

model = CorrectAndSmooth(num_correction_layers, correction_alpha,
                         num_smoothing_layers, smoothing_alpha,
                         autoscale=False)

print('Correcting predictions...', end=' ', flush=True)
y_pred = model.correct(y_pred, y_train, train_mask, adj_t)

print('Smoothing predictions...', end=' ', flush=True)
y_pred = model.smooth(y_pred, y_train, train_mask, adj_t)


print('The valid auc is {}.'.format(roc_auc_score(y_valid, y_pred[valid_mask])))
