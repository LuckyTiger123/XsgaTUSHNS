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

cuda_device = 7
epoch_number = 30
heads = 4
att_norm = True
key_type = 0
hidden_size = 256
change_to_directed = True
layer_num = 3
train_sampler = -1
dropout = 0.5
class_weight = 1
learning_rate = 0.0005
weight_decay = 1e-4
file_id = 7

model_path_list = [
    # '0.0005_0.5_0_0.pth',
    '0.0005_0.5_0_1.pth',
    # '0.0005_0.5_0_2.pth',
    '0.0005_0.5_0_3.pth',
    '0.0005_0.5_0_4.pth',
    # '0.0005_0.5_1_0.pth',
    # '0.0005_0.5_1_1.pth',
    '0.0005_0.5_1_2.pth',
    # '0.0005_0.5_1_3.pth',
    # '0.0005_0.5_1_4.pth',
    '0.0005_0.5_2_0.pth',
    '0.0005_0.5_2_1.pth',
    '0.0005_0.5_2_2.pth',
    # '0.0005_0.5_2_3.pth',
    # '0.0005_0.5_2_4.pth',
    # '0.001_0_0_0.pth',
    '0.001_0_0_1.pth',
    # '0.001_0_0_2.pth',
    '0.001_0_0_3.pth',
    '0.001_0_0_4.pth',
    '0.001_0_1_0.pth',
    # '0.001_0_1_1.pth',
    # '0.001_0_1_2.pth',
    # '0.001_0_1_3.pth',
    # '0.001_0_1_4.pth',
    # '0.001_0_2_0.pth',
    '0.001_0_2_1.pth',
    '0.001_0_2_2.pth',
    # '0.001_0_2_3.pth',
    # '0.001_0_2_4.pth',
]

# device
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

# random seed
random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# load data
dataset = XYGraphP1(root='/home/luckytiger/xinye_data_1', name='xydata')
data = dataset[0]

# deal with the node feature
x = data.x[:, :37]
x_back_label = data.x[:, 39:41]
x = torch.cat((x, x_back_label), dim=1)
x_dtf = fold_timestamp(data.x[:, 41:], fold_num=30)
x_tg = degree_frequency(data.x[:, 41:])

x = torch.cat((x, x_dtf, x_tg), dim=1)
# x = torch.cat((x, x_dtf), dim=1)
data.x = x
if change_to_directed:
    edge_index, edge_attr = to_undirected(data.edge_index, data.edge_attr)
else:
    edge_index, edge_attr = data.edge_index, data.edge_attr
data.edge_index = edge_index
data.edge_attr = edge_attr


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer_num = layer_num

        self.convs = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(
            modified_GAT(data.x.size(1), hidden_size, heads=heads, att_norm=att_norm, key_type=key_type,
                         dropout=dropout))
        self.skips.append(torch.nn.Linear(data.x.size(1), hidden_size * heads))
        self.bns.append(torch.nn.BatchNorm1d(data.x.size(1)))

        for i in range(layer_num - 2):
            self.convs.append(
                modified_GAT(hidden_size * heads, hidden_size, heads=heads, att_norm=att_norm, key_type=key_type,
                             dropout=dropout))
            self.skips.append(torch.nn.Linear(hidden_size * heads, hidden_size * heads))
            self.bns.append(torch.nn.BatchNorm1d(hidden_size * heads))

        self.convs.append(
            modified_GAT(hidden_size * heads, 2, heads=1, att_norm=att_norm, key_type=key_type, dropout=dropout))
        self.skips.append(torch.nn.Linear(hidden_size * heads, 2))
        self.bns.append(torch.nn.BatchNorm1d(hidden_size * heads))

        self.reset_parameters()

    def forward(self, x, edge_index):
        for i in range(self.layer_num):
            x = self.bns[i](x)
            x = F.relu(self.skips[i](x) + self.convs[i](x, edge_index))

        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for lin in self.skips:
            lin.reset_parameters()


train_loader = NeighborLoader(data, num_neighbors=[train_sampler] * layer_num, input_nodes=data.train_mask,
                              batch_size=1024, shuffle=True, num_workers=12)
valid_loader = NeighborLoader(data, num_neighbors=[-1] * layer_num, input_nodes=data.valid_mask, batch_size=4096,
                              shuffle=False, num_workers=12)
test_loader = NeighborLoader(data, num_neighbors=[-1] * layer_num, input_nodes=data.test_mask, batch_size=4096,
                             shuffle=False, num_workers=12)


@torch.no_grad()
def valid():
    # data.y is labels of shape (N, )
    model.eval()
    ys, preds = [], []
    for batch in tqdm(valid_loader):
        batch_size = batch.batch_size
        ys.append(batch.y[:batch_size])
        out = model(batch.x.to(device), batch.edge_index.to(device))[:batch_size]
        preds.append(F.softmax(out, dim=1)[:, 1].cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return roc_auc_score(y, pred), y, pred


@torch.no_grad()
def test():
    # data.y is labels of shape (N, )
    model.eval()
    preds = []
    for batch in tqdm(test_loader):
        batch_size = batch.batch_size
        out = model(batch.x.to(device), batch.edge_index.to(device))[:batch_size]
        preds.append(F.softmax(out, dim=1).cpu())

    pred = torch.cat(preds, dim=0).numpy()
    return pred


valid_pred_list = []
test_pred_list = []
y = None
for model_path in model_path_list:
    model = torch.load('/home/luckytiger/2022_finvcup_baseline/trained_model/' + model_path).to(device)
    valid_auc, y, c_pred = valid()
    test_pred = test()
    valid_pred_list.append(c_pred)
    test_pred_list.append(test_pred)
    print('The auc for the {} model is {}.'.format(model_path, valid_auc))

print('----------------------------------------------')
valid_result_agg = np.array(valid_pred_list)
valid_result_mean = np.mean(valid_result_agg, axis=0)

test_result_agg = np.array(test_pred_list)
test_result_mean = np.mean(test_result_agg, axis=0)

np.save('../submit/model_submit_{}.npy'.format(file_id), test_result_mean)

print('The auc score for the aggregation model is {}.'.format(roc_auc_score(y, valid_result_mean)))
