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
from our_models.load_data import fold_timestamp, to_undirected, degree_frequency
from our_models.faeture_propagation import feature_propagation
from our_models.modified_GAT_yh import modified_GAT, TimeEncoder

model_path_list = [
    'time_edge_0.0005_0.5_0_0.pth',
    'time_edge_0.0005_0.5_0_1.pth',
    'time_edge_0.0005_0.5_0_2.pth',
    'time_edge_0.0005_0.5_0_3.pth',
    'time_edge_0.0005_0.5_0_4.pth',
    'time_edge_0.0005_0.5_1_0.pth',
    'time_edge_0.0005_0.5_1_1.pth',
    'time_edge_0.0005_0.5_1_2.pth',
    'time_edge_0.0005_0.5_1_3.pth',
    'time_edge_0.0005_0.5_1_4.pth',
    'time_edge_0.0005_0.5_2_0.pth',
    'time_edge_0.0005_0.5_2_1.pth',
    'time_edge_0.0005_0.5_2_2.pth',
    'time_edge_0.0005_0.5_2_3.pth',
    'time_edge_0.0005_0.5_2_4.pth',
]

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
file_id = 0

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

# add edge_feature
edge_feat = torch.load(os.path.join('../data', 'edge_feat_all.pt'))[:, 1:]
edge_time = torch.load(os.path.join('../data', 'node_edge_time_cut.pt')).unsqueeze(1)
edge_all = torch.cat((edge_time, edge_feat), dim=1)
data.edge_attr = edge_all

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
        self.edge_mlp = torch.nn.Linear(data.edge_attr.size(1) + hidden_size - 1, hidden_size)
        self.temporal_encoder = TimeEncoder(hidden_size)

        # if use_time:
        #     input_dim = data.x.size(1) - 1
        # else:
        #     input_dim = data.x.size(1)

        self.convs.append(
            modified_GAT(data.x.size(1), hidden_size, heads=heads, att_norm=att_norm, key_type=key_type
                         , edge_dim=hidden_size, dropout=dropout))
        self.skips.append(torch.nn.Linear(data.x.size(1), hidden_size * heads))
        self.bns.append(torch.nn.BatchNorm1d(data.x.size(1)))

        for i in range(layer_num - 2):
            self.convs.append(
                modified_GAT(hidden_size * heads, hidden_size, heads=heads, att_norm=att_norm, key_type=key_type
                             , edge_dim=hidden_size, dropout=dropout))
            self.skips.append(torch.nn.Linear(hidden_size * heads, hidden_size * heads))
            self.bns.append(torch.nn.BatchNorm1d(hidden_size * heads))

        self.convs.append(
            modified_GAT(hidden_size * heads, 2, heads=1, att_norm=att_norm, key_type=key_type
                         , edge_dim=hidden_size, dropout=dropout))
        self.skips.append(torch.nn.Linear(hidden_size * heads, 2))
        self.bns.append(torch.nn.BatchNorm1d(hidden_size * heads))

        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        t = edge_attr[:, 0]
        edge_feat = edge_attr[:, 1:]
        t_emb = self.temporal_encoder(t)
        edge_msg = self.edge_mlp(torch.cat((edge_feat, t_emb), dim=1))
        for i in range(self.layer_num):
            x = self.bns[i](x)
            x = F.relu(self.skips[i](x) + self.convs[i](x, edge_index, edge_msg))

        return x

    def reset_parameters(self):
        self.edge_mlp.reset_parameters()
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
        out = model(batch.x.to(device), batch.edge_index.to(device),
                    batch.edge_attr.to(device))[:batch_size]
        # batch_x = batch.x.to(device)
        # batch_t = batch_x[:, -1]
        # batch_x = batch_x[:, :-1]
        # batch_edge_index = batch.edge_index.to(device)
        # out = model(batch_x, batch_edge_index, batch_t)[:batch_size]
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
        out = model(batch.x.to(device), batch.edge_index.to(device),
                    batch.edge_attr.to(device))[:batch_size]
        # batch_x = batch.x.to(device)
        # batch_t = batch_x[:, -1]
        # batch_x = batch_x[:, :-1]
        # batch_edge_index = batch.edge_index.to(device)
        # out = model(batch_x, batch_edge_index, batch_t)[:batch_size]
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
np.save('../submit/series2_valid_{}.npy'.format(file_id), valid_result_agg)
valid_result_mean = np.mean(valid_result_agg, axis=0)

print('The auc score for the aggregation model is {}.'.format(roc_auc_score(y, valid_result_mean)))

test_result_agg = np.array(test_pred_list)
np.save('../submit/series2_test_{}.npy'.format(file_id), test_result_agg)
test_result_mean = np.mean(test_result_agg, axis=0)

series1_valid = np.load('/home/luckytiger/2022_finvcup_baseline/submit/series1_valid_7.npy')
series1_test = np.load('/home/luckytiger/2022_finvcup_baseline/submit/series1_test_7.npy')

valid_result_agg_2 = np.vstack((valid_result_agg, series1_valid))
valid_result_mean_2 = np.mean(valid_result_agg_2, axis=0)

print('The auc score for the total aggregation model is {}.'.format(roc_auc_score(y, valid_result_mean_2)))

test_result_agg_2 = np.vstack((test_result_agg, series1_test))
test_result_mean_2 = np.mean(test_result_agg_2, axis=0)

np.save('../submit/total_test_{}.npy'.format(file_id), test_result_mean_2)

# valid_mean = np.mean(np.array([valid_result_mean, series1_valid]), axis=0)
