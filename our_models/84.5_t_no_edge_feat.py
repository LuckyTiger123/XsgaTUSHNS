import gc
import os
import sys
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader

from our_models.modified_xygraph import XYGraphP1
from our_models.load_data import fold_timestamp, to_undirected, degree_frequency
from our_models.faeture_propagation import feature_propagation
from our_models.modified_GAT_yh import modified_GAT, TimeEncoder
from our_models.focalloss import FocalLoss

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, default=0)
parser.add_argument('-t', '--train_round', type=int, default=3)
parser.add_argument('-f', '--file_id', type=int, default=0)
parser.add_argument('-e', '--epoch', type=int, default=30)
parser.add_argument('-hd', '--hidden_size', type=int, default=256)
parser.add_argument('-r', '--rand_seed', type=int, default=0)
args = parser.parse_args()

cuda_device = 6
epoch_number = 30
heads = 4
att_norm = True
key_type = 0
hidden_size = 256
change_to_directed = True
layer_num = 3
train_sampler = -1
dropout = 0.5
# dropout = 0.
class_weight = 1
learning_rate = 0.0005
# learning_rate = 1e-3
weight_decay = 1e-4
# weight_decay = 5e-4

# device
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

# random seed
random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# load data
dataset = XYGraphP1(root='/data/shangyihao/ppd', name='xydata')
data = dataset[0]

# deal with the node feature
x = data.x[:, :37]

x_back_label = data.x[:, 39:41]
x = torch.cat((x, x_back_label), dim=1)
path_x_dtf = os.path.join('/data/shangyihao/ppd/xydata/processed', 'x_dtf.pt')
if os.path.exists(path_x_dtf):
    x_dtf = torch.load(path_x_dtf)
else:
    x_dtf = fold_timestamp(data.x[:, 41:], fold_num=30)
    torch.save(x_dtf, path_x_dtf)

path_x_tg = os.path.join('/data/shangyihao/ppd/xydata/processed', 'x_tg.pt')
if os.path.exists(path_x_tg):
    x_tg = torch.load(path_x_tg)
else:
    x_tg = degree_frequency(data.x[:, 41:])
    torch.save(x_tg, path_x_tg)

# x = torch.cat((x, x_dtf, x_tg, x_nan), dim=1)
x = torch.cat((x, x_dtf), dim=1)

# add node_timestamps
use_time = False
# flag = 'max'
# if flag == 'mean':
#     x_time = torch.load(os.path.join('/data/shangyihao/ppd', 'node_time_cut_mean.pt'))
# else:
#     x_time = torch.load(os.path.join('/data/shangyihao/ppd', 'node_time_cut_max.pt'))
#
# x = torch.cat((x, x_time.unsqueeze(1)), dim=1)


# add edge_feature
# edge_feat = torch.load(os.path.join('/data/shangyihao/ppd', 'edge_feat_all.pt'))[:, 1:]
edge_time = torch.load(os.path.join('/data/shangyihao/ppd', 'node_edge_time_cut.pt')).unsqueeze(1)
# edge_all = torch.cat((edge_time, edge_feat), dim=1)
data.edge_attr = edge_time

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
        self.edge_mlp = torch.nn.Linear(hidden_size, hidden_size)
        self.temporal_encoder = TimeEncoder(hidden_size)

        if use_time:
            input_dim = data.x.size(1) - 1
        else:
            input_dim = data.x.size(1)

        self.convs.append(
            modified_GAT(input_dim, hidden_size, heads=heads, att_norm=att_norm, key_type=key_type
                         , edge_dim=hidden_size
                         , dropout=dropout))
        self.skips.append(torch.nn.Linear(input_dim, hidden_size * heads))
        self.bns.append(torch.nn.BatchNorm1d(input_dim))

        for i in range(layer_num - 2):
            self.convs.append(
                modified_GAT(hidden_size * heads, hidden_size, heads=heads, att_norm=att_norm, key_type=key_type
                             , edge_dim=hidden_size
                             , dropout=dropout))
            self.skips.append(torch.nn.Linear(hidden_size * heads, hidden_size * heads))
            self.bns.append(torch.nn.BatchNorm1d(hidden_size * heads))

        self.convs.append(
            modified_GAT(hidden_size * heads, 2, heads=1, att_norm=att_norm, key_type=key_type
                         , edge_dim=hidden_size
                         , dropout=dropout))
        self.skips.append(torch.nn.Linear(hidden_size * heads, 2))
        self.bns.append(torch.nn.BatchNorm1d(hidden_size * heads))

        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        t = edge_attr.squeeze(1)
        # edge_feat = edge_attr[:, 1:]
        t_emb = self.temporal_encoder(t)
        edge_msg = self.edge_mlp(t_emb)
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


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# func_loss = FocalLoss(gamma=2.5)


def train():
    # data.y is labels of shape (N, )
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch.batch_size
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        # batch_t = batch.x[:, -1]
        # batch_x = batch.x[:, :-1]
        # out = model(batch_x, batch.edge_index, batch_t)
        # loss = func_loss(out[:batch_size], batch.y[:batch_size])
        loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size],
                               weight=torch.Tensor([1, class_weight]).to(device))
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


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
    return roc_auc_score(y, pred)


train_loader = NeighborLoader(data, num_neighbors=[train_sampler] * layer_num, input_nodes=data.train_mask,
                              batch_size=1024, shuffle=True, num_workers=12)
valid_loader = NeighborLoader(data, num_neighbors=[train_sampler] * layer_num, input_nodes=data.valid_mask,
                              batch_size=4096, shuffle=False, num_workers=12)

# train_loader = NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[10, 5], batch_size=1024, shuffle=True,
#                                num_workers=12)
# layer_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False,
#                                num_workers=12)

best_valid_auc = 0
for epoch in range(epoch_number):
    print('-----------------------------------------------------')
    print('For the {} epoch:'.format(epoch))
    train_loss = train()
    print('The train loss is {}.'.format(train_loss))
    c_valid_auc = valid()
    print('The valid auc is {}.'.format(c_valid_auc))
    # print('Time flag is {}. The valid auc is {}.'.format(flag, c_valid_auc))

    if c_valid_auc > best_valid_auc:
        best_valid_auc = c_valid_auc
    print('-----------------------------------------------------')

print('-----------------------------------------------------')
print('The best valid auc is {}.'.format(best_valid_auc))
# print('Time flag is {}. The best valid auc is {}.'.format(flag, best_valid_auc))
print('-----------------------------------------------------')