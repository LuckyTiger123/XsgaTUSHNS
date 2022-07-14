import gc
import os
import sys
import numpy as np
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from our_models.modified_xygraph import XYGraphP1
from gtrick import FLAG
from our_models.load_data import fold_timestamp, to_undirected, degree_frequency
from our_models.faeture_propagation import feature_propagation
from our_models.modified_GAT_yh import modified_GAT, TimeEncoder
# from our_models.utils import process_time, one_hot
from our_models.focalloss import FocalLoss

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, default=7)
# parser.add_argument('-t', '--train_round', type=int, default=3)
# parser.add_argument('-f', '--file_id', type=int, default=0)
parser.add_argument('-e', '--epoch', type=int, default=30)
parser.add_argument('-mn', '--model_num', type=int, default=5)
parser.add_argument('-hd', '--hidden_size', type=int, default=256)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4)
parser.add_argument('-d', '--dropout', type=float, default=0)
parser.add_argument('-k', '--key_type', type=int, default=0)
parser.add_argument('-r', '--rand_seed', type=int, default=0)
args = parser.parse_args()

# hidden_size = args.hidden_size
cuda_device = args.cuda
# epoch_number = args.epoch
# dropout = args.dropout
model_number = args.model_num
# learning_rate = args.learning_rate
# weight_decay = args.weight_decay

key_type = args.key_type
heads = 4
att_norm = True
change_to_directed = True
layer_num = 3
train_sampler = -1
class_weight = 1
epoch_number = 30
hidden_size = 256
dropout = 0.5
learning_rate = 0.0005
weight_decay = 1e-4

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
x = torch.cat((x, x_dtf), dim=1)

use_time = False
# add edge_feature
edge_feat = torch.load(os.path.join('/data/shangyihao/ppd', 'edge_feat_all.pt'))[:, 1:]
print('Processing time ......')
# edge_time = process_time(data.edge_index, data.edge_attr[:, -1], 128).unsqueeze(1)
edge_time = torch.load(os.path.join('/data/shangyihao/ppd', 'node_edge_time_cut.pt')).unsqueeze(1)
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

    def forward(self, x, edge_index, edge_attr, perturb=None):

        if perturb is not None:
            x = x + perturb

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


def train():
    # data.y is labels of shape (N, )
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch.batch_size

        forward = lambda perturb: model(batch.x, batch.edge_index, batch.edge_attr, perturb)[:batch_size]
        loss, out = flag(model, forward, batch.x.shape[0], batch.y[:batch_size])

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


@torch.no_grad()
def predict():
    # data.y is labels of shape (N, )
    model.eval()
    ys, preds = [], []
    for batch in tqdm(test_loader):
        batch_size = batch.batch_size
        out = model(batch.x.to(device), batch.edge_index.to(device),
                    batch.edge_attr.to(device))[:batch_size]
        preds.append(F.softmax(out, dim=1)[:, 1].cpu())

    pred = torch.cat(preds, dim=0).numpy()
    return pred


train_loader = NeighborLoader(data, num_neighbors=[train_sampler] * layer_num, input_nodes=data.train_mask,
                              batch_size=1024, shuffle=True, num_workers=4)
valid_loader = NeighborLoader(data, num_neighbors=[-1] * layer_num, input_nodes=data.valid_mask, batch_size=4096,
                              shuffle=False, num_workers=4)
test_loader = NeighborLoader(data, num_neighbors=[-1] * layer_num, input_nodes=data.test_mask, batch_size=4096,
                             shuffle=False, num_workers=4)

for m_num in range(model_number):

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    flag = FLAG(data.x.size(-1), torch.nn.CrossEntropyLoss(), optimizer)

    best_valid_auc = 0
    for epoch in range(epoch_number):
        print('-----------------------------------------------------')
        print('For the {} model, {} epoch:'.format(m_num, epoch))
        train_loss = train()
        print('The train loss is {}.'.format(train_loss))
        c_valid_auc = valid()
        print('The valid auc is {}.'.format(c_valid_auc))

        if c_valid_auc > best_valid_auc:
            best_valid_auc = c_valid_auc
            torch.save(model, '../trained_model/84.8_t_FLAG_{}_{}.pth'.format(key_type, m_num))
        print('-----------------------------------------------------')

    print('-----------------------------------------------------')
    print('The best valid auc is {}.'.format(best_valid_auc))
    print('-----------------------------------------------------')
    print('Predicting ........')
    model = torch.load('../trained_model/84.8_t_FLAG_{}_{}.pth'.format(key_type, m_num)).to(device)
    pred = predict()
    np.save('../trained_model/84.8_t_FLAG_{}_{}_preds.npy'.format(key_type, m_num), pred)
