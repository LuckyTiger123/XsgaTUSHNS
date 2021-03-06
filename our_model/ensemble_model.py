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
# parser.add_argument('-kt', '--key_type', type=int, default=0)
parser.add_argument('-r', '--rand_seed', type=int, default=0)
args = parser.parse_args()

hidden_size = args.hidden_size
cuda_device = args.cuda
epoch_number = args.epoch
dropout = args.dropout
model_number = args.model_num
learning_rate = args.learning_rate
weight_decay = args.weight_decay
# key_type = args.key_type
key_type_list = [0, 1, 2]

heads = 4
att_norm = True
change_to_directed = True
layer_num = 3
train_sampler = -1
class_weight = 1

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


def train():
    # data.y is labels of shape (N, )
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch.batch_size
        out = model(batch.x, batch.edge_index)
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
        out = model(batch.x.to(device), batch.edge_index.to(device))[:batch_size]
        preds.append(F.softmax(out, dim=1)[:, 1].cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return roc_auc_score(y, pred)


train_loader = NeighborLoader(data, num_neighbors=[train_sampler] * layer_num, input_nodes=data.train_mask,
                              batch_size=1024, shuffle=True, num_workers=12)
valid_loader = NeighborLoader(data, num_neighbors=[-1] * layer_num, input_nodes=data.valid_mask, batch_size=4096,
                              shuffle=False, num_workers=12)
test_loader = NeighborLoader(data, num_neighbors=[-1] * layer_num, input_nodes=data.test_mask, batch_size=4096,
                             shuffle=False, num_workers=12)

for m_num in range(model_number):
    for key_type in key_type_list:
        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
                torch.save(model, '../trained_model/{}_{}_{}_{}.pth'.format(learning_rate, dropout, key_type, m_num))
            print('-----------------------------------------------------')

        print('-----------------------------------------------------')
        print('The best valid auc is {}.'.format(best_valid_auc))
        print('-----------------------------------------------------')
