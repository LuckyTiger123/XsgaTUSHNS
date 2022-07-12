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
# from utils.utils import prepare_folder
from torch_geometric.loader import NeighborLoader

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from our_models.modified_xygraph import XYGraphP1
from our_models.load_data import fold_timestamp, to_undirected, degree_frequency
from our_models.faeture_propagation import feature_propagation
from our_models.modified_GAT import modified_GAT

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, default=9)
parser.add_argument('-t', '--train_round', type=int, default=3)
parser.add_argument('-f', '--file_id', type=int, default=0)
parser.add_argument('-e', '--epoch', type=int, default=30)
parser.add_argument('-hd', '--hidden_size', type=int, default=256)
parser.add_argument('-r', '--rand_seed', type=int, default=0)
args = parser.parse_args()

cuda_device = 9
epoch_number = 30
heads = 4
att_norm = True
key_type = 0
hidden_size = 64
change_to_directed = True
layer_num = 3
train_sampler = -1
dropout = 0.5
class_weight = 1
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

x = torch.cat((x, x_dtf, x_tg), dim=1)
# x = torch.cat((x, x_dtf), dim=1)
# edge_index, edge_attr = to_undirected(data.edge_index, data.edge_attr)
edge_classes = torch.from_numpy(np.load(os.path.join('/data/shangyihao/ppd', 'edge_classes_directed.npy')))
edge_feat = torch.load(os.path.join('/data/shangyihao/ppd', 'edge_feat_all.pt'))
new_time = torch.from_numpy(np.load(os.path.join('/data/shangyihao/ppd', 'new_time.npy')))
data.x = x
edge_all = torch.cat([new_time.unsqueeze(1), edge_feat[:, 1:], edge_classes.unsqueeze(1)], dim=-1)
data.edge_attr = edge_all
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

        self.convs.append(torch.nn.ModuleList([
            modified_GAT(data.x.size(1), hidden_size, heads=heads, att_norm=att_norm, key_type=key_type,
                         dropout=dropout)
            for _ in range(4)
        ]))
        # self.convs.append(
        #     modified_GAT(data.x.size(1), hidden_size, heads=heads, att_norm=att_norm, key_type=key_type,
        #                  dropout=dropout))
        self.skips.append(torch.nn.Linear(data.x.size(1), hidden_size * heads))
        self.bns.append(torch.nn.BatchNorm1d(data.x.size(1)))

        for i in range(layer_num - 2):
            self.convs.append(torch.nn.ModuleList([
                modified_GAT(hidden_size * heads, hidden_size, heads=heads, att_norm=att_norm, key_type=key_type,
                             dropout=dropout)
                for _ in range(4)
            ]))
            # self.convs.append(
            #     modified_GAT(hidden_size * heads, hidden_size, heads=heads, att_norm=att_norm, key_type=key_type,
            #                  dropout=dropout))
            self.skips.append(torch.nn.Linear(hidden_size * heads, hidden_size * heads))
            self.bns.append(torch.nn.BatchNorm1d(hidden_size * heads))

        self.convs.append(torch.nn.ModuleList([
            modified_GAT(hidden_size * heads, 2, heads=1, att_norm=att_norm, key_type=key_type, dropout=dropout)
            for _ in range(4)
        ]))
        # self.convs.append(
        #     modified_GAT(hidden_size * heads, 2, heads=1, att_norm=att_norm, key_type=key_type, dropout=dropout))
        self.skips.append(torch.nn.Linear(hidden_size * heads, 2))
        self.bns.append(torch.nn.BatchNorm1d(hidden_size * heads))

        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        edge_class = edge_attr[:, -1]
        for i in range(self.layer_num):
            x = self.bns[i](x)
            x0 = self.skips[i](x)
            for j in range(4):
                idx = edge_class == j
                sub_edge = edge_index[:, idx]
                x0 += self.convs[i][j](x, sub_edge)
            x = F.relu(x0)
                # x = F.relu( x + self.convs[i](x, edge_index))

        return x

    def reset_parameters(self):
        for convs in self.convs:
            for conv in convs:
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
        out = model(batch.x, batch.edge_index, batch.edge_attr)
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
        out = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device))[:batch_size]
        preds.append(F.softmax(out, dim=1)[:, 1].cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return roc_auc_score(y, pred)


# model_dir = prepare_folder(args.dataset, '84.5_r')
# print('model_dir:', model_dir)

train_loader = NeighborLoader(data, num_neighbors=[train_sampler] * layer_num, input_nodes=data.train_mask,
                              batch_size=1024, shuffle=True, num_workers=4)
valid_loader = NeighborLoader(data, num_neighbors=[train_sampler] * layer_num, input_nodes=data.valid_mask,
                              batch_size=4096, shuffle=False, num_workers=4)

# train_loader = NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[10, 5], batch_size=1024, shuffle=True,
#                                num_workers=12)
# layer_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False,
#                                num_workers=12)


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
best_valid_auc = 0
for epoch in range(epoch_number):
    print('-----------------------------------------------------')
    print('For the {} epoch:'.format(epoch))
    train_loss = train()
    print('The train loss is {}.'.format(train_loss))
    c_valid_auc = valid()
    print('The valid auc is {}.'.format(c_valid_auc))

    if c_valid_auc > best_valid_auc:
        best_valid_auc = c_valid_auc
        # torch.save(model.state_dict(), model_dir + 'model.pt')
    print('-----------------------------------------------------')

print('-----------------------------------------------------')
print('The best valid auc is {}.'.format(best_valid_auc))
print('-----------------------------------------------------')


weight_decay_list = [5e-4, 1e-4]
learning_rate_list = [5e-3, 1e-3, 5e-4]
epoch_num = 30
round_statistic = pd.DataFrame(
    columns=['dataset', 'round', 'learning_rate', 'weight_decay', 'layer_num', 'hidden_size', 'heads', 'att_norm',
             'key_type', 'best_val_auc', 'epoch_num'])

for learning_rate in learning_rate_list:
    for weight_decay in weight_decay_list:
        gc.collect()
        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for round_num in range(args.train_round):
            model.reset_parameters()
            best_valid_auc = best_epoch = 0
            for epoch in range(epoch_num):
                loss = train()
                c_valid_auc = valid()
                if c_valid_auc > best_valid_auc:
                    best_valid_auc = c_valid_auc
                    best_epoch = epoch
                print('For the {} epoch, the loss is {}, the valid auc is {}.'.format(epoch, loss,
                                                                                      c_valid_auc))
                round_statistic.loc[round_statistic.shape[0]] = {
                    'dataset': 'xinye',
                    'round': round_num,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'best_val_auc': round(best_valid_auc, 4),
                    'epoch_num': best_epoch}

save_path = os.path.join('..', 'result')
if not os.path.exists(save_path):
    os.makedirs(save_path)
round_statistic.to_excel(os.path.join(save_path, '84.5_r.xlsx'))
print('Mission completes.')









# print('')
# print('-----------------------------------------------------')
# print('Predicting.')
#
#
# @torch.no_grad()
# def predict():
#     model.eval()
#     preds = []
#     for batch in tqdm(layer_loader):
#         batch_size = batch.batch_size
#         out = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device))[:batch_size]
#         preds.append(F.softmax(out, dim=1)[:, 1].cpu())
#     pred = torch.cat(preds, dim=0).numpy()
#     return pred
#
#
# model.load_state_dict(torch.load(model_dir + 'model.pt'))
# layer_loader = NeighborLoader(data, num_neighbors=[-1], input_nodes=None,
#                               batch_size=4096, shuffle=False, num_workers=4)
# out = predict()
# preds_train, preds_valid = out[data.train_mask], out[data.valid_mask]
# y_train, y_valid = data.y[data.train_mask].cpu(), data.y[data.valid_mask].cpu()
# evaluator = Evaluator('auc')
# train_auc = evaluator.eval(y_train, preds_train)['auc']
# valid_auc = evaluator.eval(y_valid, preds_valid)['auc']
# print('train_auc:', train_auc)
# print('valid_auc:', valid_auc)
#
# preds = out[data.test_mask].cpu().numpy()
# result_path = os.path.join(model_dir, 'preds.npy')
# np.save(result_path, preds)

