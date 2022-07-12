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

# random seed
random_seed = args.rand_seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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

# feature setup
# add feature flag, id, degree and dynamic degree feature has been tested
if args.x_type == 0:
    # add feature flag, concat id, degree and dynamic degree feature
    x = data.x[:, :34]
    x_dtf = fold_timestamp(data.x[:, 41:], fold_num=30)
    concat_x = torch.cat([data.x[:, 34:37], x_dtf], dim=1)
    data.concat_size = concat_x.size(1)
    data.x = torch.cat([x, concat_x], dim=1)
elif args.x_type == 1:
    # add feature flag and degree, concat id and dynamic degree feature
    x = data.x[:, :34]
    x_d = data.x[:, 35:37]
    x = torch.cat((x, x_d), dim=1)
    x_dtf = fold_timestamp(data.x[:, 41:], fold_num=30)
    concat_x = torch.cat([data.x[:, 34].unsqueeze(1), x_dtf], dim=1)
    data.concat_size = concat_x.size(1)
    data.x = torch.cat([x, concat_x], dim=1)
elif args.x_type == 2:
    # add feature flag, degree and dynamic degree feature, concat id
    x = data.x[:, :34]
    x_d = data.x[:, 35:37]
    x_dtf = fold_timestamp(data.x[:, 41:], fold_num=30)
    x = torch.cat((x, x_d, x_dtf), dim=1)
    concat_x = data.x[:, 34].unsqueeze(1)
    data.concat_size = concat_x.size(1)
    data.x = torch.cat([x, concat_x], dim=1)
# elif args.x_type == 3:
#     # add feature flag, degree, label and dynamic degree feature
#     x = data.x[:, :34]
#     x_d = data.x[:, 35:41]
#     x_dtf = fold_timestamp(data.x[:, 41:], fold_num=30)
#     x = torch.cat((x, x_d, x_dtf), dim=1)
#     data.x = x
else:
    # feature generation
    x = data.x[:, :37]
    x_dtf = fold_timestamp(data.x[:, 41:], fold_num=30)
    x = torch.cat((x, x_dtf), dim=1)
    concat_x = None
    data.concat_size = None
    data.x = x


# neighborhood loader
train_loader = NeighborLoader(data, num_neighbors=[-1], input_nodes=data.train_mask, batch_size=1024, shuffle=True,
                              num_workers=4)
valid_loader = NeighborLoader(data, num_neighbors=[-1], input_nodes=data.valid_mask, batch_size=4096, shuffle=False,
                              num_workers=4)


class Net(torch.nn.Module):
    def __init__(self, hidden_size, heads, key_type):
        super(Net, self).__init__()
        self.x_type = args.x_type
        if concat_x is not None:
            in_size = data.x.size(1) - concat_x.size(1)
        else:
            in_size = data.x.size(1)
        self.con1 = modified_GAT(in_size, hidden_size, heads=heads, att_norm=True, key_type=key_type)
        # self.con3 = modified_GAT(hidden_size * heads, 2, heads=1, att_norm=True, key_type=key_type)

        self.lin1 = torch.nn.Linear(in_size, hidden_size * heads)
        # self.lin3 = torch.nn.Linear(hidden_size * heads, 2)

        self.bn1 = torch.nn.BatchNorm1d(in_size)
        self.bn3 = torch.nn.BatchNorm1d(hidden_size * heads)

        # self.con3 = modified_GAT(hidden_size * heads, hidden_size, heads=heads, att_norm=True, key_type=key_type)
        # self.lin3 = torch.nn.Linear(hidden_size * heads, hidden_size * heads)

        if concat_x is not None:
            self.con3 = modified_GAT(hidden_size * heads, hidden_size, heads=heads, att_norm=True, key_type=key_type)
            self.lin3 = torch.nn.Linear(hidden_size * heads, hidden_size * heads)
            self.bn = torch.nn.BatchNorm1d(concat_x.size(1))
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear((hidden_size * heads + concat_x.size(1)), hidden_size),
                # torch.nn.BatchNorm1d(hidden_size),
                torch.nn.ReLU(inplace=True),
                # Dropout(p=self.dropout),
                torch.nn.Linear(hidden_size, 2),
            )
        else:
            self.con3 = modified_GAT(hidden_size * heads, 2, heads=1, att_norm=True, key_type=key_type)
            self.lin3 = torch.nn.Linear(hidden_size * heads, 2)
        # self.read_out0 = torch.nn.Linear((hidden_size * heads + concat_x.size(1)), hidden_size)
        # self.read_out1 = torch.nn.Linear(hidden_size, 2)

        self.reset_parameters()

    def forward(self, x, edge_index, concat_x=None):
        x = self.bn1(x)
        x = F.relu(self.lin1(x) + self.con1(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.lin3(x) + self.con3(x, edge_index))
        if concat_x is not None:
            concat_x = self.bn(concat_x)
            x = torch.cat([x, concat_x], dim=1)
            x = self.mlp(x)
        return x

    def reset_parameters(self):
        self.con1.reset_parameters()
        self.con3.reset_parameters()

        self.lin1.reset_parameters()
        self.lin3.reset_parameters()
        # self.mlp.reset_parameters()


def train():
    # data.y is labels of shape (N, )
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch.batch_size
        if concat_x is not None:
            concat_size = batch.concat_size
            input_size = batch.x.size(1) - concat_size
            out = model(batch.x[:, :input_size], batch.edge_index, batch.x[:, input_size:])
        else:
            out = model(batch.x, batch.edge_index)
        # out = model(batch.x[:, :input_size], batch.edge_index, batch.x[:, input_size:])

        loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
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
        if concat_x is not None:
            concat_size = batch.concat_size
            input_size = batch.x.size(1) - concat_size
            out = model(batch.x.to(device)[:, :input_size], batch.edge_index.to(device),
                        batch.x.to(device)[:, input_size:])[:batch_size]
        else:
            out = model(batch.x.to(device), batch.edge_index.to(device))[:batch_size]
        # concat_size = batch.concat_size
        # input_size = batch.x.size(1) - concat_size
        # out = model(batch.x.to(device)[:, :input_size], batch.edge_index.to(device),
        #             batch.x.to(device)[:, input_size:])[:batch_size]
        preds.append(F.softmax(out, dim=1)[:, 1].cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()

    return roc_auc_score(y, pred)


model_dir = prepare_tune_folder('XYGraphP1', 'gat_cs' + 'pre_c&s')

model = Net(hidden_size, head, key_type).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
model.reset_parameters()
best_valid_auc = best_epoch = 0
for epoch in range(epoch_num):
    print('-----------------------------------------------------')
    print('For the {} epoch:'.format(epoch))
    loss = train()
    print('The train loss is {}.'.format(loss))
    c_valid_auc = valid()
    print('The valid auc is {}.'.format(c_valid_auc))
    if c_valid_auc > best_valid_auc:
        best_valid_auc = c_valid_auc
        best_epoch = epoch
        torch.save(model.state_dict(), model_dir + 'model.pt')
    print('-----------------------------------------------------')

print('-----------------------------------------------------')
print('The best valid auc is {}.'.format(best_valid_auc))
print('-----------------------------------------------------')


print('-----------------------------------------------------')
print('Predicting -------')
model.load_state_dict(torch.load(model_dir + 'model.pt'))
layer_loader = NeighborLoader(data, num_neighbors=[-1], input_nodes=None, batch_size=4096, shuffle=False,
                              num_workers=4)


@torch.no_grad()
def predict():
    model.eval()
    ys, preds = [], []
    for batch in tqdm(layer_loader):
        batch_size = batch.batch_size
        ys.append(batch.y[:batch_size])
        if concat_x is not None:
            concat_size = batch.concat_size
            input_size = batch.x.size(1) - concat_size
            out = model(batch.x.to(device)[:, :input_size], batch.edge_index.to(device),
                        batch.x.to(device)[:, input_size:])[:batch_size]
        else:
            out = model(batch.x.to(device), batch.edge_index.to(device))[:batch_size]
        preds.append(F.softmax(out, dim=1)[:, 1].cpu())

    pred = torch.cat(preds, dim=0).numpy()
    # y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return pred


out = predict()
np.save(os.path.join(model_dir, 'pred.npy'), out)




# ## correct and smooth
# edge_index = torch.from_numpy(edge_index)
# adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
#                      sparse_sizes=(dataset.num_papers, dataset.num_papers),
#                      is_sorted=True)
# adj_t = adj_t.to_symmetric()
# adj_t = gcn_norm(adj_t, add_self_loops=True)
#
# train_idx
