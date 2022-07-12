import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader


from our_models.modified_xygraph_yh import XYGraphP1
from our_models.load_data import fold_timestamp, to_undirected
from our_models.faeture_propagation import feature_propagation
from our_models.modified_GAT_yh import modified_GAT


cuda_device = 9
epoch_number = 30
heads = 4
att_norm = True
key_type = 0
hidden_size = 256

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
x_dtf = fold_timestamp(data.x[:, 41:], fold_num=30)
x = torch.cat((x, x_dtf), dim=1)
data.x = x

# label = data.y[torch.cat((data.train_mask, data.valid_mask))]
# class_weight = len(label) / (2 * np.bincount(label))
# class_weight = torch.tensor(class_weight, dtype=torch.float).to(device)

# edge_index, edge_attr = to_undirected(data.edge_index, data.edge_attr)
edge_classes = torch.from_numpy(np.load(os.path.join('/data/shangyihao/ppd', 'edge_classes_directed.npy')))
edge_feat = torch.load(os.path.join('/data/shangyihao/ppd', 'edge_feat_all.pt'))
new_time = torch.from_numpy(np.load(os.path.join('/data/shangyihao/ppd', 'new_time.npy')))
edge_index, edge_attr = data.edge_index, data.edge_attr
data.edge_index = edge_index
edge_all = torch.cat([new_time.unsqueeze(1), edge_feat[:, 1:], edge_classes.unsqueeze(1)], dim=-1)
data.edge_attr = edge_all


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convs1 = torch.nn.ModuleList([
            modified_GAT(data.x.size(1), hidden_size, heads=heads, att_norm=att_norm,
                         key_type=key_type
                         # , kq_dim=hidden_size
                         # , edge_dim=edge_all.size(1)-2
                         , use_time=True
                         )
            for _ in range(4)
        ])
        # self.convs2 = torch.nn.ModuleList([
        #     modified_GAT(hidden_size * heads, hidden_size, heads=heads, att_norm=att_norm, key_type=key_type)
        #     for _ in range(3)
        # ])
        self.convs3 = torch.nn.ModuleList([
            modified_GAT(hidden_size * heads, 2, heads=1, att_norm=att_norm,
                         key_type=key_type
                         # , kq_dim=2
                         # , edge_dim=edge_all.size(1)-2
                         , use_time=True
                         )
            for _ in range(4)
        ])
        self.lin1 = torch.nn.Linear(data.x.size(1), hidden_size * heads)
        # # self.lin2 = torch.nn.Linear(hidden_size * heads, hidden_size * heads)
        self.lin3 = torch.nn.Linear(hidden_size * heads, 2)
        #
        self.bn1 = torch.nn.BatchNorm1d(data.x.size(1))
        # # self.bn2 = torch.nn.BatchNorm1d(hidden_size * heads)
        self.bn3 = torch.nn.BatchNorm1d(hidden_size * heads)

        # self.read_out = torch.nn.Linear(hidden_size * heads, 2)

        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        # time = edge_attr[:, 0]
        edge_feature = edge_attr[:, :-1].float()
        edge_class = edge_attr[:, -1]
        x = self.bn1(x)
        x0 = self.lin1(x)
        for i in range(4):
            idx = edge_class == i
            sub_edge = edge_index[:, idx]
            sub_edge_feature = edge_feature[idx]
            x0 += self.convs1[i](x, sub_edge, sub_edge_feature)
        x = F.relu(x0)
        # x = self.read_out(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # x = self.bn2(x)
        # x0 = self.lin2(x)
        # for i in range(3):
        #     idx = edge_attr == i
        #     sub_edge = edge_index[:, idx]
        #     x0 += self.convs2[i](x, sub_edge)
        # x = F.relu(x0)
        # x = F.dropout(x, p=0.5, training=self.training)

        x = self.bn3(x)
        x0 = self.lin3(x)
        for i in range(4):
            idx = edge_class == i
            sub_edge = edge_index[:, idx]
            sub_edge_feature = edge_feature[idx]
            x0 += self.convs3[i](x, sub_edge, sub_edge_feature)
        x = F.relu(x0)
        # x = F.dropout(x, p=0.5, training=self.training)

        # x = self.bn1(x)
        # x = F.relu(self.lin1(x) + self.con1(x, edge_index))
        # # x = self.bn2(x)
        # # x = F.relu(self.lin2(x) + self.con2(x, edge_index))
        # x = self.bn3(x)
        # x = F.relu(self.lin3(x) + self.con3(x, edge_index))
        return x

    def reset_parameters(self):
        self.lin3.reset_parameters()
        # self.lin2.reset_parameters()
        self.lin1.reset_parameters()
        for con in self.convs1:
            con.reset_parameters()
        # for con in self.convs2:
        #     con.reset_parameters()
        for con in self.convs3:
            con.reset_parameters()
        # self.read_out.reset_parameters()


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)


def train():
    # data.y is labels of shape (N, )
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch.batch_size
        out = model(batch.x, batch.edge_index, batch.edge_attr)
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
        out = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device))[:batch_size]
        preds.append(F.softmax(out, dim=1)[:, 1].cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()

    return roc_auc_score(y, pred)


train_loader = NeighborLoader(data, num_neighbors=[-1]*2, input_nodes=data.train_mask, batch_size=1024, shuffle=True,
                              num_workers=4)
valid_loader = NeighborLoader(data, num_neighbors=[-1]*2, input_nodes=data.valid_mask, batch_size=4096, shuffle=False,
                              num_workers=4)

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

    if c_valid_auc > best_valid_auc:
        best_valid_auc = c_valid_auc
    print('-----------------------------------------------------')

print('-----------------------------------------------------')
print('The best valid auc is {}.'.format(best_valid_auc))
print('-----------------------------------------------------')
