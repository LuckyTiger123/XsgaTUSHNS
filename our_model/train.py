import os
import sys
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from our_model.modified_xygraph import XYGraphP1
from our_models.load_data import fold_timestamp, to_undirected, degree_frequency
from our_model.faeture_propagation import feature_propagation
from our_models.extra import label_feature

cuda_device = 7
epoch_number = 500
k = 1
eps = 1
change_to_directed = True

# device
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

# random seed
random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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
label_feature = label_feature(data)

if change_to_directed:
    edge_index, edge_attr = to_undirected(data.edge_index, data.edge_attr)
else:
    edge_index, edge_attr = data.edge_index, data.edge_attr
data.edge_index = edge_index
data.edge_attr = edge_attr

data = data.to(device)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = torch.nn.Linear(data.x.size(1), 256)
        self.lin2 = torch.nn.Linear(256, 64)
        self.lin3 = torch.nn.Linear(64, 2)

        self.bn1 = torch.nn.BatchNorm1d(data.x.size(1))
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(64)

        self.reset_parameters()

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.lin1(x))
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.bn3(x)
        x = F.relu(self.lin3(x))
        return x

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-7)


def train():
    # data.y is labels of shape (N, )
    model.train()

    optimizer.zero_grad()

    out = model(data.x[data.train_mask])
    loss = F.cross_entropy(out, data.y[data.train_mask])

    # if no_conv:
    #     out = model(data.x[train_idx])
    # else:
    #     out = model(data.x, data.adj_t)[train_idx]
    # loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()
    print('The train loss is {}.'.format(loss))

    return


@torch.no_grad()
def valid():
    # data.y is labels of shape (N, )
    model.eval()

    # if no_conv:
    #     out = model(data.x)
    # else:
    #     out = model(data.x, data.adj_t)

    out = model(data.x[data.valid_mask])

    return roc_auc_score(data.y[data.valid_mask].cpu(), F.softmax(out, dim=1)[:, 1].cpu())


best_valid_auc = 0
for epoch in range(epoch_number):
    print('-----------------------------------------------------')
    print('For the {} epoch:'.format(epoch))
    train()
    c_valid_auc = valid()
    print('The valid auc is {}.'.format(c_valid_auc))

    if c_valid_auc > best_valid_auc:
        best_valid_auc = c_valid_auc
    print('-----------------------------------------------------')

print('-----------------------------------------------------')
print('The best valid auc is {}.'.format(best_valid_auc))
print('-----------------------------------------------------')
