import os
import sys
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


from our_models.modified_xygraph_yh import XYGraphP1
from our_models.load_data import fold_timestamp, to_undirected
from our_models.faeture_propagation import feature_propagation

cuda_device = 7
epoch_number = 500
k = 1
eps = 1

# device
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

# random seed
random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

dataset = XYGraphP1(root='/data/shangyihao/ppd', name='xydata')
data = dataset[0]

fp = feature_propagation(k=k, eps=eps).to(device)

x_raw_feature = data.x[:, :17].to(device)
x_raw_mask = data.x[:, 17:34].to(device)

edge_index, edge_attr = to_undirected(data.edge_index, data.edge_attr)
# edge_index, edge_attr = data.edge_index, data.edge_attr
edge_index = edge_index.to(device)
edge_attr = edge_attr.to(device)

# fix feature
x_fix = fp(x_raw_feature, x_raw_mask, edge_index)

# remain feature
x = data.x[:, 17:37]

x_dtf = fold_timestamp(data.x[:, 41:], fold_num=30)

x = torch.cat((x, x_dtf), dim=1).to(device)

# combine x
x = torch.cat((x_fix, x), dim=1)

data.x = x

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.007, weight_decay=5e-7)


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
