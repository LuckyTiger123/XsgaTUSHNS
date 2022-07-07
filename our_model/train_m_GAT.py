import os
import sys
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from our_model.modified_xygraph import XYGraphP1
from our_model.load_data import fold_timestamp, to_undirected
from our_model.faeture_propagation import feature_propagation
from our_model.modified_GAT import modified_GAT

cuda_device = 7
epoch_number = 300
heads = 2
att_norm = True
key_type = 0
hidden_size = 64

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
x_dtf = fold_timestamp(data.x[:, 41:], fold_num=30)
x = torch.cat((x, x_dtf), dim=1)
data.x = x

edge_index, edge_attr = to_undirected(data.edge_index, data.edge_attr)
# edge_index, edge_attr = data.edge_index, data.edge_attr
data.edge_index = edge_index.to(device)
data.edge_attr = edge_attr.to(device)

data = data.to(device)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.con1 = modified_GAT(data.x.size(1), hidden_size, heads=heads, att_norm=att_norm, key_type=key_type)
        self.con2 = modified_GAT(hidden_size * heads, hidden_size, heads=heads, att_norm=att_norm, key_type=key_type)
        self.con3 = modified_GAT(hidden_size * heads, 2, heads=1, att_norm=att_norm, key_type=key_type)

        self.lin1 = torch.nn.Linear(data.x.size(1), hidden_size * heads)
        self.lin2 = torch.nn.Linear(hidden_size * heads, hidden_size * heads)
        self.lin3 = torch.nn.Linear(hidden_size * heads, 2)

        self.bn1 = torch.nn.BatchNorm1d(data.x.size(1))
        self.bn2 = torch.nn.BatchNorm1d(hidden_size * heads)
        self.bn3 = torch.nn.BatchNorm1d(hidden_size * heads)

        self.reset_parameters()

    def forward(self, x, edge_index):
        x = self.bn1(x)
        x = F.relu(self.lin1(x) + self.con1(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.lin2(x) + self.con2(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.lin3(x) + self.con3(x, edge_index))
        return x

    def reset_parameters(self):
        self.con1.reset_parameters()
        self.con2.reset_parameters()
        self.con3.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-7)


def train():
    # data.y is labels of shape (N, )
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        


    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    print('The train loss is {}.'.format(loss))

    return


@torch.no_grad()
def valid():
    # data.y is labels of shape (N, )
    model.eval()
    out = model(data.x, data.edge_index)

    return roc_auc_score(data.y[data.valid_mask].cpu(), F.softmax(out[data.valid_mask], dim=1)[:, 1].cpu())


train_loader = NeighborLoader(data, num_neighbors=[-1], input_nodes=data.train_mask, batch_size=1024, shuffle=True,
                              num_workers=12)
valid_loader = NeighborLoader(data, num_neighbors=[-1], input_nodes=data.valid_mask, batch_size=4096, shuffle=False,
                              num_workers=12)

# train_loader = NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[10, 5], batch_size=1024, shuffle=True,
#                                num_workers=12)
# layer_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False,
#                                num_workers=12)

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
