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


from our_models.modified_xygraph_yh import XYGraphP1
from our_models.load_data import fold_timestamp, to_undirected
from our_models.modified_GAT_yh import modified_GAT

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, default=7)
parser.add_argument('-t', '--train_round', type=int, default=1)
parser.add_argument('-f', '--file_id', type=int, default=0)
parser.add_argument('-e', '--epoch', type=int, default=30)
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

# feature generation
x = data.x[:, :37]
x_dtf = fold_timestamp(data.x[:, 41:], fold_num=30)
x = torch.cat((x, x_dtf), dim=1)
data.x = x
edge_index, edge_attr = to_undirected(data.edge_index, data.edge_attr)
# edge_index, edge_attr = data.edge_index, data.edge_attr
data.edge_index = edge_index
data.edge_attr = edge_attr

# hyperparamter list
# att_norm_list = [True, False]
key_type_list = [0, 1, 2]
learning_rate_list = [0.005, 0.001, 0.0005]
hidden_size_list = [256, 128, 64]
# head_list = [1]
head_list = [1, 2, 4]
# layer_list = [2, 3]
weight_decay_list = [5e-4, 1e-4]

# result statistics
# result_statistic = pd.DataFrame(
#     columns=['dataset', 'learning_rate', 'weight_decay', 'layer_num', 'hidden_size', 'att_norm', 'key_type',
#              'best_val_auc', 'average_val_auc', 'val_std'])

round_statistic = pd.DataFrame(
    columns=['dataset', 'round', 'learning_rate', 'weight_decay', 'layer_num', 'hidden_size', 'heads', 'att_norm',
             'key_type', 'best_val_auc', 'epoch_num'])

# neighborhood loader
train_loader = NeighborLoader(data, num_neighbors=[-1], input_nodes=data.train_mask, batch_size=1024, shuffle=True,
                              num_workers=12)
valid_loader = NeighborLoader(data, num_neighbors=[-1], input_nodes=data.valid_mask, batch_size=4096, shuffle=False,
                              num_workers=12)


class Net(torch.nn.Module):
    def __init__(self, hidden_size, heads, key_type):
        super(Net, self).__init__()
        self.con1 = modified_GAT(data.x.size(1), hidden_size, heads=heads, att_norm=True, key_type=key_type)
        # self.con2 = modified_GAT(hidden_size * heads, hidden_size, heads=heads, att_norm=True, key_type=key_type)
        self.con3 = modified_GAT(hidden_size * heads, 2, heads=1, att_norm=True, key_type=key_type)

        self.lin1 = torch.nn.Linear(data.x.size(1), hidden_size * heads)
        # self.lin2 = torch.nn.Linear(hidden_size * heads, hidden_size * heads)
        self.lin3 = torch.nn.Linear(hidden_size * heads, 2)

        self.bn1 = torch.nn.BatchNorm1d(data.x.size(1))
        # self.bn2 = torch.nn.BatchNorm1d(hidden_size * heads)
        self.bn3 = torch.nn.BatchNorm1d(hidden_size * heads)

        self.reset_parameters()

    def forward(self, x, edge_index):
        x = self.bn1(x)
        x = F.relu(self.lin1(x) + self.con1(x, edge_index))
        # x = self.bn2(x)
        # x = F.relu(self.lin2(x) + self.con2(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.lin3(x) + self.con3(x, edge_index))
        return x

    def reset_parameters(self):
        self.con1.reset_parameters()
        # self.con2.reset_parameters()
        self.con3.reset_parameters()

        self.lin1.reset_parameters()
        # self.lin2.reset_parameters()
        self.lin3.reset_parameters()


def train():
    # data.y is labels of shape (N, )
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch.batch_size
        out = model(batch.x, batch.edge_index)
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
        out = model(batch.x.to(device), batch.edge_index.to(device))[:batch_size]
        preds.append(F.softmax(out, dim=1)[:, 1].cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()

    return roc_auc_score(y, pred)


for learning_rate in learning_rate_list:
    for weight_decay in weight_decay_list:
        for hidden_size in hidden_size_list:
            for head in head_list:
                for key_type in key_type_list:
                    gc.collect()
                    model = Net(hidden_size, head, key_type).to(device)
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
                            'layer_num': 2,
                            'hidden_size': hidden_size,
                            'heads': head,
                            'att_norm': True,
                            'key_type': key_type,
                            'best_val_auc': round(best_valid_auc, 4),
                            'epoch_num': best_epoch}

save_path = os.path.join('..', 'result')
if not os.path.exists(save_path):
    os.makedirs(save_path)
round_statistic.to_excel(os.path.join(save_path, 'modified_GAT_2layer_{}.xlsx'.format(args.file_id)))
print('Mission completes.')
