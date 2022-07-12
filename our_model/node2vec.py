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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.nn import Node2Vec

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from our_model.modified_xygraph import XYGraphP1
from our_model.load_data import fold_timestamp, to_undirected, degree_frequency
from our_model.faeture_propagation import feature_propagation
from our_model.modified_GAT import modified_GAT

cuda_device = 6
change_to_directed = True
epoch_num = 100
walk_per_node = 10
hidden_size = 128
file_id = 0
walk_length = 20
context_size = 10

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

model = Node2Vec(data.edge_index, embedding_dim=hidden_size, walk_length=walk_length, context_size=context_size,
                 walks_per_node=walk_per_node, num_negative_samples=1, p=1, q=1, sparse=True).to(device)

loader = model.loader(batch_size=1024, shuffle=True, num_workers=12)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.valid_mask], data.y[data.valid_mask],
                     max_iter=150)
    return acc


for epoch in range(1, epoch_num + 1):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
    torch.save(model().cpu(),
               '../node2vec/random_walk_feature_{}_{}_{}_{}.pt'.format(walk_length, context_size, epoch_num, file_id))

# @torch.no_grad()
# def plot_points(colors):
#     model.eval()
#     z = model(torch.arange(data.x.size(0), device=device)[data.valid_mask])
#     z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
#     y = data.y.cpu().numpy()
#
#     plt.figure(figsize=(8, 8))
#     for i in range(dataset.num_classes):
#         plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
#     plt.axis('off')
#     plt.show()
#
#
# colors = [
#     '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
#     '#ffd700'
# ]
# plot_points(colors)

print('Mission completes.')
