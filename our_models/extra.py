import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch_geometric as tg
from torch_geometric.nn.conv import MessagePassing
from typing import Callable, Optional, Union
from torch import Tensor
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import remove_self_loops, degree, add_self_loops
from torch_scatter import scatter_add


class MyConv(MessagePassing):

    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class FeatureBlock(MessagePassing):
    def __init__(self, aggr='add', **kwargs):
        super(FeatureBlock, self).__init__(aggr=aggr, **kwargs)

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        h = self.propagate(edge_index, x=h, edge_weight=edge_weight)
        return h

    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__, self.out_channels, self.num_layers)


def label_norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    row, col = edge_index
    out_deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    in_deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    in_deg = in_deg + 1e-9
    out_deg = out_deg + 1e-9
    deg_inv_sqrt = 1 / (in_deg.sqrt())
    deg_out_sqrt = 1 / (out_deg.sqrt())
    return edge_index, deg_inv_sqrt[col] * deg_out_sqrt[row]


def label_norm1(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    row, col = edge_index
    out_deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    in_deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    in_deg = in_deg + 1e-9
    out_deg = out_deg + 1e-9
    deg_inv_sqrt = 1 / (in_deg.sqrt())
    deg_out_sqrt = 1 / (out_deg.sqrt())
    return edge_index, deg_inv_sqrt[col] * deg_out_sqrt[row], in_deg.view(-1, 1)


class Label_Extract(torch.nn.Module):
    def __init__(self):
        super(Label_Extract, self).__init__()
        self.conv1 = FeatureBlock('add')
        self.conv2 = FeatureBlock('add')

    def forward(self, data, is_direct):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        label = data.label
        x = label
        edge_index, norm, in_degree = label_norm1(edge_index, x.shape[0], None, dtype=x.dtype)
        edge_index1, norm1, out_degree = label_norm1(edge_index[[1, 0], :], x.shape[0], None, dtype=x.dtype)
        # norm_l = norm.view(-1,1)
        # norm_r = norm1.view(-1,1)
        h1 = self.conv1(x, edge_index[[1, 0], :], norm1)
        h2 = self.conv1(x, edge_index[[0, 1], :], norm)
        h3 = self.conv1(h2, edge_index[[1, 0], :], norm1)
        norm2 = norm * norm1
        re = self.conv1(torch.ones((x.shape[0], 1), dtype=torch.float), edge_index[[1, 0], :], norm2)
        h3 -= label * re
        # h3[h3<0.000001]-=h3[h3<0.000001]

        h4 = self.conv1(h1, edge_index[[0, 1], :], norm)
        norm2 = norm * norm1
        re = self.conv1(torch.ones((x.shape[0], 1), dtype=torch.float), edge_index[[0, 1], :], norm2)
        h4 -= label * re
        # h4[h4<0.000001]-=h4[h4<0.0000011]

        edge_index = tg.utils.to_undirected(edge_index)
        edge_index = tg.utils.remove_self_loops(edge_index)[0]
        # edge_weight = (edge_weight-edge_weight.min())/(edge_weight.max()-edge_weight.min()+1)
        edge_index, norm = label_norm(edge_index, x.shape[0], None, dtype=x.dtype)
        # edge_index2 = data.edge_index[[1,0],:]
        xx = None
        x1 = self.conv1(x, edge_index, norm)
        x2 = self.conv1(x1, edge_index, norm)
        re = self.conv1(torch.ones((x1.shape[0], 1), dtype=torch.float), edge_index, norm * norm)
        # x1 = self.conv1(x, edge_index,norm*norm)
        x2 = x2 - label * re
        x2 = x2 / (x2.sum(dim=1).view(-1, 1) + 1e-5)
        x1 = x1 / (x1.sum(dim=1).view(-1, 1) + 1e-5)
        h1 = h1 / (h1.sum(dim=1).view(-1, 1) + 1e-5)
        h2 = h2 / (h2.sum(dim=1).view(-1, 1) + 1e-5)
        h3 = h3 / (h3.sum(dim=1).view(-1, 1) + 1e-5)
        h4 = h4 / (h4.sum(dim=1).view(-1, 1) + 1e-5)
        return torch.cat([x2, x1, h3, h2, h1, h4], dim=1)


def label_feature(data):
    data = data.to('cpu')
    y = torch.zeros(data.x.shape[0], 4)
    y[:, 3] += (data.y == 3)
    y[:, 2] += (data.y == 2)
    y[:, 1] += (data.y == 1)
    y[:, 0] += (data.y == 0)
    y[data.test_mask] -= y[data.test_mask]
    # y[data.valid_mask] -= y[data.valid_mask]
    conv = Label_Extract()
    # data.edge_index = torch.cat([data.edge_index.coo()[0].view(1,-1),data.edge_index.coo()[1].view(1,-1)],dim=0)

    data.label = y
    y1 = conv(data, is_direct=False)
    data.x = torch.cat((data.x, y1), dim=1)
    return data
