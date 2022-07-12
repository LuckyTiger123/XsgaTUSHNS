import os
import sys
import math
import torch
import numpy as np
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch import Tensor
import torch_geometric.utils as utils


def add_feature_flag(x: Tensor):
    feature_flag = torch.zeros_like(x)
    feature_flag[x == -1] = 1
    x[x == -1] = 0
    return torch.cat((x, feature_flag), dim=1)


def add_degree_feature(x: Tensor, edge_index: Tensor):
    row, col = edge_index
    in_degree = utils.degree(col, x.size(0), x.dtype)
    out_degree = utils.degree(row, x.size(0), x.dtype)

    return torch.cat((x.T, in_degree.unsqueeze(0), out_degree.unsqueeze(0))).T


def add_id_feature(x: Tensor):
    id_feature = torch.tensor([range(x.size(0))], dtype=x.dtype)
    return torch.cat((x.T, id_feature)).T


def add_dynamic_degree(x: Tensor):
    dd_path = '/data/shangyihao/ppd/origin_dd.npy'
    dd = torch.from_numpy(np.load(dd_path)).to(dtype=x.dtype)
    return torch.cat((x, dd), dim=1)


def add_label_feature(x: Tensor, y: Tensor):
    y = torch.abs(y)
    y = torch.where(y > 4, 4, y)
    y_one_hot = F.one_hot(y)[:, :-1].to(dtype=x.dtype)
    return torch.cat((x, y_one_hot), dim=1)


def fold_timestamp(x: Tensor, fold_num: int = 50):
    if fold_num > 1:
        out_degree = x[:, [i * 2 for i in range(578)]]
        in_degree = x[:, [i * 2 + 1 for i in range(578)]]
        slices_num = math.ceil(578 / fold_num)
        result = None
        for i in range(slices_num):
            out_degree_c = out_degree[:, i * fold_num:(i + 1) * fold_num].sum(dim=1)
            in_degree_c = in_degree[:, i * fold_num:(i + 1) * fold_num].sum(dim=1)
            if result is None:
                result = out_degree_c.unsqueeze(1)
                result = torch.cat((result, in_degree_c.unsqueeze(1)), dim=1)
            else:
                result = torch.cat((result, out_degree_c.unsqueeze(1), in_degree_c.unsqueeze(1)), dim=1)
        return result
    else:
        return x


def to_undirected(edge_index: Tensor, edge_attr: Tensor):
    return utils.to_undirected(edge_index, edge_attr=edge_attr)


def degree_frequency(x: Tensor):
    x_out = x[:, 0::2]
    x_in = x[:, 1::2]
    index_tensor = torch.tensor([range(1, x_in.size(1) + 1)])

    degree_out = (x_out > 0).sum(dim=1)
    item_out = (x_out > 0) * index_tensor
    item_out_2 = torch.zeros(item_out.size())
    item_out_2[item_out == 0] = 1000
    item_out_2 = item_out_2 + item_out
    max_date = item_out.max(dim=1)[0]
    min_date = item_out_2.min(dim=1)[0]
    min_date[min_date == 1000] = 0
    out_r = (max_date - min_date) / (degree_out + 1e-9)
    out_r = out_r.reshape(-1, 1)

    degree_in = (x_in > 0).sum(dim=1)
    item_in = (x_in > 0) * index_tensor
    item_in_2 = torch.zeros(item_in.size())
    item_in_2[item_in == 0] = 1000
    item_in_2 = item_in_2 + item_in
    max_date = item_in.max(dim=1)[0]
    min_date = item_in_2.min(dim=1)[0]
    min_date[min_date == 1000] = 0
    in_r = (max_date - min_date) / (degree_in + 1e-9)
    in_r = in_r.reshape(-1, 1)

    result = torch.cat((out_r, in_r), dim=1)

    return result
