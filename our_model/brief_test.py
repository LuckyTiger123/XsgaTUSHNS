import os
import sys
import torch
import numpy as np
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch_geometric.utils as utils

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from our_model.modified_xygraph import XYGraphP1
from our_model.load_data import fold_timestamp, degree_frequency

# cuda_device = 7
# device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')
#
# dataset = XYGraphP1(root='/home/luckytiger/xinye_data_1', name='xydata', transform=T.ToSparseTensor())
# data = dataset[0].to(device)

# file_path = '/home/luckytiger/xinye_data_1/phase1_gdata.npz'
# dataset = np.load(file_path)

# dd_path = '/home/luckytiger/xinye_data_1/origin_dd.npy'
# dd = torch.from_numpy(np.load(dd_path))
# print(dd.size())

# y = torch.abs(torch.from_numpy(dataset['y']))
# y = torch.where(y > 4, 4, y)
# y_one_hot = F.one_hot(y)
# print(y.size())
# print(y_one_hot[:, :-1].size())

# edge_type = torch.from_numpy(dataset['edge_type'])
# edge_timestamp = torch.from_numpy(dataset['edge_timestamp'])
# print(edge_type.size())
# print(edge_timestamp.size())
#
# edge_feature = torch.cat((edge_type.unsqueeze(1), edge_timestamp.unsqueeze(1)), dim=-1)
# print(edge_feature.size())

dataset = XYGraphP1(root='/home/luckytiger/xinye_data_1', name='xydata')
data = dataset[0]

x_back_label = data.x[:, 41:]

x = degree_frequency(x_back_label)
