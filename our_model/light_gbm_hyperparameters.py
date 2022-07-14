import os
import sys
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from our_model.modified_xygraph import XYGraphP1
from our_model.load_data import fold_timestamp, to_undirected, degree_frequency
from our_model.faeture_propagation import feature_propagation
from our_models.extra import label_feature

change_to_directed = True
num_round = 200
k = 1
eps = 1
cuda_device = 5
file_id = 10

# device
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

# load data
dataset = XYGraphP1(root='/home/luckytiger/xinye_data_1', name='xydata')
print('Dataset load successfully...')
data = dataset[0]

# feature propagate
fp = feature_propagation(k=k, eps=eps).to(device)

# laod node2vec feature
# x_node2vec = torch.load(n2v_path).detach()

if change_to_directed:
    edge_index, edge_attr = to_undirected(data.edge_index, data.edge_attr)
else:
    edge_index, edge_attr = data.edge_index, data.edge_attr

# deal with the node feature
# out = fp(data.x[:, :37], edge_index).cpu()
out = fp(data.x[:, :17], edge_index, 1, data.x[:, 17:34])
# out_f = fp(data.x[:, :17], edge_index, 1, data.x[:, 17:34])
# out = torch.cat((out, out_f), dim=1)

x = data.x[:, 17:37]
x_back_label = data.x[:, 39:41]
x = torch.cat((x, x_back_label), dim=1)
x_f = fp(x, edge_index, 1)
# x_f2 = fp(x, edge_index, 2)
x = torch.cat((x, x_f), dim=1)

# x_dtf = fp(fold_timestamp(data.x[:, 41:], fold_num=30), edge_index, 2)
x_dtf = fold_timestamp(data.x[:, 41:], fold_num=30)
# x_dtf_f = fp(x_dtf, edge_index, 1)
# x_dtf = torch.cat((x_dtf, x_dtf_f), dim=1)

x_tg = degree_frequency(data.x[:, 41:])
x_tg_f = fp(degree_frequency(data.x[:, 41:]), edge_index, 1)
# x_tg_ff = fp(degree_frequency(data.x[:, 41:]), edge_index, 2)
x_tg = torch.cat((x_tg, x_tg_f), dim=1)

x = torch.cat((out, x, x_dtf, x_tg), dim=1)
data.x = x

data = label_feature(data)

data.edge_index = edge_index
data.edge_attr = edge_attr

params_test2 = {'max_bin': range(5, 256, 10), 'min_data_in_leaf': range(1, 102, 10)}

gsearch1 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.05,
                                 n_estimators=200, max_depth=5, num_leaves=30, bagging_fraction=0.6, lambda_l1=0.9,
                                 lambda_l2=1, feature_fraction=0.8, max_bin=185, min_data_in_leaf=81),
    param_grid=params_test2, scoring='roc_auc', cv=5, n_jobs=-1)

gsearch1.fit(data.x[data.train_mask].numpy(), data.y[data.train_mask].numpy())

print(gsearch1.cv_results_)

print('------------------------------------------')
print(gsearch1.best_score_)

print(gsearch1.best_params_)
