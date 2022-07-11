import os
import sys
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from our_model.modified_xygraph import XYGraphP1
from our_model.load_data import fold_timestamp, to_undirected, degree_frequency
from our_model.faeture_propagation import feature_propagation

change_to_directed = True
num_round = 300

# load data
dataset = XYGraphP1(root='/home/luckytiger/xinye_data_1', name='xydata')
print('Dataset load successfully...')
data = dataset[0]

# deal with the node feature
x = data.x[:, :200]
x_back_label = data.x[:, 202:204]
x = torch.cat((x, x_back_label), dim=1)
x_dtf = fold_timestamp(data.x[:, 204:], fold_num=30)
x_tg = degree_frequency(data.x[:, 204:])
x = torch.cat((x, x_dtf, x_tg), dim=1)
# x = torch.cat((x, x_dtf), dim=1)
data.x = x
if change_to_directed:
    edge_index, edge_attr = to_undirected(data.edge_index, data.edge_attr)
else:
    edge_index, edge_attr = data.edge_index, data.edge_attr
data.edge_index = edge_index
data.edge_attr = edge_attr

params = {
    'objective': 'multiclass',
    # 'objective': 'cross_entropy',
    'num_class': 2,
    # 'max_depth': 6,
    'num_threads': 4,
    # 'device_type': 'gpu',
    # 'gpu_device_id': 6,
    'seed': 0,
    'min_split_gain': 0.1,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.8,
    'lambda_l2': 2,
    # 'learning_rate': 0.05,
    # 'is_unbalance': True
}

train_data = lgb.Dataset(data=data.x[data.train_mask].numpy(), label=data.y[data.train_mask].numpy())
valid_data = lgb.Dataset(data=data.x[data.valid_mask].numpy(), label=data.y[data.valid_mask].numpy())

bst = lgb.train(params, train_data, num_round, valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(stopping_rounds=15)])

ypred = bst.predict(data.x[data.valid_mask].numpy(), num_iteration=bst.best_iteration)

test_auc = roc_auc_score(data.y[data.valid_mask].numpy(), ypred[:, 1])

print('The valid auc is {}.'.format(test_auc))
