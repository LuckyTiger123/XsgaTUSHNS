import os
import sys
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

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
file_id = 14

# device
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

# load data
dataset = XYGraphP1(root='/home/luckytiger/xinye_data_1', name='xydata')
print('Dataset load successfully...')
data = dataset[0]

# feature propagate
fp = feature_propagation(k=k, eps=eps)

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

params = {
    # 'objective': 'multiclass',
    # 'num_class': 2,
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 5,
    'num_leaves': 30,
    'num_threads': 4,
    'max_bin': 185,
    'min_data_in_leaf': 81,
    # 'max_bin': 195,
    # 'min_data_in_leaf': 61,
    'seed': file_id,
    'min_split_gain': 0.5,
    'bagging_fraction': 0.6,
    'feature_fraction': 0.8,
    'bagging_freq': 0,
    'lambda_l1': 0.9,
    'lambda_l2': 1.0,
    'learning_rate': 0.05,
    # 'scale_pos_weight': 1,
    # 'is_unbalance': True,
}

# train_data = lgb.Dataset(data=data.x[torch.cat((data.train_mask, data.valid_mask), dim=-1)].numpy(),
#                          label=data.y[torch.cat((data.train_mask, data.valid_mask), dim=-1)].numpy())
train_data = lgb.Dataset(data=data.x[data.train_mask].numpy(), label=data.y[data.train_mask].numpy())
valid_data = lgb.Dataset(data=data.x[data.valid_mask].numpy(), label=data.y[data.valid_mask].numpy())

cv = lgb.cv(params, train_data, num_round, nfold=5, stratified=False, shuffle=True, early_stopping_rounds=20,
            verbose_eval=50, show_stdv=True, return_cvbooster=True)

print('best n_estimators:', len(cv['auc-mean']))
print('best cv score:', pd.Series(cv['auc-mean']).max())

bst = cv['cvbooster']

# bst = lgb.train(params, train_data, num_round, valid_sets=[valid_data],
#                 callbacks=[lgb.early_stopping(stopping_rounds=15)])

# y_pred = bst.predict(data.x[data.valid_mask].numpy(), num_iteration=bst.best_iteration)


train_ypred = bst.predict(data.x[data.train_mask].numpy(), num_iteration=bst.best_iteration)
v_ypred = bst.predict(data.x[data.valid_mask].numpy(), num_iteration=bst.best_iteration)

v_ypred_list = []

for i in range(5):
    test_auc = roc_auc_score(data.y[data.valid_mask].numpy(), v_ypred[i])
    train_auc = roc_auc_score(data.y[data.train_mask].numpy(), train_ypred[i])
    print('The train auc is {}.'.format(train_auc))
    print('The valid auc is {}.'.format(test_auc))
    v_ypred_list.append(v_ypred[i])

ypred_mean = np.mean(np.array(v_ypred_list), axis=0)
np.save('../submit/series6_valid_{}.npy'.format(file_id), ypred_mean)
print('-----------------------------------------')
test_auc = roc_auc_score(data.y[data.valid_mask].numpy(), ypred_mean)
print('The valid auc is {}.'.format(test_auc))

t_ypred = bst.predict(data.x[data.test_mask].numpy(), num_iteration=bst.best_iteration)
t_ypred_list = []
for i in range(5):
    ypred = t_ypred[i]
    t_ypred_list.append(ypred)

t_ypred_agg = np.array(t_ypred_list)
t_ypred_mean = np.mean(t_ypred_agg, axis=0)

np.save('../submit/series6_test_{}.npy'.format(file_id), t_ypred_mean)
