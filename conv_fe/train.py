# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

from glob import glob

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

pd.options.display.max_columns = 20

exp_name = 'best_lgb_add_stampMean'

def concat_features(path):
    return pd.concat([pd.read_pickle(file) for file in glob(path)], axis=1)


train_features = concat_features('./xydata/feats_add_stampMean/*')
test_features = concat_features('./xydata/feats_add_stampMean/*')

train_data = pd.read_pickle('./xydata/proc/train_df.pkl')
test_data = pd.read_pickle('./xydata/proc/test_df.pkl')

train_data = train_data.merge(train_features, how='left', on=['user_id'])
test_data = test_data.merge(test_features, how='left', on=['user_id'])


features = [col for col in list(train_data) if col not in ['target']]

train_id = train_data[['user_id', 'target']].copy()
train_y = train_data['target'].values
train_x = train_data[features]

test_id = test_data[['user_id']].copy()
test_x = test_data[list(train_x)].copy()

# for col in ['as_father_edge_type_last']:
#     train_x[col] = train_x[col].astype('category')
#     test_x[col] = test_x[col].astype('category')
#

train_id[f'prediction'] = np.nan
test_id[f'prediction_1'] = 0

n_fold = 5
k_fold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=22)

num_boost_round = 15000
auc_list = []
test_preds = []
evals_result = {}

params = {
    'objective': 'binary',
    'metric': ['auc'],
    'boosting_type': 'gbdt',

    'learning_rate': 0.02,
    'max_depth': 7,
    'num_leaves': 32,
    # 'num_leaves': 2**4 -1,

    'min_child_weight': 10,
    'min_data_in_leaf': 40,
    #     'reg_lambda': 20,  # L2
    # 'reg_alpha': 120,  # L1

    'feature_fraction': 0.75,
    'subsample': 0.75,
    'seed': 114,

    'nthread': -1,
    'bagging_freq': 1,
    'verbose': -1,
}

print(train_x.head())
print(train_x.shape)

for fold, (tr_idx, val_idx) in enumerate(k_fold.split(train_x, train_y)):
    X_train, y_train, X_valid, y_valid = \
        train_x.loc[tr_idx], train_y[tr_idx], train_x.loc[val_idx], train_y[val_idx]

    d_train = lgb.Dataset(X_train, y_train)
    d_valid = lgb.Dataset(X_valid, y_valid, reference=d_train)

    gbm = lgb.train(params, d_train, num_boost_round,  # feval=amex_metric_mod_lgbm,
                    valid_sets=[d_train, d_valid], valid_names=['train', 'valid'],
                    evals_result=evals_result, verbose_eval=50, early_stopping_rounds=100)

    bst_round = np.argmax(evals_result['valid']['auc'])
    trn_score = evals_result['train']['auc'][bst_round]
    val_score = evals_result['valid']['auc'][bst_round]

    train_id.loc[val_idx, 'prediction'] = gbm.predict(X_valid)
    test_id['prediction_1'] += gbm.predict(test_x) / n_fold

    print(f'fold{fold}  AUC: {val_score}')
    auc_list.append(val_score)

    # feature_importance = pd.DataFrame(
    #     {'name': gbm.feature_name(), 'importance': gbm.feature_importance('gain')}). \
    #     sort_values(by='importance', ascending=False)
    # feature_importance.to_csv(f'../doc/fold{fold}_feat_importance.csv', index=False)


avg_cv_auc = round(np.mean(auc_list), 4)
print(f'CV AUC score: {avg_cv_auc}')

# records = pd.DataFrame(params, index=[0])
# records['exp_name'] = exp_name
# records['feature_num'] = train_x.shape[1]
# records['cv_score'] = avg_cv_auc
#
# records.to_csv('../lgb_exp_records.csv', index=False, mode='a', header=False)


# train_id.to_csv(f'../oof/oof_lgb_{exp_name}_cv{avg_cv_auc}.csv', index=False)

test_id['prediction_0'] = 1 - test_id['prediction_1']
np.save(f'./submit/submit_lgb_{exp_name}_cv{avg_cv_auc}.npy', test_id[['prediction_0', 'prediction_1']].values)
