# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import pandas as pd
import os
from tools import feat_last, feat_max, feat_std, feat_min, feat_mean, feat_sum, feat_nunique

pd.options.display.max_columns = 100

train_data = pd.read_pickle('../xydata//proc/train_df.pkl')
test_data = pd.read_pickle('../xydata//proc/test_df.pkl')

edge_df = pd.read_pickle('../xydata//proc/edge_df.pkl')

all_target_df = pd.read_pickle('../xydata//proc/all_df.pkl')
# 没用 all_target_df = all_target_df.replace(-1, np.nan)
label_df = all_target_df[['user_id', 'target']].copy()

# 统计父亲的特征, to_id作为user_id, 找父亲的target和feature
all_target_df = all_target_df.rename(columns={'user_id': 'father_id'})
edge_df = edge_df.rename(columns={'to_id': 'user_id', 'from_id': 'father_id'})
print("before edge_df---------------------------\n", edge_df)
edge_df = edge_df.merge(all_target_df, how='left', on=['father_id'])


# 特征和标签都是父亲的
edge_df.columns = [f'fathers_{col}' if col not in ['user_id', 'father_id'] else col for col in edge_df.columns]
edge_df = edge_df[['user_id'] + [col for col in edge_df.columns if col != 'user_id']].copy()

def main(label, edge):
    history_df = label.merge(edge, how='left', on=['user_id'])
    history_df = history_df.sort_values(['user_id', 'fathers_edge_timestamp'])
    print(history_df)
    # 筛选近期的关联
    # history_df = history_df[history_df['fathers_edge_timestamp'] > 0].reset_index(drop=True)

    # 有多少父亲
    label = feat_nunique(label, history_df, ['user_id'], 'father_id', name=f'father_id_total_nunique')

    # 用户有多少黑产父亲，统计分别有多少逾期，未逾期，以及2, 3的父亲
        #label = feat_nunique(label, history_df[history_df.fathers_target == 0], ['user_id'], 'father_id', name=f'target0_father_id_nunique')
        #label = feat_nunique(label, history_df[history_df.fathers_target == 1], ['user_id'], 'father_id', name=f'target1_father_id_nunique')
    label = feat_nunique(label, history_df[history_df.fathers_target == 2], ['user_id'], 'father_id', name=f'target2_father_id_nunique')
    label = feat_nunique(label, history_df[history_df.fathers_target == 3], ['user_id'], 'father_id', name=f'target3_father_id_nunique')
    label = feat_nunique(label, history_df[history_df.fathers_target == -100], ['user_id'], 'father_id', name=f'target-100_father_id_nunique')

    # 有多少审批通过的父亲#前景节点
        #mask = history_df.fathers_target.isin([0, 1, -100])
        #label = feat_nunique(label, history_df[mask], ['user_id'], 'father_id', name=f'approved_father_id_nunique')

    # 各类父亲的占比
    for feat in ['target2_father_id_nunique', 'target3_father_id_nunique',
                 'target-100_father_id_nunique']:
        label[f'{feat}_by_total'] = label[feat] / label['father_id_total_nunique']

    # 用户当儿子时, 他的父亲与他建立联系的联系人类型的统计
    label = feat_mean(label, history_df, ['user_id'], 'fathers_edge_type', name=f'fathers_edge_type_mean')
    label = feat_min(label, history_df, ['user_id'], 'fathers_edge_type', name=f'fathers_edge_type_min')
    label = feat_max(label, history_df, ['user_id'], 'fathers_edge_type', name=f'fathers_edge_type_max')
    label = feat_std(label, history_df, ['user_id'], 'fathers_edge_type', name=f'fathers_edge_type_std')
    label = feat_nunique(label, history_df, ['user_id'], 'fathers_edge_type', name=f'fathers_edge_type_nunique')
    # 没用 label = feat_last(label, history_df, ['user_id'], 'fathers_edge_type', name=f'fathers_edge_type_last')

    # 用户当儿子时，他的父亲与他建立联系的timestamp统计
    label = feat_mean(label, history_df, ['user_id'], 'fathers_edge_timestamp', name=f'fathers_edge_time_mean')
    #label = feat_min(label, history_df, ['user_id'], 'fathers_edge_timestamp', name=f'fathers_edge_time_min')
    #label = feat_max(label, history_df, ['user_id'], 'fathers_edge_timestamp', name=f'fathers_edge_time_max')
    label = feat_last(label, history_df, ['user_id'], 'fathers_edge_timestamp', name=f'fathers_edge_time_last')
    # 没用 label = feat_std(label, history_df, ['user_id'], 'fathers_edge_timestamp', name=f'fathers_edge_time_std')

    # 用户父亲的原始特征的统计值
    agg_feat = [f'fathers_feature_{i}' for i in range(17)]
    label = feat_mean(label, history_df, ['user_id'], agg_feat)
    label = feat_std(label, history_df, ['user_id'], agg_feat)

    # # 用户父亲 father_id的统计
    # 没用 label = feat_mean(label, history_df, ['user_id'], 'father_id', name=f'father_id_mean')

    label = label.set_index(['user_id', 'target'])

    return label


# 生成特征
features_df = main(label_df, edge_df)

print("_______________________________________")
print(features_df.head())
print(features_df.shape)

os.makedirs('../xydata/feats_add_stampMean', exist_ok=True)
features_df.to_pickle('../xydata/feats_add_stampMean/011_one_step_fathers_agg.pkl')

