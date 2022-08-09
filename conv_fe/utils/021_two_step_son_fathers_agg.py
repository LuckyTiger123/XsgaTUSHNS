# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import pandas as pd

from tools import feat_last, feat_max, feat_std, feat_min, feat_mean, feat_nunique

pd.options.display.max_columns = 100

train_data = pd.read_pickle('../xydata/proc/train_df.pkl')
test_data = pd.read_pickle('../xydata/proc/test_df.pkl')

init_cols = list(train_data)

edge_df = pd.read_pickle('../xydata/proc/edge_df.pkl')

son_df = edge_df.rename(columns={'from_id': 'user_id', 'to_id': 'son_id', 'edge_type': 'edge_type_1', 'edge_timestamp': 'edge_timestamp_1'}).copy()
father_df = edge_df.rename(columns={'to_id': 'user_id', 'from_id': 'father_id', 'edge_type': 'edge_type_2', 'edge_timestamp': 'edge_timestamp_2'}).copy()

son_father_df = pd.merge(son_df,
                         father_df.rename(columns={'user_id': 'son_id', 'father_id': 'son_father_id'}),
                         how='inner', on=['son_id'])

all_target_df = pd.read_pickle('../xydata/proc/all_df.pkl')
label_df = all_target_df[['user_id', 'target']].copy()

# 统计用户拥有相同儿子的父亲的特征, 找son_father_id的target和feature
all_target_df = all_target_df.rename(columns={'user_id': 'son_father_id'})
son_father_df = son_father_df.merge(all_target_df, how='left', on=['son_father_id'])

# 特征和标签都是son_father的，也就是拥有相同儿子的爸爸们
son_father_df.columns = [f'son_fathers_{col}'
                         if col not in ['user_id', 'son_father_id', 'father_id', 'son_id', 'edge_type_1', 'edge_type_2', 'edge_timestamp_1', 'edge_timestamp_2']
                         else col for col in son_father_df.columns]
son_father_df = son_father_df[['user_id'] + [col for col in son_father_df.columns if col != 'user_id']].copy()

son_father_df = son_father_df[son_father_df.user_id != son_father_df.son_father_id].reset_index(drop=True)

son_father_df['edge_timestamp_diff'] = abs(son_father_df['edge_timestamp_1'] - son_father_df['edge_timestamp_2'])
son_father_df['edge_type_two_step_same'] = (son_father_df['edge_type_1'] == son_father_df['edge_type_2']) * 1.0


def main(label, edge):

    history_df = label.merge(edge, how='left', on=['user_id'])
    history_df = history_df.sort_values(['user_id', 'edge_timestamp_1', 'edge_timestamp_2'])

    # 筛选近期的关联
    # history_df = history_df[history_df['fathers_edge_timestamp'] > 0].reset_index(drop=True)

    # 用户有多少相同儿子的朋友
    label = feat_nunique(label, history_df, ['user_id'], 'son_father_id', name=f'son_father_id_total_nunique')

    # 用户有多少黑产兄弟，统计分别有多少逾期，未逾期，以及2, 3的父亲
    #label = feat_nunique(label, history_df[history_df.son_fathers_target == 0], ['user_id'], 'son_father_id',
                         #name=f'target0_son_father_id_nunique')
    #label = feat_nunique(label, history_df[history_df.son_fathers_target == 1], ['user_id'], 'son_father_id',
                         #name=f'target1_son_father_id_nunique')
    label = feat_nunique(label, history_df[history_df.son_fathers_target == 2], ['user_id'], 'son_father_id',
                         name=f'target2_son_father_id_nunique')
    label = feat_nunique(label, history_df[history_df.son_fathers_target == 3], ['user_id'], 'son_father_id',
                         name=f'target3_son_father_id_nunique')
    label = feat_nunique(label, history_df[history_df.son_fathers_target == -100], ['user_id'], 'son_father_id',
                         name=f'target-100_son_father_id_nunique')

    # 有多少审批通过的兄弟
    #mask = history_df.son_fathers_target.isin([0, 1, -100])
    #label = feat_nunique(label, history_df[mask], ['user_id'], 'son_father_id', name=f'approved_son_fathers_id_nunique')

    # 各类兄弟的占比
    for feat in ['target2_son_father_id_nunique',
                 'target3_son_father_id_nunique',
                 'target-100_son_father_id_nunique']:
        label[f'{feat}_by_total'] = label[feat] / label['son_father_id_total_nunique']

    label = label.set_index(['user_id', 'target'])

    return label


# 生成特征
features_df = main(label_df, son_father_df)

print(features_df.head())
print(features_df.shape)

features_df.to_pickle('../xydata/feats_add_last_del_01approved/021_two_step_son_fathers_agg.pkl')


