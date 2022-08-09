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

father_df = edge_df.rename(columns={'to_id': 'user_id', 'from_id': 'father_id', 'edge_type': 'edge_type_1', 'edge_timestamp': 'edge_timestamp_1'}).copy()
son_df = edge_df.rename(columns={'from_id': 'user_id', 'to_id': 'son_id', 'edge_type': 'edge_type_2', 'edge_timestamp': 'edge_timestamp_2'}).copy()
father_son_df = pd.merge(father_df,
                         son_df.rename(columns={'user_id': 'father_id', 'son_id': 'father_son_id', 'edge_type_1': 'edge_type_2', 'edge_timestamp_1': 'edge_timestamp_2'}),
                         how='inner', on=['father_id'])

all_target_df = pd.read_pickle('../xydata/proc/all_df.pkl')
label_df = all_target_df[['user_id', 'target']].copy()

# 统计用户拥有相同父亲的儿子的特征, 找father_son_id的target和feature
all_target_df = all_target_df.rename(columns={'user_id': 'father_son_id'})
father_son_df = father_son_df.merge(all_target_df, how='left', on=['father_son_id'])

# 特征和标签都是father_son_的，也就是拥有相同父亲的儿子们
father_son_df.columns = [f'father_sons_{col}'
                         if col not in ['user_id', 'father_son_id', 'father_id', 'son_id', 'edge_type_1', 'edge_type_2', 'edge_timestamp_1', 'edge_timestamp_2']
                         else col for col in father_son_df.columns]
father_son_df = father_son_df[['user_id'] + [col for col in father_son_df.columns if col != 'user_id']].copy()

father_son_df = father_son_df[father_son_df.user_id != father_son_df.father_son_id].reset_index(drop=True)


def main(label, edge):

    history_df = label.merge(edge, how='left', on=['user_id'])
    history_df = history_df.sort_values(['user_id', 'edge_timestamp_1', 'edge_timestamp_2'])

    # 筛选近期的关联
    # history_df = history_df[history_df['fathers_edge_timestamp'] > 0].reset_index(drop=True)

    # 用户有多少相同爸爸的小伙伴
    label = feat_nunique(label, history_df, ['user_id'], 'father_son_id', name=f'father_son_id_total_nunique')

    # 用户有多少黑产小伙伴，统计分别有多少逾期，未逾期，以及2, 3的小伙伴儿子
        #label = feat_nunique(label, history_df[history_df.father_sons_target == 0], ['user_id'], 'father_son_id',
        #                 name=f'target0_father_son_id_nunique')
        #label = feat_nunique(label, history_df[history_df.father_sons_target == 1], ['user_id'], 'father_son_id',
        #                 name=f'target1_father_son_id_nunique')
    label = feat_nunique(label, history_df[history_df.father_sons_target == 2], ['user_id'], 'father_son_id',
                        name=f'target2_father_son_id_nunique')
    label = feat_nunique(label, history_df[history_df.father_sons_target == 3], ['user_id'], 'father_son_id',
                         name=f'target3_father_son_id_nunique')
    label = feat_nunique(label, history_df[history_df.father_sons_target == -100], ['user_id'], 'father_son_id',
                         name=f'target-100_father_son_id_nunique')

    # 有多少审批通过的小伙伴
        #mask = history_df.father_sons_target.isin([0, 1, -100])
        #label = feat_nunique(label, history_df[mask], ['user_id'], 'father_son_id', name=f'approved_father_son_id_nunique')

    # 各类小伙伴的占比
    for feat in [ 'target2_father_son_id_nunique',
                 'target3_father_son_id_nunique',
                 'target-100_father_son_id_nunique', ]:
        label[f'{feat}_by_total'] = label[feat] / label['father_son_id_total_nunique']

    label = label.set_index(['user_id', 'target'])

    return label


# 生成特征
features_df = main(label_df, father_son_df)

print(features_df.head())
print(features_df.shape)

features_df.to_pickle('../xydata/feats_add_last_del_01approved/022_two_step_father_sons_agg.pkl')


