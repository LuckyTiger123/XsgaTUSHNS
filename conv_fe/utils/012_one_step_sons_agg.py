# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import pandas as pd
import numpy as np
from tools import feat_last, feat_max, feat_std, feat_min, feat_mean, feat_sum, feat_nunique

pd.options.display.max_columns = 100

train_data = pd.read_pickle('../xydata/proc/train_df.pkl')
test_data = pd.read_pickle('../xydata/proc/test_df.pkl')

edge_df = pd.read_pickle('../xydata/proc/edge_df.pkl')

all_target_df = pd.read_pickle('../xydata/proc/all_df.pkl')
label_df = all_target_df[['user_id', 'target']].copy()

# 统计儿子的特征, from_id作为user_id, 找儿子的target和feature
all_target_df = all_target_df.rename(columns={'user_id': 'son_id'})
edge_df = edge_df.rename(columns={'from_id': 'user_id', 'to_id': 'son_id'})
edge_df = edge_df.merge(all_target_df, how='left', on=['son_id'])

# 特征和标签都是儿子的
edge_df.columns = [f'sons_{col}' if col not in ['user_id', 'son_id'] else col for col in edge_df.columns]
edge_df = edge_df[['user_id'] + [col for col in edge_df.columns if col != 'user_id']].copy()


def main(label, edge):

    history_df = label.merge(edge, how='left', on=['user_id'])
    history_df = history_df.sort_values(['user_id', 'sons_edge_timestamp'])

    # 筛选近期的关联
    # history_df = history_df[history_df['fathers_edge_timestamp'] > 0].reset_index(drop=True)

    # 用户有多少儿子
    label = feat_nunique(label, history_df, ['user_id'], 'son_id', name=f'son_id_total_nunique')

    # 用户有多少黑产儿子，分别统计有多少逾期，未逾期，以及2, 3的儿子
    #label = feat_nunique(label, history_df[history_df.sons_target == 0], ['user_id'], 'son_id', name=f'target0_son_id_nunique')
    #label = feat_nunique(label, history_df[history_df.sons_target == 1], ['user_id'], 'son_id', name=f'target1_son_id_nunique')
    label = feat_nunique(label, history_df[history_df.sons_target == 2], ['user_id'], 'son_id', name=f'target2_son_id_nunique')
    label = feat_nunique(label, history_df[history_df.sons_target == 3], ['user_id'], 'son_id', name=f'target3_son_id_nunique')
    label = feat_nunique(label, history_df[history_df.sons_target == -100], ['user_id'], 'son_id', name=f'target-100_son_id_nunique')

    # 用户有多少审批通过的儿子
    #mask = history_df.sons_target.isin([0, 1, -100])
    #label = feat_nunique(label, history_df[mask], ['user_id'], 'son_id', name=f'approved_son_id_nunique')

    # 各类儿子的占比
    for feat in ['target2_son_id_nunique',
                 'target3_son_id_nunique',
                 'target-100_son_id_nunique',]:
        label[f'{feat}_by_total'] = label[feat] / label['son_id_total_nunique']

    # 用户当父亲时，与他儿子主动建立联系的联系人类型的统计
    label = feat_mean(label, history_df, ['user_id'], 'sons_edge_type', name=f'sons_edge_type_mean')
    label = feat_min(label, history_df, ['user_id'], 'sons_edge_type', name=f'sons_edge_type_min')
    label = feat_max(label, history_df, ['user_id'], 'sons_edge_type', name=f'sons_edge_type_max')
    label = feat_std(label, history_df, ['user_id'], 'sons_edge_type', name=f'sons_edge_type_std')
    label = feat_nunique(label, history_df, ['user_id'], 'sons_edge_type', name=f'sons_edge_type_nunique')
    # 没用 label = feat_last(label, history_df, ['user_id'], 'sons_edge_type', name=f'sons_edge_type_last')

    # 用户当父亲时，与他儿子主动建立联系的timestamp统计
    label = feat_mean(label, history_df, ['user_id'], 'sons_edge_timestamp', name=f'sons_edge_time_mean')
    #label = feat_min(label, history_df, ['user_id'], 'sons_edge_timestamp', name=f'sons_edge_time_min')
    #label = feat_max(label, history_df, ['user_id'], 'sons_edge_timestamp', name=f'sons_edge_time_max')
    label = feat_last(label, history_df, ['user_id'], 'sons_edge_timestamp', name=f'sons_edge_time_last')
    # label = feat_std(label, history_df, ['user_id'], 'sons_edge_timestamp', name=f'sons_edge_time_std')

    # # 没用：用户当父亲时，近期与他儿子主动建立联系的联系人类型的统计
    # mask = history_df.sons_edge_timestamp <= 100
    # label = feat_mean(label, history_df[mask], ['user_id'], 'sons_edge_type', name=f'sons_edge_type_mean_t100')
    # label = feat_min(label, history_df[mask], ['user_id'], 'sons_edge_type', name=f'sons_edge_type_min_t100')
    # label = feat_max(label, history_df[mask], ['user_id'], 'sons_edge_type', name=f'sons_edge_type_max_t100')
    # label = feat_std(label, history_df[mask], ['user_id'], 'sons_edge_type', name=f'sons_edge_type_std_t100')

    # 用户儿子的原始特征的统计值  没用
    # agg_feat = [f'sons_feature_{i}' for i in range(17)]
    # label = feat_mean(label, history_df, ['user_id'], agg_feat)
    # label = feat_std(label, history_df, ['user_id'], agg_feat)

    # 用户儿子 son_id的统计
    # label = feat_mean(label, history_df, ['user_id'], 'son_id', name=f'son_id_mean')

    label = label.set_index(['user_id', 'target'])

    return label


# 生成特征
features_df = main(label_df, edge_df)

print(features_df.head())
print(features_df.shape)

features_df.to_pickle('../xydata/feats_add_stampMean/012_one_step_sons_agg.pkl')

