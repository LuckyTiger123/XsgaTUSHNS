# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

from datetime import timedelta

import numpy as np
import pandas as pd
from dateutil.parser import parse


def date_add_days(start_date, days):
    end_date = parse(start_date[:10]) + timedelta(days)
    end_date = end_date.strftime('%Y-%m-%d')
    return end_date


PrOriginalEp = np.zeros((2000, 2000))
PrOriginalEp[1, 0] = 1
PrOriginalEp[2, range(2)] = [0.5, 0.5]
for i in range(3, 2000):
    scale = (i-1)/2.
    x = np.arange(-(i+1)/2.+1, (i+1)/2., step=1)/scale
    y = 3./4.*(1-x**2)
    y = y/np.sum(y)
    PrOriginalEp[i, range(i)] = y
PrEp = PrOriginalEp.copy()
for i in range(3, 2000):
    PrEp[i, :i] = (PrEp[i, :i]*i+1)/(i+1)


def feat_kernel_median(df, df_feature, fe, value, pr, name=""):
    def get_median(a, pr=pr):
        a = np.array(a)
        x = a[~np.isnan(a)]
        n = len(x)
        weight = np.repeat(1.0, n)
        idx = np.argsort(x)
        x = x[idx]
        if n < pr.shape[0]:
            pr = pr[n, :n]
        else:
            scale = (n-1)/2.
            xxx = np.arange(-(n+1)/2.+1, (n+1)/2., step=1)/scale
            yyy = 3./4.*(1-xxx**2)
            yyy = yyy/np.sum(yyy)
            pr = (yyy*n+1)/(n+1)
        ans = np.sum(pr*x*weight) / float(np.sum(pr * weight))
        return ans

    df_count = pd.DataFrame(df_feature.groupby(fe)[value].apply(get_median)).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left")
    return df


def feat_std(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].agg({'std'})).reset_index()
    if not name:
        df_count.columns = fe + [f"_".join(col) for col in df_count.columns[1:]]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left")
    return df


def feat_max(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].agg({'max'})).reset_index()
    if not name:
        df_count.columns = fe + [f"_".join(col) for col in df_count.columns[1:]]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left")
    return df


def feat_min(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].agg({'min'})).reset_index()
    if not name:
        df_count.columns = fe + [f"_".join(col) for col in df_count.columns[1:]]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left")
    return df


def feat_nunique(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].agg({'nunique'})).reset_index()
    if not name:
        df_count.columns = fe + [f"_".join(col) for col in df_count.columns[1:]]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left")
    return df


def feat_count(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].agg({'count'})).reset_index()
    if not name:
        df_count.columns = fe + [f"_".join(col) for col in df_count.columns[1:]]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left")
    return df


def feat_sum(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].agg({'sum'})).reset_index()
    if not name:
        df_count.columns = fe + [f"_".join(col) for col in df_count.columns[1:]]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left")
    return df


def feat_mean(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].agg({'mean'})).reset_index()
    if not name:
        df_count.columns = fe + [f"_".join(col) for col in df_count.columns[1:]]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left")
    return df


def feat_last(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].agg({'last'})).reset_index()
    if not name:
        df_count.columns = fe + [f"_".join(col) for col in df_count.columns[1:]]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left")
    return df


def merge_nunique(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].agg({'nunique'})).reset_index()
    add.columns = columns+[cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_sum(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].sum()).reset_index()
    add.columns = columns+[cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_max(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].max()).reset_index()
    add.columns = columns+[cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_min(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].min()).reset_index()
    add.columns = columns+[cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_std(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].std()).reset_index()
    add.columns = columns+[cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_median(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].median()).reset_index()
    add.columns = columns+[cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_mean(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
    add.columns = columns+[cname]
    df = df.merge(add, on=columns, how="left")
    return df
