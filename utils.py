import pandas as pd
import numpy as np
import pandas as pd
import warnings

def missing_values(data):
    missing = data.isnull().sum()
    percent = data.isnull().sum()/data.isnull().count()
    data_missing = pd.concat([missing, percent], axis=1, keys=['Total', 'Percent'], \
                             sort=True).sort_values(by='Total', ascending=False)
    print(data_missing.head(6))
    return data_missing

def list_to_one_hot(data: pd.DataFrame, col_name: str, index_name: str) -> pd.DataFrame:
    unlist = data.loc[:, col_name].apply(pd.Series) \
    .set_index(data.loc[:, index_name]) \
    .stack() \
    .reset_index(level=1, drop=True) \
    .to_frame(col_name)
    one_hot = pd.get_dummies(unlist) \
    .groupby(['index']).agg('sum')  \
    .astype('int') \
    .astype('Int64') \
    .reset_index()
    return one_hot

def get_redundant_pairs(data_frame: pd.DataFrame):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = data_frame.columns
    for index in range(0, data_frame.shape[1]):
        for inside_index in range(0, index+1):
            pairs_to_drop.add((cols[index], cols[inside_index]))
    return pairs_to_drop

def get_top_abs_correlations(data_frame: pd.DataFrame, num_of_features: int=5):
    au_corr = data_frame.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(data_frame)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[:num_of_features]

