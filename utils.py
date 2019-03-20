import pandas as pd
import numpy as np
import pandas as pd
import warnings

def check_singular_columns(
                        training: pd.DataFrame,
                        test: pd.DataFrame
) -> None:
    """
    Checks whether there ar repeated columns betwwen two data sets.
    :param training:
    :param test:
    :return:
    """
    print("Columns in the training set that are not found in the test set: ",
          test.columns.difference(training.columns), sep="\n")
    print("Columns in the test set that are not found in the training set: ",
        test.columns.difference(training.columns), sep="\n")

def type_preprocessing(
                    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Changes types of columns required for further analysis.
    :param data:
    :return:
    """
    data.loc[:, 'ram_wielkosc'] = data.loc[:, 'ram_wielkosc'].apply(lambda x: float(x.split()[0]) if x is not None else x)
    data.loc[:, 'cpu_rdzenie'] = data.loc[:, 'cpu_rdzenie'].apply(lambda x: float(x) if x is not None else x)
    return data


def missing_values(
                    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Returns pandas Data Frame with number of missing values and their percentage part in data.
    :param data:
    :return:
    """
    missing = data.isnull().sum()
    percent = data.isnull().sum()/data.isnull().count()
    data_missing = pd.concat([missing, percent], axis=1, keys=['Total', 'Percent'], \
                             sort=True).sort_values(by='Total', ascending=False)
    print(data_missing.head(6))
    return data_missing

def list_to_one_hot(
                    data: pd.DataFrame,
                    col_name: str,
                    index_name: str
) -> pd.DataFrame:
    """
    Convert values inside columns from list type to one hot encoded columns.
    :param data:
    :param col_name:
    :param index_name:
    :return:
    """
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

def get_redundant_pairs(
                        data_frame: pd.DataFrame
) -> set:
    """
    Gets diagonal and lower triangular pairs of correlation matrix.
    :param data_frame:
    :return:
    """
    pairs_to_drop = set()
    cols = data_frame.columns
    for index in range(0, data_frame.shape[1]):
        for inside_index in range(0, index+1):
            pairs_to_drop.add((cols[index], cols[inside_index]))
    return pairs_to_drop

def get_top_abs_correlations(
                            data_frame: pd.DataFrame,
                            num_of_features: int = 5
) -> list:
    """
    Return correlation pairs with highest correlation value.
    :param data_frame:
    :param num_of_features:
    :return:
    """
    au_corr = data_frame.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(data_frame)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[:num_of_features]

