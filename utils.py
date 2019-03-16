import pandas as pd

def list_to_one_hot(data: pd.DataFrame, col_name: str, index_name: str) -> pd.DataFrame:
    unlist = data.loc[:, col_name].apply(pd.Series) \
    .set_index(data.loc[:, index_name]) \
    .stack() \
    .reset_index(level=1, drop=True) \
    .to_frame(col_name)
    one_hot = pd.get_dummies(unlist) \
    .groupby(['index']).agg('sum')  \
    .reset_index()
    return one_hot