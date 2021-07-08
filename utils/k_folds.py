from sklearn.model_selection import GroupKFold
import pandas as pd


def split_data(df_data1, df_data2, number_fold):
    if df_data2 is not None:
        df_data1 = pd.read_csv(df_data1, index_col=False)
        df_data2 = pd.read_csv(df_data2, index_col=False)
        data = pd.concat((df_data1, df_data2), axis=0)
    else:
        data = df_data1

    kf = GroupKFold(n_splits=number_fold)
    df_train = {}
    df_split = {}
    for fold, (train_index, test_index) in enumerate(kf.split(data, data, data.iloc[:, 0])):
        df_train[fold] = data.iloc[train_index].reset_index(drop=True)
        df_split[fold] = data.iloc[test_index].reset_index(drop=True)

    return df_train, df_split
