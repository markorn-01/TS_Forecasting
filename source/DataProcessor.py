import numpy as np
import pandas as pd
from sklearn import preprocessing as pre

def get_data(csv_path):
    #this path is used for testing only. Otherwise, use the path in the main function.
    df = pd.read_csv(csv_path) 
    return df

def prepare_data(df, label=None, features=None, date=None):
    #set labels and features
    date_time = pd.to_datetime(df.pop(date), format='%Y-%m-%d')
    df[df.isna()] = 0.00
    if features == None:
        return df[[label]] , date_time
    return df[[label, features]] , date_time

def add_time(df, date_time):
    #add time features
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24*60*60
    year = (365.2425)*day
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    return df

def split_data(df):
    #split dataset into train, validation, and test (80%, 10%, 10%)
    n = len(df)
    train_df = df[0: int(n*0.8)]
    val_df = df[int(n*0.8): int(n*0.9)]
    test_df = df[int(n*0.9):]
    num_features = df.shape[1]
    return train_df, val_df, test_df, num_features

def normalize_data(train_df, val_df, test_df):
    #normalize the data
    # train_mean = train_df.mean()
    # train_std = train_df.std()
    # train_df = (train_df - train_mean) / train_std
    # val_df = (val_df - train_mean) / train_std
    # test_df = (test_df - train_mean) / 
    columns = train_df.columns
    train_df = train_df.values
    val_df = val_df.values
    test_df = test_df.values
    min_max_scaler = pre.MinMaxScaler()
    train_scaled = min_max_scaler.fit_transform(train_df)
    val_scaled = min_max_scaler.fit_transform(val_df)
    test_scaled = min_max_scaler.fit_transform(test_df)
    train_df = pd.DataFrame(train_scaled, columns=columns)
    val_df = pd.DataFrame(val_scaled, columns=columns)
    test_df = pd.DataFrame(test_scaled, columns=columns)
    return train_df, val_df, test_df
         