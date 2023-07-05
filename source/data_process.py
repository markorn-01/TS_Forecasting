import numpy as np
import pandas as pd

def get_data(csv_path, label=None):
    #this path is used for testing only. Otherwise, use the path in the main function.
    csv_path = "data/gold/LBMA-GOLD.csv"
    df = pd.read_csv(csv_path) 
    date_time = pd.to_datetime(df.pop('Date'))
    df[df.isna()] = 0.00
    return df[[label]] , date_time    

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
    column_indices = {name: i for i, name in enumerate(df.columns)}
    n = len(df)
    train_df = df[0: int(n*0.8)]
    val_df = df[int(n*0.8): int(n*0.9)]
    test_df = df[int(n*0.9):]
    num_features = df.shape[1]
    return train_df, val_df, test_df, num_features

def normalize_data(train_df, val_df, test_df):
    #normalize the data
    train_mean = train_df.mean()
    train_std = train_df.std()
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    return train_df, val_df, test_df
         