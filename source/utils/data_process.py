import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing as pre

def get_data(csv_path):
    # get data from a given csv path.
    df = pd.read_csv(csv_path) 
    return df

def prepare_data(df, label=None, features=[], date=None):
    # set labels and features
    date_time = pd.to_datetime(df.pop(date))
    df[df.isna()] = 0.00
    if features == None:
        return df[[label]] , date_time
    df.reindex(columns=features)
    df = df[[label] + features]
    return  df, date_time

# def create_time_features(date_time):
#     timestamp_s = date_time.map(pd.Timestamp.timestamp)
#     day = 24*60*60
#     month = 30 * day
#     year = (365.2425)*day
#     day_sin = np.sin(timestamp_s * (2 * np.pi / day))
#     day_cos = np.cos(timestamp_s * (2 * np.pi / day))
#     month_sin = np.sin(timestamp_s * (2 * np.pi / month))
#     month_cos = np.cos(timestamp_s * (2 * np.pi / month))
#     # year_sin = np.sin(timestamp_s * (2 * np.pi / year))
#     # year_cos = np.cos(timestamp_s * (2 * np.pi / year))
#     return day_sin, day_cos, month_sin, month_cos #, year_sin, year_cos

def create_time_features(date_time):
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    min_timestamp = timestamp_s.min()
    max_timestamp = timestamp_s.max()
    normalized_timestamp = (timestamp_s - min_timestamp) / (max_timestamp - min_timestamp)
    day = 1.0
    month = 12.0
    day_sin = np.sin(normalized_timestamp * (2 * np.pi / day))
    day_cos = np.cos(normalized_timestamp * (2 * np.pi / day))
    month_sin = np.sin(normalized_timestamp * (2 * np.pi / month))
    month_cos = np.cos(normalized_timestamp * (2 * np.pi / month))
    
    return day_sin, day_cos, month_sin, month_cos

def add_time(df, date_time):
    # add time features and set date_time as index
    # df['Month sin'], df['Month cos'], \
    # df['Year sin'], df['Year cos'] = create_time_features(date_time)
    df['Day sin'], df['Day cos'], \
    df['Month sin'], df['Month cos'] = create_time_features(date_time)
    df.set_index(date_time, inplace=True)
    return df

def split_data(df):
    # split dataset into train, validation, and test (80%, 10%, 10%)
    n = len(df)
    train_df = df[0: int(n*0.8)]
    val_df = df[int(n*0.8): int(n*0.9)]
    test_df = df[int(n*0.9):]
    num_features = df.shape[1]
    return train_df, val_df, test_df, num_features

def normalize_data(train_df, val_df, test_df, 
                   method='minmax'):
    # normalize the data
    if method == 'minmax':
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
    if method == 'meanstd':
        train_mean = train_df.mean()
        train_std = train_df.std()
        train_df = (train_df - train_mean) / train_std
        val_df = (val_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std
    return train_df, val_df, test_df

def show_result(model, val_performance, performance, 
                metric_name, y_label):
    x = np.arange(len(performance))
    width = 0.3
    metric_index = model.metrics_names.index(metric_name)
    val = [v[metric_index] for v in val_performance.values()]
    test = [v[metric_index] for v in performance.values()]
    plt.ylabel(y_label)
    plt.bar(x - 0.17, val, width, label='Validation')
    plt.bar(x + 0.17, test, width, label='Test')
    plt.xticks(ticks=x, 
               labels=performance.keys(),
               rotation=45)
    plt.legend()
    plt.show()