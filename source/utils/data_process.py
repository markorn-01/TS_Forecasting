import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing as pre

def get_data(csv_path):
    # get data from a given csv path.
    df = pd.read_csv(csv_path) 
    return df

def fill_missing_gold(df):
    first_valid_euro_am = df['EURO (AM)'].first_valid_index()
    df_cleaned = df.loc[first_valid_euro_am:].reset_index(drop=True)
    df_cleaned[['USD (PM)', 'GBP (AM)', 'GBP (PM)']] = df_cleaned[['USD (PM)', 'GBP (AM)', 'GBP (PM)']].interpolate(method='linear')
    df_cleaned['USD (AM)'] = df_cleaned['USD (AM)'].fillna(method='ffill')
    df_cleaned['EURO (PM)'] = df_cleaned['EURO (PM)'].fillna(method='bfill')
    return df_cleaned

def transfer_features_jena(df):
    df.loc[df['wv (m/s)'] == -9999.0] = 0.0
    df.loc[df['max. wv (m/s)'] == -9999.0] = 0.0
    return df
def prepare_data(df, label=None, features=[], date=None, format='%Y-%m-%d'):
    # set labels and features
    date_time = pd.to_datetime(df.pop(date), format=format)
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
#     print(day_sin, day_cos, month_sin, month_cos)
#     return day_sin, day_cos, month_sin, month_cos #, year_sin, year_cos

def create_time_features(date_time):
    # timestamp_s = date_time.map(pd.Timestamp.timestamp)
    # min_timestamp = timestamp_s.min()
    # max_timestamp = timestamp_s.max()
    # normalized_timestamp = (timestamp_s - min_timestamp) / (max_timestamp - min_timestamp)
    # day = 1.0
    # month = 12.0
    # day_sin = np.sin(normalized_timestamp * (2 * np.pi / day))
    # day_cos = np.cos(normalized_timestamp * (2 * np.pi / day))
    # month_sin = np.sin(normalized_timestamp * (2 * np.pi / month))
    # month_cos = np.cos(normalized_timestamp * (2 * np.pi / month))
    # # print(day_sin, day_cos, month_sin, month_cos)
    # return day_sin, day_cos, month_sin, month_cos
    date_time = pd.to_datetime(date_time)
    
    # Extract day of the month and month of the year
    day_of_month = date_time.dt.day
    month_of_year = date_time.dt.month
    
    # Calculate the total days in the month for each date
    days_in_month = date_time.dt.days_in_month

    # Compute the sin and cos transformations for day and month
    day_sin = np.sin(2 * np.pi * day_of_month / days_in_month)
    day_cos = np.cos(2 * np.pi * day_of_month / days_in_month)

    month_sin = np.sin(2 * np.pi * month_of_year / 12)
    month_cos = np.cos(2 * np.pi * month_of_year / 12)
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
    columns = train_df.columns
    train_df = train_df.values
    val_df = val_df.values
    test_df = test_df.values
    if method == 'minmax':
        scaler = pre.MinMaxScaler()
    else:
        scaler = pre.StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)
    train_df = pd.DataFrame(train_scaled, columns=columns)
    val_df = pd.DataFrame(val_scaled, columns=columns)
    test_df = pd.DataFrame(test_scaled, columns=columns)
    return train_df, val_df, test_df, scaler

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