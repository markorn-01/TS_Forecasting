import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/jena_climate/jena_climate_2009_2016.csv")

# Assuming the CSV file has a "Date" column with date information and a "USD (AM)" column with the USD prices.
# We'll convert the "Date" column to a datetime type for plotting.
data['Date'] = pd.to_datetime(data['Date Time'])

# Filter the data for the desired date range (2015 to 2023)
start_date = '2009-01-01'
end_date = '2009-11-10'
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] < end_date)]
filtered_data = filtered_data[["Date Time",  "p (mbar)",  "T (degC)",  "Tpot (K)"]]
print(filtered_data)
# Create the plot

# Reload the data
# data = pd.DataFrame({
#     'Dataset': ['LBMA', None, None, None, None, 'Jena', None, None, None, None],
#     'Model': ['CNN', 'LSTM', 'Residual LSTM', 'LSTM - CNN', 'Transformer - CNN',
#               'CNN', 'LSTM', 'Residual LSTM', 'LSTM - CNN', 'Transformer - CNN'],
#     'Loss': [0.000166, 0.000567, 0.00022, 0.000109, 0.02184, 
#              0.000028, 0.000013, 0.000013, 0.000013, 0.000435],
#     'MAE': [0.010265, 0.017746, 0.011367, 0.008101, 0.133461, 
#             0.004075, 0.002345, 0.002485, 0.002465, 0.015121],
#     'RMSE': [0.012876, 0.023813, 0.014849, 0.010424, 0.147784, 
#              0.0053, 0.003542, 0.00366, 0.003631, 0.020868],
#     'MAPE': [1.073646, 1.836101, 1.177629, 0.85034, 13.59844, 
#              0.709105, 0.398333, 0.425988, 0.419826, 3.172732]
# })

# # Fill the Dataset column
# data['Dataset'] = data['Dataset'].fillna(method='ffill')

# # Metrics and titles for plotting
# metrics = ['Loss', 'MAE', 'RMSE', 'MAPE']
# titles = ['Loss', 'Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Percentage Error (MAPE)']

# # Plot and save the visualized metrics
# fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# for ax, metric, title in zip(axs.ravel(), metrics, titles):
#     data.pivot(index='Model', columns='Dataset', values=metric).plot(kind='bar', ax=ax)
#     ax.set_title(title + ' by Dataset')
#     ax.set_ylabel(metric)
#     ax.set_xlabel('Model')
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
#     ax.set_xticklabels(data['Model'].unique(), rotation=45, ha='right')

# plt.tight_layout()
# plt.savefig(file_path)
# plt.close()
# import matplotlib.pyplot as plt
# import datetime
# import numpy as np
# import pandas as pd
# import seaborn as sns
# # import tensorflow as tf
# from tensorflow.keras.models import load_model
# from utils.data_process import *
# from utils.window_generate import *
# from utils.model_generate import *


# # Load the Jena Climate dataset
# dataset_url = "data/jena_climate/jena_climate_2009_2016.csv"
# data = pd.read_csv(dataset_url)
# print(data.describe().transpose())
# plt.figure(figsize=(15, 10))

# # Histogram for USD
# plt.subplot(3, 2, 1)
# plt.hist(data['USD (AM)'].dropna(), bins=50, color='blue', alpha=0.7)
# plt.title('Distribution of Gold Prices in USD (AM)')
# plt.xlabel('Price in USD')
# plt.ylabel('Frequency')

# plt.subplot(3, 2, 2)
# plt.hist(data['USD (PM)'].dropna(), bins=50, color='red', alpha=0.7)
# plt.title('Distribution of Gold Prices in USD (PM)')
# plt.xlabel('Price in USD')
# plt.ylabel('Frequency')

# # Histogram for GBP
# plt.subplot(3, 2, 3)
# plt.hist(data['GBP (AM)'].dropna(), bins=50, color='blue', alpha=0.7)
# plt.title('Distribution of Gold Prices in GBP (AM)')
# plt.xlabel('Price in GBP')
# plt.ylabel('Frequency')

# plt.subplot(3, 2, 4)
# plt.hist(data['GBP (PM)'].dropna(), bins=50, color='red', alpha=0.7)
# plt.title('Distribution of Gold Prices in GBP (PM)')
# plt.xlabel('Price in GBP')
# plt.ylabel('Frequency')

# # Histogram for EURO
# plt.subplot(3, 2, 5)
# plt.hist(data['EURO (AM)'].dropna(), bins=50, color='blue', alpha=0.7)
# plt.title('Distribution of Gold Prices in EURO (AM)')
# plt.xlabel('Price in EURO')
# plt.ylabel('Frequency')

# plt.subplot(3, 2, 6)
# plt.hist(data['EURO (PM)'].dropna(), bins=50, color='red', alpha=0.7)
# plt.title('Distribution of Gold Prices in EURO (PM)')
# plt.xlabel('Price in EURO')
# plt.ylabel('Frequency')

# # Adjust layout for better visualization
# plt.tight_layout()
# plt.show()
# data['Date'] = pd.to_datetime(data['Date'])

# # Plotting the trend of gold prices over time for each currency
# plt.figure(figsize=(15, 10))

# # USD
# plt.subplot(3, 1, 1)
# plt.plot(data['Date'], data['USD (AM)'], label='USD (AM)', color='blue', alpha=0.7)
# plt.plot(data['Date'], data['USD (PM)'], label='USD (PM)', color='red', alpha=0.7)
# plt.title('Gold Price Trend in USD Over Time')
# plt.legend()
# plt.ylabel('Price in USD')
# plt.xlabel('Date')

# # GBP
# plt.subplot(3, 1, 2)
# plt.plot(data['Date'], data['GBP (AM)'], label='GBP (AM)', color='blue', alpha=0.7)
# plt.plot(data['Date'], data['GBP (PM)'], label='GBP (PM)', color='red', alpha=0.7)
# plt.title('Gold Price Trend in GBP Over Time')
# plt.legend()
# plt.ylabel('Price in GBP')
# plt.xlabel('Date')

# # EURO
# plt.subplot(3, 1, 3)
# plt.plot(data['Date'], data['EURO (AM)'], label='EURO (AM)', color='blue', alpha=0.7)
# plt.plot(data['Date'], data['EURO (PM)'], label='EURO (PM)', color='red', alpha=0.7)
# plt.title('Gold Price Trend in EURO Over Time')
# plt.legend()
# plt.ylabel('Price in EURO')
# plt.xlabel('Date')

# # Adjust layout for better visualization
# plt.tight_layout()
# plt.show()
# df = df[-1000:]
# df['Date Time'] = pd.to_datetime(df['Date Time'])
# df.set_index('Date Time', inplace=True)
# from statsmodels.tsa.seasonal import seasonal_decompose

# # Decompose the time series
# result = seasonal_decompose(df['T (degC)'], model='additive', period=365)
# trend = result.trend
# seasonal = result.seasonal
# residual = result.resid

# # Create a 4-subplot figure
# fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# # Plot the original time series
# axs[0].plot(df.index, df['T (degC)'], label='Original', color='black')
# axs[0].set_title('Original Time Series')

# # Plot Trend
# axs[1].plot(trend, label='Trend', color='blue')
# axs[1].set_title('Trend')

# # Plot Seasonal
# axs[2].plot(seasonal, label='Seasonal', color='green')
# axs[2].set_title('Seasonal')

# # Plot Residual
# axs[3].plot(residual, label='Residual', color='red')
# axs[3].set_title('Residual')

# # Add a common x-axis label
# plt.xlabel('Date')

# # Adjust subplot layout
# plt.tight_layout()

# # Show the plot
# plt.show()


# # Assuming your time series data is in 'y' (e.g., df['T (degC)'])
# y = df['T (degC)']

# # Perform FFT
# fft_result = np.fft.fft(y)
# frequencies = np.fft.fftfreq(len(fft_result))

# # Plot the power spectrum to identify significant frequencies
# plt.figure(figsize=(12, 6))
# plt.plot(frequencies, np.abs(fft_result))
# plt.title("FFT Power Spectrum")
# plt.xlabel("Frequency")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.show()

# df = get_data(csv_path="data/gold/LBMA-GOLD.csv")
# date = pd.to_datetime(df['Date'])
# df, datetime = prepare_data(df, label='USD (AM)', date='Date')
# df = add_time(df, datetime)
# print(df)
# train_df, val_df, test_df, num_features = split_data(df)
# train_df, val_df, test_df = normalize_data(train_df, val_df, test_df)
# wide_window = WindowGenerator(
#     input_width=24,
#     label_width=24,
#     train_df=train_df,
#     test_df=test_df,
#     val_df=val_df,
#     shift=1,
#     label_columns=['USD (AM)']
# )

# model = load_model("models/lstm.keras")
# print(wide_window.test)
# predictions = model.predict(wide_window.test)
# print(predictions)
# train_df, val_df, test_df, num_features = split_data(df)
# train_df, val_df, test_df = normalize_data(train_df, val_df, test_df)
# model = load_model("models/cnn.keras")
# model.predict(test_df)
# print(len(df.groupby(pd.Grouper(freq='5Y')).mean()))

# plt.plot(df['Month sin'][:150], label='Month sin')
# plt.plot(df['Month cos'][:150], label='Month cos')
# plt.plot(df['Year sin'][:1000], label='Year sin')
# plt.plot(df['Year cos'][:1000], label='Year cos')
# plt.legend()
# plt.show()

# plt.hist(df['USD (AM)'].iloc[:365])
# plt.xlabel('Price')
# plt.ylabel('Count')
# plt.title('Price Distribution (USD (AM))')
# plt.show()
# df['EURO (AM)'] = df['EURO (AM)'].fillna(method='ffill')
# plt.plot(df['USD (AM)'][:180], label='USD (AM)')
# plt.plot(df['USD (PM)'][:180], label='USD (PM)')
# # plt.plot(df['GBP (AM)'], label='GBP (AM)')
# # plt.plot(df['EURO (AM)'], label='EURO (AM)')
# plt.legend()
# plt.show()

#----------------------------------------------------------------
#                           DRAFT

#--------------------------------Dense---------------------------
# dense = tf.keras.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=32, activation='relu'),
#     tf.keras.layers.Dense(units=32, activation='relu'),
#     tf.keras.layers.Dense(units=1),
#     tf.keras.layers.Reshape([1, -1]),
# ])
# model = compile_and_fit(dense, conv_window)
# model.save('models/dense.keras')

# val_performance['Dense'] = dense.evaluate(conv_window.val)
# performance['Dense'] = dense.evaluate(conv_window.test, verbose=0)
# wide_conv_window.plot(dense, plot_col='USD (AM)')

# wg_single_predictor = WindowGenerator(input_width=1, 
#                      label_width=1, 
#                      shift=1,
#                      train_df=train_df,
#                      val_df=val_df,
#                      test_df=test_df,
#                      label_columns=['USD (AM)'])

# wg_plotter = WindowGenerator(input_width=24, 
#                      label_width=24, 
#                      shift=1,
#                      train_df=train_df,
#                      val_df=val_df,
#                      test_df=test_df,
#                      label_columns=['USD (AM)'])

# wg_multi = WindowGenerator(input_width=24, 
#                      label_width=24, 
#                      shift=24,
#                      train_df=train_df,
#                      val_df=val_df,
#                      test_df=test_df,
#                      label_columns=['USD (AM)'])

# val_performance = {}
# performance = {}

# linear = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=1)
# ])

# history = compile_and_fit(linear, wg_single_predictor)

# val_performance['Linear'] = linear.evaluate(wg_single_predictor.val)
# performance['Linear'] = linear.evaluate(wg_single_predictor.test, verbose=0)

# # wg_plotter.plot(model=linear, plot_col='USD (AM)')
# #----------------------------------------------------------------
# dense = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=64, activation='relu'),
#     tf.keras.layers.Dense(units=64, activation='relu'),
#     tf.keras.layers.Dense(units=1)
# ])

# history = compile_and_fit(dense, wg_single_predictor)

# val_performance['Dense'] = dense.evaluate(wg_single_predictor.val)
# performance['Dense'] = dense.evaluate(wg_single_predictor.test, verbose=0)
# # wg_plotter.plot(model=dense, plot_col='USD (AM)')

# #----------------------------------------------------------------
# CONV_WIDTH = 3
# conv_window = WindowGenerator(
#     input_width=CONV_WIDTH,
#     label_width=1,
#     shift=1,
#     train_df=train_df,
#     val_df=val_df,
#     test_df=test_df,
#     label_columns=['USD (AM)'])

# conv_model = tf.keras.Sequential([
#     tf.keras.layers.Conv1D(filters=32,
#                            kernel_size=(CONV_WIDTH,),
#                            activation='relu'),
#     tf.keras.layers.Dense(units=32, activation='relu'),
#     tf.keras.layers.Dense(units=1),
# ])

# history = compile_and_fit(conv_model, conv_window)

# IPython.display.clear_output()
# val_performance['Conv'] = conv_model.evaluate(conv_window.val)
# performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

# LABEL_WIDTH = 24
# INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
# wide_conv_window = WindowGenerator(
#     input_width=INPUT_WIDTH,
#     label_width=LABEL_WIDTH,
#     shift=1,
#     train_df=train_df,
#     val_df=val_df,
#     test_df=test_df,
#     label_columns=['USD (AM)'])

# # wide_conv_window.plot(conv_model, plot_col='USD (AM)')
# #----------------------------------------------------------------
# x = np.arange(len(performance))
# width = 0.3
# metric_name = 'mean_absolute_error'
# metric_index = linear.metrics_names.index('mean_absolute_error')
# val_mae = [v[metric_index] for v in val_performance.values()]
# test_mae = [v[metric_index] for v in performance.values()]

# plt.ylabel('mean_absolute_error [USD (AM), normalized]')
# plt.bar(x - 0.17, val_mae, width, label='Validation')
# plt.bar(x + 0.17, test_mae, width, label='Test')
# plt.xticks(ticks=x, labels=performance.keys(),
#            rotation=45)
# _ = plt.legend()
# plt.show()
# #----------------------------------------------------------------
# metric_name = 'mean_squared_error'
# metric_index = linear.metrics_names.index('mean_squared_error')
# val_mse = [v[metric_index] for v in val_performance.values()]
# test_mse = [v[metric_index] for v in performance.values()]

# plt.ylabel('mean_squared_error [USD (AM), normalized]')
# plt.bar(x - 0.17, val_mse, width, label='Validation')
# plt.bar(x + 0.17, test_mse, width, label='Test')
# plt.xticks(ticks=x, labels=performance.keys(),
#            rotation=45)
# _ = plt.legend()
# plt.show()

# import numpy as np
# import xgboost as xgb

# data = np.array([1,2,3,4,5,6,7,8,9,10])
# window_size = 3
# stride = 1

# windowed_data = []
# targets = []

# for i in range(len(data) - window_size):
#     windowed_data.append(data[i:i+window_size])
#     targets.append(data[i+window_size])
    
# windowed_data = np.array(windowed_data)
# targets = np.array(targets)

# train_size = int(0.8 * len(windowed_data))
# train_data, test_data = windowed_data[:train_size], windowed_data[train_size:]
# train_targets, test_targets = targets[:train_size], targets[train_size:]

# dtrain = xgb.DMatrix(train_data, label=train_targets)

# params = {
#     'max_depth': 3,
#     'objective': 'reg:squarederror',
#     'eta': 0.1,
#     'eval_metric': 'mae'
# }

# model = xgb.train(params, dtrain)

# dtest = xgb.DMatrix(test_data)
# predicitons = model.predict(dtest)
# rmse = np.sqrt(np.mean((predicitons - test_targets)**2))
# print('root mean squared error: ', rmse)
    