import datetime
import numpy as np
import pandas as pd
import seaborn as sns
# import tensorflow as tf
from tensorflow.keras.models import load_model
from data_process import *
from window_generate import *
from model_generate import *


df = get_data(csv_path="data/gold/LBMA-GOLD.csv")
date = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df, datetime = prepare_data(df, label='USD (AM)', date='Date')
df = add_time(df, datetime)
print(df)
train_df, val_df, test_df, num_features = split_data(df)
train_df, val_df, test_df = normalize_data(train_df, val_df, test_df)
wide_window = WindowGenerator(
    input_width=24,
    label_width=24,
    train_df=train_df,
    test_df=test_df,
    val_df=val_df,
    shift=1,
    label_columns=['USD (AM)']
)

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
    