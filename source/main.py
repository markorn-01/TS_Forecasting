import datetime
import IPython
import IPython.display
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from DataProcessor import *
from WindowGenerator import *
from ModelGenerator import *

df = get_data(csv_path="data/gold/LBMA-GOLD.csv")
df, datetime = prepare_data(df, label='USD (AM)', date='Date') 
df = add_time(df, datetime)
train_df, val_df, test_df, num_features = split_data(df)
train_df, val_df, test_df = normalize_data(train_df, val_df, test_df)
print(train_df)
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