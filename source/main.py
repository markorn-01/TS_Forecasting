import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
from sklearn import preprocessing as pre
from utils.data_process import *
from utils.window_generate import *
from utils.model_generate import *
from models.Transformer import Transformer
from models.CNN import CNN
from models.LSTM import LSTMModel as LSTM, ResidualLSTMModel as ResidualLSTM, ResidualWrapper
from models.LSTMCNN import LSTMCNN
# data preprocessing
df = get_data(csv_path="data/gold/LBMA-GOLD.csv")
df = fill_missing_gold(df)
label = 'USD (AM)'
date = 'Date'
# label = 'T (degC)'
# date = 'Date Time'
df, datetime = prepare_data(df, label=label, features=[col for col in df.columns if col != label and col != date] ,date=date)
# df = get_data(csv_path="data/jena_climate/jena_climate_2009_2016.csv")
# df = transfer_features_jena(df)
# df, datetime = prepare_data(df, label=label, features=[col for col in df.columns if col != label and col != date] ,date=date, format='mixed')
df = add_time(df, datetime)

train_df, val_df, test_df, num_features = split_data(df)
train_df, val_df, test_df, scaler = normalize_data(train_df, val_df, test_df)
#----------------------------------------------------------------
# window generating

CONV_WIDTH = 3
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    train_df=train_df,
    test_df=test_df,
    val_df=val_df,
    shift=1,
    label_columns=['USD (AM)'])

wide_window = WindowGenerator(
    input_width=12,
    label_width=12,
    train_df=train_df,
    test_df=test_df,
    val_df=val_df,
    shift=1,
    label_columns=[label]
)

# ex_window = WindowGenerator(
#     input_width=5,
#     label_width=1,
#     train_df=train_df,
#     test_df=test_df,
#     val_df=val_df,
#     shift=1,
#     label_columns=['USD (AM)']
# )

label_width = 30
input_width = label_width + CONV_WIDTH - 1

wide_conv_window = WindowGenerator(
    input_width=input_width,
    label_width=label_width,
    train_df=train_df,
    test_df=test_df,
    val_df=val_df,
    shift=1,
    label_columns=['USD (AM)']
)
# print(wide_conv_window.example[0].shape)
val_performance = {}
performance = {}

# print(len(wide_window.train_df))
# print(len(wide_window.val_df))
# print(len(wide_window.test_df))
# #----------------------------------------------------------------
# # model generating
# transformer = Transformer(num_features=wide_window.train_df.shape[-1], num_heads=5, num_transformer_layers=2)
# lstmcnn = LSTM(lstm_units=32)
# --------------------------------cnn---------------------------
cnn = CNN(conv_width=1)
model = compile_and_fit(cnn, wide_window)
tf.saved_model.save(model, export_dir='train/models/cnn/')
# val_performance['Conv'] = cnn.evaluate(wide_window.val)
# performance['Conv'] = cnn.evaluate(wide_window.test, verbose=0)
# #--------------------------------lstm---------------------------
lstm = LSTM(lstm_units=32)
model = compile_and_fit(lstm, wide_window)
tf.saved_model.save(model, export_dir='train/models/lstm/')
# val_performance['LSTM'] = lstm.evaluate(wide_window.val)
# performance['LSTM'] = lstm.evaluate(wide_window.test, verbose=0)
# #--------------------------------lstm residual---------------------------
# lstmresidual = ResidualWrapper(ResidualLSTM(lstm_units=32, num_features=wide_window.train_df.shape[-1]))
# model = compile_and_fit(lstmresidual, wide_window)
# val_performance['lstmresidual'] = lstmresidual.evaluate(wide_window.val)
# performance['lstmresidual'] = lstmresidual.evaluate(wide_window.test, verbose=0)
# #--------------------------------lstmcnn---------------------------
# lstmcnn = LSTMCNN(lstm_units=32, conv_width=1)
# model = compile_and_fit(lstmcnn, wide_window)
# val_performance['LSTMCNN'] = lstmcnn.evaluate(wide_window.val)
# performance['LSTMCNN'] = lstmcnn.evaluate(wide_window.test, verbose=0)
# #--------------------------------transformer---------------------------
transformer = Transformer(num_features=wide_window.train_df.shape[-1], num_heads=8, num_transformer_layers=2)
model = compile_and_fit(transformer, wide_window)
tf.saved_model.save(model, export_dir='train/models/transformer/')
# val_performance['Transformer'] = transformer.evaluate(wide_window.val)
# performance['Transformer'] = transformer.evaluate(wide_window.test, verbose=0)
# #--------------------------------show result---------------------------
# # print({key: [round(val, 6) for val in values] for key, values in val_performance.items()})
# # print({key: [round(val, 6) for val in values] for key, values in performance.items()})
# file_path1 = 'train/train_results/jena/val_performance.csv'
# file_path2 = 'train/train_results/jena/performance.csv'

# # Write val_performance to a CSV file
# with open(file_path1, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Model', 'Loss', 'MAE', 'RMSE', 'MAPE'])  # Header row
#     for key, values in val_performance.items():
#         writer.writerow([key] + [round(val, 6) for val in values])

# # Write performance to a CSV file
# with open(file_path2, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Model', 'Loss', 'MAE', 'RMSE', 'MAPE'])  # Header row
#     for key, values in performance.items():
#         writer.writerow([key] + [round(val, 6) for val in values])

# print('Data written to CSV files:', file_path1, file_path2)

# model.save_weights('train/transformer.h5')
# performance['Transformer'] = transformer.evaluate(wide_window.test, verbose=0)
# wide_window.plot(lstmcnn, plot_col='USD (AM)', max_subplots=2)
# print(model.metrics_names)
#--------------------------------CNN----------------------------
# wide_window.plot(cnn, plot_col='USD (AM)')

# #--------------------------------LSTM---------------------------
# loaded = tf.saved_model.load('train/models/lstm/')

# example_inputs, example_labels =  wide_window.example
# predictions = loaded(example_inputs, training=False)
# values = predictions[0, :, wide_window.column_indices['USD (AM)']].numpy()
# for val in values:
#     print(val)
# wide_window.plot(loaded, plot_col='USD (AM)', max_subplots=3)
# print(lstm.load('train/models/lstm.pkl'))
# val_performance['LSTM'] = lstm.evaluate(wide_window.val)
# performance['LSTM'] = lstm.evaluate(wide_window.test, verbose=0)
# print(performance)
# wide_window.plot(lstm, plot_col='USD (AM)')

# # show result
# show_result(model=lstm, 
#             val_performance=val_performance, 
#             performance=performance, 
#             metric_name='mean_squared_error',
#             y_label='Loss [USD (AM), normalized]')

# show_result(model=lstm, 
#             val_performance=val_performance, 
#             performance=performance, 
#             metric_name='mean_absolute_error',
#             y_label='MAE [USD (AM), normalized]')