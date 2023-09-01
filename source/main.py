import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from utils.data_process import *
from utils.window_generate import *
from utils.model_generate import *
from models.Transformer import Transformer
from models.CNN import CNN
from models.LSTM import LSTMModel as LSTM
# data preprocessing
df = get_data(csv_path="data/gold/LBMA-GOLD.csv")
df, datetime = prepare_data(df, label='USD (AM)', date='Date') 
df = add_time(df, datetime)
train_df, val_df, test_df, num_features = split_data(df)
train_df, val_df, test_df = normalize_data(train_df, val_df, test_df)
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
    label_columns=['USD (AM)']
)

ex_window = WindowGenerator(
    input_width=5,
    label_width=1,
    train_df=train_df,
    test_df=test_df,
    val_df=val_df,
    shift=1,
    label_columns=['USD (AM)']
)

label_width = 24
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

val_performance = {}
performance = {}

#----------------------------------------------------------------
# model generating
transformer = Transformer(window=wide_window, num_heads=5, num_transformer_layers=2)
model = compile_and_fit(transformer, wide_window)
model.save('train/models/transformer', save_format='tf')
performance['Transformer'] = transformer.evaluate(wide_window.test, verbose=0)
wide_window.plot(transformer, plot_col='USD (AM)', max_subplots=1)
# print(model.metrics_names)
#--------------------------------CNN----------------------------

# cnn = CNN(conv_width=CONV_WIDTH)
# model = compile_and_fit(cnn, conv_window)
# model.save('train/models/cnn', save_format='tf')

# val_performance['Conv'] = cnn.evaluate(conv_window.val)
# performance['Conv'] = cnn.evaluate(conv_window.test, verbose=0)
# wide_conv_window.plot(cnn, plot_col='USD (AM)')

# #--------------------------------LSTM---------------------------
# lstm = LSTM(lstm_units=32)
# model = compile_and_fit(lstm, wide_window)
# model.save('train/models/lstm', save_format='tf')

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