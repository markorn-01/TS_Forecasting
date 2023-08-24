import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from data_process import *
from window_generate import *
from model_generate import *
from models.transformers import Transformer
from models.SOTA import SOTA
from models.ODE import ODE
# data preprocessing
df = get_data(csv_path="data/gold/LBMA-GOLD.csv")
df, datetime = prepare_data(df, label='USD (AM)', date='Date') 
df = add_time(df, datetime)
train_df, val_df, test_df, num_features = split_data(df)
train_df, val_df, test_df = normalize_data(train_df, val_df, test_df)
print(train_df.head())
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
    input_width=24,
    label_width=24,
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

transformer = Transformer(window=ex_window)
print(transformer)
model = compile_and_fit(transformer.model, ex_window)
eval_performance = transformer.model.evaluate(ex_window.test, verbose=0)
#--------------------------------CNN----------------------------
# cnn = tf.keras.Sequential([
#     tf.keras.layers.Conv1D(filters=32,
#                            kernel_size=(CONV_WIDTH,),
#                            activation='relu'),
    # tf.keras.layers.Dense(units=32, activation='relu'),
    # tf.keras.layers.Dense(units=1),
# ])
# model = compile_and_fit(cnn, conv_window)
# model.save('models/cnn.keras')

# val_performance['Conv'] = cnn.evaluate(conv_window.val)
# performance['Conv'] = cnn.evaluate(conv_window.test, verbose=0)
# wide_conv_window.plot(cnn, plot_col='USD (AM)')

# #--------------------------------LSTM---------------------------
# lstm = tf.keras.models.Sequential([
#     tf.keras.layers.LSTM(32, return_sequences=True),
#     tf.keras.layers.Dense(units=1)
# ])

# model = compile_and_fit(lstm, wide_window)
# model.save('models/lstm.keras')

# val_performance['LSTM'] = lstm.evaluate(wide_window.val)
# performance['LSTM'] = lstm.evaluate(wide_window.test, verbose=0)
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