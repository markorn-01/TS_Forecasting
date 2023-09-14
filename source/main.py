import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.data_process import *
from utils.window_generate import *
from utils.model_generate import *
from models.Transformer import Transformer
from models.CNN import CNN
from models.LSTM import LSTMModel as LSTM, ResidualLSTMModel as ResidualLSTM, ResidualWrapper
# data preprocessing
df = get_data(csv_path="data/gold/LBMA-GOLD.csv")
df, datetime = prepare_data(df, label='USD (AM)', date='Date')
# df = get_data(csv_path="data/jena_climate/jena_climate_2009_2016.csv")
# df, datetime = prepare_data(df, label='T (degC)', date='Date Time', features=['p (mbar)', 'T (degC)', 'rho (g/m**3)'])
df = add_time(df, datetime)
train_df, val_df, test_df, num_features = split_data(df)
train_df, val_df, test_df = normalize_data(train_df, val_df, test_df)
#----------------------------------------------------------------
# window generating

# CONV_WIDTH = 3
# conv_window = WindowGenerator(
#     input_width=CONV_WIDTH,
#     label_width=1,
#     train_df=train_df,
#     test_df=test_df,
#     val_df=val_df,
#     shift=1,
#     label_columns=['USD (AM)'])

wide_window = WindowGenerator(
    input_width=12,
    label_width=12,
    train_df=train_df,
    test_df=test_df,
    val_df=val_df,
    shift=1,
    label_columns=['USD (AM)']
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

# label_width = 24
# input_width = label_width + CONV_WIDTH - 1

# wide_conv_window = WindowGenerator(
#     input_width=input_width,
#     label_width=label_width,
#     train_df=train_df,
#     test_df=test_df,
#     val_df=val_df,
#     shift=1,
#     label_columns=['USD (AM)']
# )

# val_performance = {}
# performance = {}

# #----------------------------------------------------------------
# # model generating
# transformer = Transformer(window=wide_window, num_heads=5, num_transformer_layers=2)
# model = compile_and_fit(transformer, wide_window)
# model.save_weights('train/transformer.h5')
# performance['Transformer'] = transformer.evaluate(wide_window.test, verbose=0)
# wide_window.plot(transformer, plot_col='T (degC)', max_subplots=3)
# print(model.metrics_names)
#--------------------------------CNN----------------------------

# cnn = CNN(conv_width=CONV_WIDTH)
# model = compile_and_fit(cnn, conv_window)
# model.save('train/models/cnn', save_format='tf')

# val_performance['Conv'] = cnn.evaluate(conv_window.val)
# performance['Conv'] = cnn.evaluate(conv_window.test, verbose=0)
# wide_conv_window.plot(cnn, plot_col='USD (AM)')

# #--------------------------------LSTM---------------------------
lstm = ResidualWrapper(ResidualLSTM(lstm_units=32, num_features=wide_window.train_df.shape[-1]))
# model = compile_and_fit(lstm, wide_window, checkpoint_filepath='train/models/lstm')
# model.save('lstm')
print(lstm.load('train/models/lstm.pkl'))
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