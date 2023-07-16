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

#data preprocessing
df = get_data(csv_path="data/gold/LBMA-GOLD.csv")
df, datetime = prepare_data(df, label='USD (AM)', date='Date') 
df = add_time(df, datetime)
train_df, val_df, test_df, num_features = split_data(df)
train_df, val_df, test_df = normalize_data(train_df, val_df, test_df)

#----------------------------------------------------------------
#window generating

wide_window = WindowGenerator(
    input_width=24,
    label_width=24,
    train_df=train_df,
    test_df=test_df,
    val_df=val_df,
    shift=1,
    label_columns=['USD (AM)']
)

# conv_width = 3
# label_width = 24
# input_width = label_width + conv_width - 1

# wide_conv_window = WindowGenerator(
#     input_width=input_width,
#     label_width=label_width,
#     train_df=train_df,
#     test_df=test_df,
#     val_df=val_df,
#     shift=1,
#     label_columns=['USD (AM)']
# )

val_peroformance = {}
performance = {}
#----------------------------------------------------------------
#model generating
lstm = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(units=1)
])

model = compile_and_fit(lstm, wide_window)
model.save('models/lstm.keras')

val_peroformance['LSTM'] = lstm.evaluate(wide_window.val)
performance['LSTM'] = lstm.evaluate(wide_window.test, verbose=0)

wide_window.plot(lstm, plot_col='USD (AM)')

