import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from data_process import *
from WindowGenerator import *
from ModelGenerator import *

df = get_data(csv_path="data/gold/LBMA-GOLD.csv")
df, datetime = prepare_data(df, label='USD (AM)', date='Date') 
df = add_time(df, datetime)

train_df, val_df, test_df, num_features = split_data(df)
# train_df, val_df, test_df = normalize_data(train_df, val_df, test_df)

wg_single_predictor = WindowGenerator(input_width=1, 
                     label_width=1, 
                     shift=1,
                     train_df=train_df,
                     val_df=val_df,
                     test_df=test_df,
                     label_columns=['USD (AM)'])

wg_plotter = WindowGenerator(input_width=24, 
                     label_width=24, 
                     shift=1,
                     train_df=train_df,
                     val_df=val_df,
                     test_df=test_df,
                     label_columns=['USD (AM)'])


val_performance = {}
performance = {}



dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(dense, wg_single_predictor)

val_performance['Dense'] = dense.evaluate(wg_single_predictor.val)
performance['Dense'] = dense.evaluate(wg_single_predictor.test, verbose=0)
# print(wg_single_predictor)
# wg_plotter.plot(model=linear, plot_col='USD (AM)')
