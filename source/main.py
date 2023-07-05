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

df, datetime = get_data("data/gold/LBMA-GOLD.csv", 'USD (AM)')
df = add_time(df, datetime)
train_df, val_df, test_df, num_features = split_data(df)
train_df, val_df, test_df = normalize_data(train_df, val_df, test_df)
w2 = WindowGenerator(input_width=6, 
                     label_width=1, 
                     shift=1,
                     train_df=train_df,
                     val_df=val_df,
                     test_df=test_df,
                     label_columns=['USD (AM)'])

# Stack three slices, the length of the total window.
# example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
#                            np.array(train_df[100:100+w2.total_window_size]),
#                            np.array(train_df[200:200+w2.total_window_size])])

# example_inputs, example_labels = w2.split_window(example_window)

# print('All shapes are: (batch, time, features)')
# print(f'Window shape: {example_window.shape}')
# print(f'Inputs shape: {example_inputs.shape}')
# print(f'Labels shape: {example_labels.shape}')

print(df.head())
