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
df, datetime = prepare_data(df, label='USD (AM)', date='Date', features=['USD (PM)'])
df = add_time(df, datetime)
# plt.plot(df['Month sin'][:150], label='Month sin')
# plt.plot(df['Month cos'][:150], label='Month cos')
# plt.plot(df['Year sin'][:1000], label='Year sin')
# plt.plot(df['Year cos'][:1000], label='Year cos')
# plt.legend()
# plt.show()

# plt.hist(df['USD (AM)'], bins=1500)
# plt.xlabel('Price')
# plt.ylabel('Count')
# plt.title('Price Distribution (USD (AM))')
# plt.show()
# df['EURO (AM)'] = df['EURO (AM)'].fillna(method='ffill')
plt.plot(df['USD (AM)'][:180], label='USD (AM)')
plt.plot(df['USD (PM)'][:180], label='USD (PM)')
# # plt.plot(df['GBP (AM)'], label='GBP (AM)')
# # plt.plot(df['EURO (AM)'], label='EURO (AM)')
plt.legend()
plt.show()