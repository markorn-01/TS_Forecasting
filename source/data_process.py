import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

def get_data(csv_path):
    #this path is used for testing only. Otherwise, use the path in the main function.
    csv_path = "data/jena_climate/jena_climate_2009_2016.csv"
    df = pd.read_csv(csv_path) 
    df = df[5::6]
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    return df, date_time    

def clean_data(df):
    #Convert error values to 0.0
    df[df['wv (m/s)']==-9999.0] = 0.0
    df[df['max. wv (m/s)']==-9999.0] = 0.0
    #extract wind velocity
    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')
    # Convert to radians.
    wd_rad = df.pop('wd (deg)')*np.pi / 180
    # Calculate the wind x and y components.
    df['Wx'] = wv*np.cos(wd_rad)
    df['Wy'] = wv*np.sin(wd_rad)
    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv*np.cos(wd_rad)
    df['max Wy'] = max_wv*np.sin(wd_rad)
    return df

