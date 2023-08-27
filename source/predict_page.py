import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from utils.data_process import create_time_features

def set_up_page():
    st.title("Prediction of Gold Price by Deep Learning Models")
    st.write("This page is under construction.")
    
def get_options():
    model = st.sidebar.selectbox("Select a model", ["CNN", "LSTM", "Transformer"])
    start_date = pd.to_datetime('2023-07-01', format='%Y-%m-%d')
    end_date = start_date + pd.DateOffset(years= 5)
    predict_date = st.date_input('Predict date', \
                                 min_value=start_date, \
                                 max_value=end_date,\
                                value=pd.Timestamp.now())
    return model, predict_date

def make_prediction(model, predict_date):
    # model = load_model(f"models/{str(model).lower()}.keras")
    # timestamp_s = pd.to_datetime(predict_date, format='%Y-%m-%d')
    # st.write(type(timestamp_s))
    # day = 24*60*60
    # month = 30 * day
    # year = (365.2425)*day
    # month_sin = np.sin(timestamp_s * (2 * np.pi / month))
    # month_cos = np.cos(timestamp_s * (2 * np.pi / month))
    # year_sin = np.sin(timestamp_s * (2 * np.pi / year))
    # year_cos = np.cos(timestamp_s * (2 * np.pi / year))
    # month_sin, month_cos, year_sin, year_cos = create_time_features(predict_date_series)
    # df = get_data(csv_path="data/gold/LBMA-GOLD.csv")
    
    # st.write(model.predict())
    pass

def show_predict_page():
    set_up_page()
    model, predict_date = get_options()
    make_prediction(model, predict_date)
    
    