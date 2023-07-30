import streamlit as st
import pandas as pd
from data_process import create_time_features

def set_up_page():
    st.title("Prediction of Gold Price by Deep Learning Models")
    st.write("This page is under construction.")
    
def get_options():
    model = st.sidebar.selectbox("Select a model", ["LSTM", "CNN"])
    start_date = pd.to_datetime('2023-07-01', format='%Y-%m-%d')
    end_date = start_date + pd.DateOffset(years= 5)
    predict_date = st.date_input('Predict date', start_date, \
                                 min_value=start_date, \
                                 max_value=end_date)
    return model, predict_date

def make_prediction(model, predict_date):
    pass

def show_predict_page():
    set_up_page()
    model, predict_date = get_options()
    make_prediction(model, predict_date)
    
    