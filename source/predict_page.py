import streamlit as st
import pandas as pd

def show_predict_page():
    st.title("Prediction of Gold Price by Deep Learning Models")
    st.write("This page is under construction.")
    st.sidebar.selectbox("Select a model", ["LSTM", "CNN"])