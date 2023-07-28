import streamlit as st
from data_page import show_data_page
from predict_page import show_predict_page

st.title("Model Deployment")
option = st.sidebar.selectbox("Pick an option", ("Inspect data", "Make a prediction"))

if option == "Inspect data":
    show_data_page()
else:
    show_predict_page()