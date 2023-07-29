import streamlit as st
st.set_page_config(page_title="Gold Price Website", page_icon=":moneybag:", layout="wide")

from data_page import show_data_page
from predict_page import show_predict_page


option = st.sidebar.selectbox("Pick an option", ("Inspect data", "Make a prediction"))

if option == "Inspect data":
    show_data_page()
else:
    show_predict_page()