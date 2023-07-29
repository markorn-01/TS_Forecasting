import streamlit as st
import pandas as pd
from data_process import *

@st.cache_data
def collect_data():
    df = get_data(csv_path="data/gold/LBMA-GOLD.csv")
    df, datetime = prepare_data(df, label='USD (AM)', date='Date') 
    df = add_time(df, datetime)
    return df

df = collect_data()

def show_data_page():
    st.title("Description of Gold Price Dataset")
    st.write("This dataset contains the daily gold price from 1968 to June, 2023. The data is collected from [LBMA](https://www.lbma.org.uk/).")
    st.write("""
        ### Mean of the gold price (USD) over 5-year periods
             """)
    five_year_mean = df.groupby(pd.Grouper(freq='5Y')).mean()
    st.line_chart(five_year_mean['USD (AM)'])