import streamlit as st
import pandas as pd
import datetime
from data_process import *

@st.cache_data
def collect_data():
    df = get_data(csv_path="data/gold/LBMA-GOLD.csv")
    df, datetime = prepare_data(df, label='USD (AM)', date='Date') 
    df = add_time(df, datetime)
    return df

df = collect_data()

def show_date():
    today = pd.to_datetime('1968-01-02', format='%Y-%m-%d')
    tomorrow = pd.to_datetime('2023-06-30', format='%Y-%m-%d')
    start_date = st.date_input('Start date', today, min_value=today, max_value=tomorrow)
    end_date = st.date_input('End date', tomorrow, min_value=today, max_value=tomorrow)
    if start_date <= end_date:
        st.write(df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)])
    else:
        st.error('Error: End date must fall after start date.')

def show_data_page():
    st.title("Description of Gold Price Dataset")
    st.write("This dataset contains the daily gold price from January, 1968 to June, 2023. The data is collected from [LBMA](https://www.lbma.org.uk/).")
    show_date()
    st.write("""
        ### Mean of the gold price (USD) every 5-year period from 1968 to 2023
             """)
    five_year_mean = df.groupby(pd.Grouper(freq='5Y')).mean()
    st.line_chart(five_year_mean['USD (AM)'])