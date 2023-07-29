import streamlit as st
import pandas as pd
from data_process import *

@st.cache_data
def collect_data():
    df = get_data(csv_path="data/gold/LBMA-GOLD.csv")
    label_col = 'USD (AM)'
    feature_col = 'USD (PM)'
    df, datetime = prepare_data(df, label=label_col, features=[feature_col], date='Date') 
    df = add_time(df, datetime)
    return df, label_col, feature_col

df, label, features = collect_data()

def show_by_date():
    first_date = pd.to_datetime('1968-01-02', format='%Y-%m-%d')
    last_date = pd.to_datetime('2023-06-30', format='%Y-%m-%d')
    start_date = st.date_input('Start date', first_date, min_value=first_date, max_value=last_date)
    end_date = st.date_input('End date', last_date, min_value=first_date, max_value=last_date)
    if (not (first_date <= start_date <= last_date)) or (not (first_date <= end_date <= last_date)):
        st.error('Error: Date must fall between 1968-01-02 and 2023-06-30.')
    elif start_date <= end_date:
        st.write(df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)])
    else:
        st.error('Error: End date must fall after start date.')

def show_statistics():
    st.write("""
        ### Mean of the gold price (USD) every 5-year period from 1968 to 2023
             """)
    five_year_mean = df.groupby(pd.Grouper(freq='5Y')).mean()
    st.line_chart(five_year_mean[[label, features]])
    st.write("""
        ### Distribution of the gold price (USD) from 1968 to 2023
        """)
    st.bar_chart(df[[label, features]])
    
def show_data_page():
    st.title("Description of Gold Price Dataset")
    st.write("This dataset contains the daily gold price from January, 1968 to June, 2023. It is collected from [LBMA](https://www.lbma.org.uk/).")
    show_by_date()
    show_statistics()
    