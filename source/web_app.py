import streamlit as st
import pandas as pd
from utils.data_process import *
from utils.window_generate import *
st.set_page_config(page_title="Gold Price Website", page_icon=":moneybag:", layout="centered")

def get_options():
    option = st.sidebar.selectbox("Pick a dataset", ("LBMA-Gold", "Jena Climate"), index=0)
    date, df = get_dataset(option)
    predicted_column = st.sidebar.selectbox("Select Predicted Column", df.columns, index=1)
    feature_options = [col for col in df.columns if col != predicted_column]
    feature_columns = st.sidebar.multiselect("Select Feature Columns", feature_options, default=[])
    model = st.sidebar.selectbox("Select a model", ["CNN", "LSTM", "Transformer"])
    time_interval = st.sidebar.number_input("Enter the number of days", min_value=1)
    return df, date, predicted_column, feature_columns, model, time_interval

@st.cache_data
def get_dataset(option):
    if option == "LBMA-Gold":
        csv_path = "data/gold/LBMA-GOLD.csv"
        date = "Date"
    else:
        csv_path = "data/jena_climate/jena_climate_2009_2016.csv"
        date = "Date Time"
    df = get_data(csv_path=csv_path)
    return date,df
    
def show_script():
    st.title("Description of Dataset")
    st.write("This dataset contains the daily gold price from January, 1968 to June, 2023. It is collected from [LBMA](https://www.lbma.org.uk/).")

def show_by_date(df, date):
    first_date = pd.to_datetime('1968-01-02', format='%Y-%m-%d')
    last_date = pd.to_datetime('2023-06-30', format='%Y-%m-%d')
    # Convert start_date and end_date to datetime format
    start_date = pd.to_datetime(st.date_input('Start date', first_date, min_value=first_date, max_value=last_date))
    end_date = pd.to_datetime(st.date_input('End date', last_date, min_value=first_date, max_value=last_date))
    df.set_index(date, inplace=True)
    if (not (first_date <= start_date <= last_date)) or (not (first_date <= end_date <= last_date)):
        st.error('Error: Date must fall between 1968-01-02 and 2023-06-30.')
    elif start_date <= end_date:
        st.write(df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)])
    else:
        st.error('Error: End date must fall after start date.')

def show_statistics(df, label, features):
    st.write("""
        ### Mean of the gold price (USD) every 5-year period from 1968 to 2023
             """)
    five_year_mean = df.groupby(pd.Grouper(freq='5Y')).mean()
    chart_columns = tuple([label] + features)
    st.line_chart(five_year_mean[list(chart_columns)])
    st.write("""
        ### Distribution of the gold price (USD) from 1968 to 2023
        """)
    st.bar_chart(df[list(chart_columns)])

def show_page():
    df, date, predicted_column, feature_columns, model, time_interval = get_options()
    show_script()
    df, datetime = prepare_data(df, label=predicted_column, date=date, features=feature_columns)
    show_by_date(df, datetime)
    show_statistics(df, predicted_column, feature_columns)
    df = add_time(df, datetime)
    train_df, val_df, test_df, num_features = split_data(df)
    train_df, val_df, test_df = normalize_data(train_df, val_df, test_df)
    wide_window = WindowGenerator(
        input_width=12,
        label_width=12,
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        shift=1,
        label_columns=[predicted_column]
    )

show_page()