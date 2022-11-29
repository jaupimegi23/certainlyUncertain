import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from dashboard_utils import get_chart
import datetime


arimax_path = os.path.join(os.path.dirname(os.getcwd()), 'certainlyUncertain', 'models', 'model_save','arimax')

# Load data
df_arimax_ec15 = pd.read_csv(os.path.join(arimax_path, 'ec15', 'arimax_test.csv'))
df_arimax_ec15.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
df_arimax_ec15['date'] = pd.to_datetime(df_arimax_ec15['date'])
df_arimax_ec15['date_short'] = pd.to_datetime(df_arimax_ec15['date'].dt.strftime('%Y-%m-%d'))

# Remove all rows with 12 in the hour column
df_arimax_ec15 = df_arimax_ec15[df_arimax_ec15['date'].dt.hour != 12]
df_arimax_ec15.reset_index(drop=True, inplace=True)

# Get countries
cols = df_arimax_ec15.columns
cols_0 = [col for col in cols if '0' in col]
countries = [col.split('-')[1] for col in cols_0]
countries = [country[:3] for country in countries]
countries = np.unique(countries)

########## DASHBOARD PARAMS ##########

st.title("Weather Uncertainty Dashboard")
st.write("By Certainly Uncertains")

# Select date
date = st.sidebar.date_input('Select date', df_arimax_ec15['date'][15])

# Select prediction model
prediction_model = st.sidebar.selectbox('Select model', ['ARIMAX', 'VAR', 'Sparse Regression', 'XGBoost'])
# Select country
country = st.sidebar.selectbox('Select country', countries)
# Select weather variable
weather_variable = st.sidebar.selectbox('Select weather variable', ['WIND', 'TEMP'])



########## UNCERTAINTY METRICS ##########

# 24h forecast versus actual
st.subheader('Uncertainty Forecast')
st.write('Currently at: ', date)

#Filter df for 24h pred column and country in column name and date
pred_24 = df_arimax_ec15[[col for col in df_arimax_ec15.columns if '24' in col and country in col and weather_variable in col
                         and 'pred' in col]]
ens_48  = df_arimax_ec15[[col for col in df_arimax_ec15.columns if '48' in col and country in col and weather_variable in col
                         and 'ens' in col]]
model0_48 = df_arimax_ec15[[col for col in df_arimax_ec15.columns if '48' in col and country in col and weather_variable in col
                            and 'model_0' in col]]
date_inp = pd.to_datetime(date)
date_index = df_arimax_ec15[df_arimax_ec15['date_short'] == date_inp].index[0]
pred_24_metr = pred_24[df_arimax_ec15['date_short'] == date_inp]
ens_48_metr = ens_48[df_arimax_ec15['date_short'] == date_inp]
model0_48_metr = model0_48[df_arimax_ec15['date_short'] == date_inp]

m5, m6, m7 = st.columns((1,1,1))
m5.metric(label = 'Weather variable forecasted value', value = np.round(model0_48_metr.values[0][0],2))
m6.metric(label ='Prediction Model Uncertainty', value = np.round(pred_24_metr.values[0][0],2))
m7.metric(label ='Ensemble Uncertainty',value = np.round(ens_48_metr.values[0][0],2))

col1, col2, col3 = st.columns(3)
with col2:
   st.write('Prediction model uncertainty starts at:', date + pd.Timedelta(hours=24))
with col3:
   st.write('Ensemble uncertainty starts at:', date )
with col1:
   st.write('Forecasting for:', date + pd.Timedelta(hours=48))


########## UNCERTAINTY EVOLUTION ##########

st.subheader('Uncertainty Evolution')

ens_24  = df_arimax_ec15[[col for col in df_arimax_ec15.columns if '24' in col and country in col and weather_variable in col
                         and 'ens' in col and 'pred' not in col]]
model0_24 = df_arimax_ec15[[col for col in df_arimax_ec15.columns if '24' in col and country in col and weather_variable in col
                            and 'model_0' in col]]

window_back = 10
data_to_plot = pd.DataFrame(columns=['date', 'weather', 'lower', 'upper', 'lower_pred', 'upper_pred'])
data_to_plot['date'] = df_arimax_ec15['date'][date_index-window_back:date_index+3]
data_to_plot.reset_index(drop=True, inplace=True)
data_to_plot['weather'][0:window_back+2] = np.array(model0_24[date_index-10:date_index+2]).ravel()
data_to_plot['weather'][window_back+2] = model0_48.values[date_index+1][0]
data_to_plot['lower'][0:window_back+2] = data_to_plot['weather'][0:window_back+2].values * np.array((1 - ens_24[date_index-10:date_index+2])).ravel()
data_to_plot['lower'][window_back+2] = data_to_plot['weather'][window_back+2] * (1 - ens_48.values[date_index+1][0])
data_to_plot['upper'][0:window_back+2] = data_to_plot['weather'][0:window_back+2].values * np.array((1 + ens_24[date_index-10:date_index+2])).ravel()
data_to_plot['upper'][window_back+2] = data_to_plot['weather'][window_back+2] * (1 + ens_48.values[date_index+1][0])
data_to_plot['lower_pred'][0:window_back+2] = data_to_plot['weather'][0:window_back+2].values * np.array((1 - ens_24[date_index-10:date_index+2])).ravel()
data_to_plot['lower_pred'][window_back+2] = data_to_plot['weather'][window_back+2] * (1 - pred_24.values[date_index+1][0])
data_to_plot['upper_pred'][0:window_back+2] = data_to_plot['weather'][0:window_back+2].values * np.array((1 + ens_24[date_index-10:date_index+2])).ravel()
data_to_plot['upper_pred'][window_back+2] = data_to_plot['weather'][window_back+2] * (1 + pred_24.values[date_index+1][0])


chart = get_chart(data_to_plot)
st.altair_chart(chart, use_container_width=True)