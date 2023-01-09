import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from dashboard_utils import get_chart
import datetime


arimax_path = os.path.join(os.path.dirname(os.getcwd()), 'certainlyUncertain', 'models', 'model_save','arimax')
lightgbm_path = os.path.join(os.path.dirname(os.getcwd()), 'certainlyUncertain', 'dashboard', 'lightgbm')
sparse_path = os.path.join(os.path.dirname(os.getcwd()), 'certainlyUncertain', 'dashboard', 'sparse_regression_test')
test_data_path = os.path.join(os.path.dirname(os.getcwd()), 'certainlyUncertain', 'preprocessing', 'data')

# Load data
test_ec15 = pd.read_hdf(os.path.join(test_data_path, 'EC15_split.h5'),key="test")
test_gefs = pd.read_hdf(os.path.join(test_data_path, 'GEFS_split.h5'),key="test")

# LIGHT GBM DATA
lightgbm_ec15 = pd.read_csv(os.path.join(lightgbm_path, 'test_lgb_EC15.csv'))
lightgbm_gefs = pd.read_csv(os.path.join(lightgbm_path, 'test_lgb_GEFS.csv'))
lightgbm_ec15.columns = ['ens_std-' + col + '-24_pred' for col in lightgbm_ec15.columns]
lightgbm_gefs.columns = ['ens_std-' + col + '-24_pred' for col in lightgbm_gefs.columns]
lightgbm_ec15.rename(columns={'ens_std-date-24_pred':'date'}, inplace=True)
lightgbm_gefs.rename(columns={'ens_std-date-24_pred':'date'}, inplace=True)

df_lgb_ec15 = test_ec15.merge(lightgbm_ec15,how='left', left_on=test_ec15.index, right_on='date')
df_lgb_gefs = test_gefs.merge(lightgbm_gefs,how='left', left_on=test_gefs.index, right_on='date')

df_lgb_ec15['date'] = pd.to_datetime(df_lgb_ec15['date'])
df_lgb_ec15['date_short'] = pd.to_datetime(df_lgb_ec15['date'].dt.strftime('%Y-%m-%d'))
df_lgb_gefs['date'] = pd.to_datetime(df_lgb_gefs['date'])
df_lgb_gefs['date_short'] = pd.to_datetime(df_lgb_gefs['date'].dt.strftime('%Y-%m-%d'))

# Remove all rows with 12 in the hour column
df_lgb_ec15 = df_lgb_ec15[df_lgb_ec15['date'].dt.hour != 12]
df_lgb_gefs = df_lgb_gefs[df_lgb_gefs['date'].dt.hour != 12]
df_lgb_ec15.reset_index(drop=True, inplace=True)
df_lgb_gefs.reset_index(drop=True, inplace=True)



# SPARSE REGRESSION DATA
#read all the csv files in the sparse_regression_test folder
sparse_ec15 = pd.DataFrame()
sparse_gefs = pd.DataFrame()
for file in os.listdir(sparse_path):
      if 'EC15' in file:
            ec15_file = pd.read_csv(os.path.join(sparse_path, file))
            print(np.shape(ec15_file))
            ec15_file.rename(columns={'0':file}, inplace=True)
            ec15_file.rename(columns={file:file[:-4]}, inplace=True)
            sparse_ec15 = pd.concat([sparse_ec15, ec15_file], axis=1)
      elif 'GEFS' in file:
            gefs_file = pd.read_csv(os.path.join(sparse_path, file))
            gefs_file.rename(columns={'0':file}, inplace=True)
            gefs_file.rename(columns={file:file[:-4]}, inplace=True)
            sparse_gefs = pd.concat([sparse_gefs, gefs_file], axis=1)
sparse_ec15.columns = [col + '_pred' for col in sparse_ec15.columns]
sparse_gefs.columns = [col + '-_pred' for col in sparse_gefs.columns]



# ARIMAX DATA
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
# change test_ec15.index to datetime
date_var = pd.to_datetime(test_ec15.index, format='%Y-%m-%d')
date = st.sidebar.date_input('Select date', date_var[35])

# Select ML model
prediction_model = st.sidebar.selectbox('Select ML model', ['ARIMAX', 'VAR', 'Sparse Regression', 'LightGBM'])

# Select Physical model
ens_model = st.sidebar.selectbox('Select Physical model', ['EC15', 'GEFS'])

# Select country
country = st.sidebar.selectbox('Select country', countries)

# Select weather variable
weather_variable = st.sidebar.selectbox('Select weather variable', ['WIND', 'TEMP'])

if prediction_model == 'LightGBM':
   if ens_model == 'EC15':
         data = df_lgb_ec15
   elif ens_model == 'GEFS':
         data = df_lgb_gefs
else:
   if ens_model == 'EC15':
         data = df_lgb_ec15
   elif ens_model == 'GEFS':
         data = df_lgb_gefs

########## UNCERTAINTY METRICS ##########

# 24h forecast versus actual
st.subheader('Uncertainty Forecast')
st.write('Currently at: ', date)

#Filter df for 24h pred column and country in column name and date
pred_24 = data[[col for col in data.columns if '24' in col and country in col and weather_variable in col
                         and 'pred' in col]]
ens_48  = data[[col for col in data.columns if '48' in col and country in col and weather_variable in col
                         and 'ens' in col]]
model0_48 = data[[col for col in data.columns if '48' in col and country in col and weather_variable in col
                            and 'model_0' in col]]
date_inp = pd.to_datetime(date)
date_index = data[data['date_short'] == date_inp].index[0]
pred_24_metr = pred_24[data['date_short'] == date_inp]
ens_48_metr = ens_48[data['date_short'] == date_inp]
model0_48_metr = model0_48[data['date_short'] == date_inp]

m5, m6, m7 = st.columns((1,1,1))
m5.metric(label = 'Weather variable forecasted value', value = np.round(model0_48_metr.values[0][0],2))
m6.metric(label ='Machine Learning Model Uncertainty', value = np.round(pred_24_metr.values[0][0],2))
m7.metric(label ='Physical Model Uncertainty',value = np.round(ens_48_metr.values[0][0],2))

col1, col2, col3 = st.columns(3)
with col2:
   st.write('ML model uncertainty starts at:', date + pd.Timedelta(hours=24))
with col3:
   st.write('Physical model uncertainty starts at:', date )
with col1:
   st.write('Forecasting for:', date + pd.Timedelta(hours=48))


########## UNCERTAINTY EVOLUTION ##########

st.subheader('Uncertainty Evolution')

ens_24  = data[[col for col in data.columns if '24' in col and country in col and weather_variable in col
                         and 'ens' in col and 'pred' not in col]]
model0_24 = data[[col for col in data.columns if '24' in col and country in col and weather_variable in col
                            and 'model_0' in col]]

window_back = 10
data_to_plot = pd.DataFrame(columns=['date', 'weather', 'lower', 'upper', 'lower_pred', 'upper_pred'])
data_to_plot['date'] = data['date'][date_index-window_back:date_index+3]
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