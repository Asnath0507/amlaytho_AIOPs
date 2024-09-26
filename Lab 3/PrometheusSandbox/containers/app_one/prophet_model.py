# Importing the important libraries
from prophet import Prophet  # Corrected import for Prophet
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import json

# Reading the data from the Train Json File
with open("train_gauge.json", 'r') as train_file:  # Use 'with' to manage file closing automatically
    prom = json.load(train_file)

# Create DataFrame from Prometheus data
train_dataframe = pd.DataFrame(prom['data']['result'][0]['values'])
train_dataframe.columns = ['ds', 'y']  # Assigning column names
train_dataframe['ds'] = train_dataframe['ds'].apply(lambda sec: datetime.fromtimestamp(float(sec)))  # Ensure float for Unix timestamp
train_dataframe['y'] = train_dataframe['y'].astype(float)  # Ensure target values are floats
train_dataframe

# Reading the data from the Test Json File
with open("test_gauge.json", 'r') as test_file:  # Use 'with' to manage file closing automatically
    prom = json.load(test_file)

# Create DataFrame from Prometheus data
test_dataframe = pd.DataFrame(prom['data']['result'][0]['values'])
test_dataframe.columns = ['ds', 'y']  # Assigning column names
test_dataframe['ds'] = test_dataframe['ds'].apply(lambda sec: datetime.fromtimestamp(float(sec)))  # Ensure float for Unix timestamp
test_dataframe['y'] = test_dataframe['y'].astype(float)  # Ensure target values are floats
test_dataframe

# Making Predictions Using Prophet in Python
# Removed 'growth="flat"' since Prophet doesn't support it. Use default growth model or specify 'linear'.
model = Prophet(interval_width=0.99, yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
model.fit(train_dataframe)

# Creating a future dataframe based on the test data for prediction
prediction = model.predict(test_dataframe[['ds']])  # Model only needs 'ds' for future prediction


# Merge actual and predicted values
performance = pd.merge(test_dataframe, prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

# Finding MAE value
performance_MAE = mean_absolute_error(performance['y'], performance['yhat'])
print(f'The MAE for the model is {performance_MAE}')

# Finding MAPE value
performance_MAPE = mean_absolute_percentage_error(performance['y'], performance['yhat'])
print(f'The MAPE for the model is {performance_MAPE}')

# Creating an anomaly indicator
performance['anomaly'] = performance.apply(lambda rows: 1 if ((rows.y < rows.yhat_lower) | (rows.y > rows.yhat_upper)) else 0, axis=1)

# Take a look at the anomalies
anomalies = performance[performance['anomaly'] == 1].sort_values(by='ds')
print(anomalies)

