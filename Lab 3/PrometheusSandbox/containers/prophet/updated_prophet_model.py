# Importing the necessary libraries
from prophet import Prophet
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import time
from prometheus_client import Gauge, start_http_server
import os

# Define the gauge metrics for Prometheus
anomaly_gauge = Gauge('anomaly_count', 'Number of anomalies detected')
mae_gauge = Gauge('MAE', 'Mean Absolute Error')
mape_gauge = Gauge('MAPE', 'Mean Absolute Percentage Error')

# Start the Prometheus metrics server on port 9095
# start_http_server(9095)


# Function to fetch data from Prometheus with proper error handling
def get_prometheus_data(train_gauge, start_time, end_time, step):
    """
    Fetches Prometheus data using the Prometheus API.
    :param train_gauge: Name of the Prometheus metric to query.
    :param start_time: Start time for the query.
    :param end_time: End time for the query.
    :param step: Time step for the query in seconds (e.g., '60s').
    :return: DataFrame containing the timestamp ('ds') and metric value ('y').
    """
    PROMETHEUS_URL = os.environ.get('PROMETHEUS_URL', 'http://localhost:9090')

    params = {
        'query': train_gauge,
        'start': start_time,
        'end': end_time,
        'step': step
    }

    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params)
        response.raise_for_status()
        data = response.json()

        if 'data' in data and len(data['data']['result']) > 0:
            values = data['data']['result'][0]['values']
            df = pd.DataFrame(values, columns=['ds', 'y'])
            df['ds'] = df['ds'].apply(lambda sec: datetime.fromtimestamp(float(sec)))
            df['y'] = df['y'].astype(float)
            return df
        else:
            return pd.DataFrame(columns=['ds', 'y'])

    except requests.exceptions.Timeout:
        print("Request timed out. Retrying...")
        return pd.DataFrame(columns=['ds', 'y'])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Prometheus: {e}")
        return pd.DataFrame(columns=['ds', 'y'])


# Function to initialize the Prophet model
def initialize_model():
    """
    Initializes and returns a Prophet model with flat growth.
    :return: Prophet model object
    """
    return Prophet(interval_width=0.99, growth="flat", yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)


# Function to clean the DataFrame by removing NaN values
def clean_dataframe(df):
    """
    Cleans the DataFrame by removing or imputing NaN values.
    :param df: DataFrame containing the data to clean.
    :return: Cleaned DataFrame.
    """
    # Drop rows with NaN values
    df = df.dropna()
    return df


# Main function to run the forecasting cycle
def run_forecasting_cycle():
    start_http_server(9095)
    # Initialize the DataFrame to store test step results
    results_df = pd.DataFrame(columns=['current_time', 'anomalies_detected', 'MAE', 'MAPE'])

    print("Fetching training data...")

    step = '5s'  # Time step for fetching data

    now = datetime.now(timezone.utc)
    start_time = (now - timedelta(minutes=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_time = now.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Fetch and clean training data
    train_dataframe = get_prometheus_data('train_gauge', start_time, end_time, '30s')
    train_dataframe = clean_dataframe(train_dataframe)

    # if train_dataframe.empty or train_dataframe.shape[0] < 2:
    #     print("Insufficient training data available after cleaning. Skipping cycle.")
    #     time.sleep(60)
    #     return

    print("Training model...")
    model = initialize_model()
    model.fit(train_dataframe)

    while True:
        print("Fetching test data...")
        # Wait 1 minute before pulling the test data
        time.sleep(60)

        # Fetch test data (last 1 minute)
        now = datetime.now(timezone.utc)
        test_start_time = (now - timedelta(minutes=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        test_end_time = now.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Fetch and clean test data
        test_dataframe = get_prometheus_data('test_gauge', test_start_time, test_end_time, '10s')
        test_dataframe = clean_dataframe(test_dataframe)

        if test_dataframe.empty:
            print("No test data available after cleaning. Skipping cycle.")
            continue

        # Making predictions using the Prophet model
        print("Making predictions...")
        prediction = model.predict(test_dataframe[['ds']])

        # Merge actual and predicted values
        performance = pd.merge(test_dataframe, prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

        # Evaluate model performance (MAE and MAPE)
        performance_MAE = mean_absolute_error(performance['y'], performance['yhat'])
        performance_MAPE = mean_absolute_percentage_error(performance['y'], performance['yhat'])
        print(f'The MAE for the model is {performance_MAE}')
        print(f'The MAPE for the model is {performance_MAPE}')

        # Identify anomalies
        performance['anomaly'] = performance.apply(
            lambda row: 1 if row['y'] < row['yhat_lower'] or row['y'] > row['yhat_upper'] else 0, axis=1)
        anomaly_count = performance['anomaly'].sum()

        # Set the values for Prometheus metrics
        anomaly_gauge.set(anomaly_count)
        mae_gauge.set(performance_MAE)
        mape_gauge.set(performance_MAPE)

        # Create a new row with the current test results
        current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        new_row = pd.DataFrame([{
            'current_time': current_time,
            'anomalies_detected': anomaly_count,
            'MAE': performance_MAE,
            'MAPE': performance_MAPE
        }])

        # Concatenate the new row to the results DataFrame
        results_df = pd.concat([results_df, new_row], ignore_index=True)  # Corrected line

        # Print the results DataFrame to the console
        print("Test Step Results:")
        print(results_df)

        # Repeat every 60 seconds
        print("Waiting for next cycle...\n")
        time.sleep(60)

# Run the forecasting cycle
run_forecasting_cycle()
