import json
import requests
import pandas as pd
from prophet import Prophet
from prometheus_client import Gauge, start_http_server
import time
import argparse
import logging
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tabulate import tabulate

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Dynamic Service Configuration for Monitor App")
parser.add_argument("source_service", help="Name of the source service")
parser.add_argument("destination_service", help="Name of the destination service")
parser.add_argument("training_data", help="Path to the training data file")
parser.add_argument("prometheus_port", type=int, help="Port number for Prometheus server")
args = parser.parse_args()

# Dynamic metric prefix
prefix = f"lab7_{args.source_service}_2_{args.destination_service}_"

# Load training data
with open(args.training_data) as f:
    prom = json.load(f)

# Extract and process the training data
metric_data = prom['data']['result'][0]['values']
df_train = pd.DataFrame(metric_data, columns=['ds', 'y'])
df_train['ds'] = pd.to_numeric(df_train['ds'], errors='coerce')
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
df_train.dropna(inplace=True)

# Normalize 'ds' and convert timestamps
df_train['ds'] = df_train['ds'] - df_train['ds'].iloc[0]
df_train['ds'] = df_train['ds'].apply(lambda sec: datetime.fromtimestamp(sec))

# Train Prophet model
model = Prophet()
model.fit(df_train)
logger.info("Prophet model trained successfully.")

# Define Prometheus metrics
anomaly_gauge = Gauge(prefix + 'anomaly_count', 'Number of detected anomalies')
mae_gauge = Gauge(prefix + 'mae_score', 'Mean Absolute Error (MAE)')
mape_gauge = Gauge(prefix + 'mape_score', 'Mean Absolute Percentage Error (MAPE)')
y_min_gauge = Gauge(prefix + 'y_min', 'Minimum value of the metric')
y_max_gauge = Gauge(prefix + 'y_max', 'Maximum value of the metric')
y_gauge = Gauge(prefix + 'current_value', 'Current value of the metric')
yhat_gauge = Gauge(prefix + 'predicted_value', 'Predicted value by Prophet')

# Start Prometheus server
start_http_server(args.prometheus_port)
logger.info(f"Prometheus server started on port {args.prometheus_port}.")

results = []

def fetch_current_data():
    """Fetch current metric data from Prometheus."""
    query = f"histogram_quantile(0.5, sum by (le) (rate(istio_request_duration_milliseconds_bucket{{app='{args.source_service}', destination_app='{args.destination_service}', reporter='source'}}[1m])))"
    try:
        response = requests.get('http://prometheus.istio-system:9090/api/v1/query', params={'query': query})
        response.raise_for_status()
        data = response.json()['data']['result']
        if not data:
            logger.warning("No data returned from Prometheus.")
            return None, None
        timestamp, value = data[0]['value']
        if value.lower() == 'nan':
            logger.warning("Received NaN value from Prometheus. Skipping...")
            return None, None
        return float(timestamp), float(value)
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None, None

def monitor():
    """Continuously fetch data, make predictions, and update Prometheus metrics."""
    test_start_time = time.time()
    while True:
        # Fetch the current data point
        timestamp, value = fetch_current_data()
        if value is None:
            logger.warning("Data fetch failed. Retrying in 60 seconds.")
            time.sleep(60)
            continue
        value = value if not pd.isna(value) else 0

        # Create test data point
        current_time = time.time() - test_start_time
        df_test = pd.DataFrame({'ds': [datetime.fromtimestamp(current_time)], 
                                'y': [value]})
        
        # Make a prediction using the Prophet model
        forecast = model.predict(df_test)
        predicted_value = forecast['yhat'].iloc[0]
        y_min = forecast['yhat_lower'].iloc[0]
        y_max = forecast['yhat_upper'].iloc[0]

        logger.info(f"Actual value: {value}, Predicted value: {predicted_value}")
        if value is None or pd.isna(value):
            logger.warning("Received NaN value from Prometheus. Skipping...")
            time.sleep(60)
            continue

        # Calculate residual (error) between actual and predicted
        residual = abs(value - predicted_value)
        threshold = df_train['y'].std() * 1.5
        is_anomaly = residual > threshold
        anomaly_count = 1 if is_anomaly else 0

        # Update Prometheus metrics
        anomaly_gauge.set(anomaly_count)
        y_gauge.set(value)
        yhat_gauge.set(predicted_value)
        y_min_gauge.set(y_min)
        y_max_gauge.set(y_max)

        # Calculate MAE and MAPE
        # remove none values from value and predicted_value
        value = value if not pd.isna(value) else 0
        predicted_value = predicted_value if not pd.isna(predicted_value) else 0

        if not(pd.isna(value) or pd.isna(predicted_value)):
            mae = mean_absolute_error([value], [predicted_value])
            mape = mean_absolute_percentage_error([value], [predicted_value])
            mae_gauge.set(mae)
            mape_gauge.set(mape)
        else:
            mae = None
            mape = None

        results.append({
            'Timestamp': datetime.now(),
            'Anomalies': anomaly_count,
            'MAE': mae if mae is not None else 'N/A',
            'MAPE': mape if mape is not None else 'N/A'
        })

        # Print the results in tabular form
        df_results = pd.DataFrame(results)
        print(tabulate(df_results, headers='keys', tablefmt='grid', showindex=False))

        # Wait before the next iteration
        time.sleep(60)

if __name__ == "__main__":
    monitor()