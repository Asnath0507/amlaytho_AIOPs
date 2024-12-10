import json
import requests
import pandas as pd
from prophet import Prophet
from prometheus_client import Gauge, start_http_server
import time
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tabulate import tabulate
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Monitor Prometheus metrics.')
parser.add_argument("source_service", type=str, help="The source service for the metric.")
parser.add_argument("destination_service", type=str, help="The destination service for the metric.")
parser.add_argument("training_data", type=str, help="The training data file.")
parser.add_argument("prometheus_port", type=int, help="The port of the Prometheus server.")
args = parser.parse_args()

# Load training data from boutique_training.json
with open(args.training_data) as f:
    prom = json.load(f)

# Extract values based on the known structure
metric_data = prom['data']['result'][0]['values']

# Convert to DataFrame
df_train = pd.DataFrame(metric_data, columns=['ds', 'y'])

# Convert 'ds' column from timestamp to datetime and 'y' column to numeric
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')  # Convert 'y' to numeric, setting invalid parsing as NaN
df_train.dropna(subset=['y'], inplace=True)  # Drop any rows with NaN values in 'y'

# Reset training data to 0 origin before HMS conversion 
df_train['ds'] = df_train['ds'] - df_train['ds'].iloc[0]     
df_train['ds'] = df_train['ds'].apply(lambda sec: datetime.fromtimestamp(sec))

# Train the Prophet model on the training data
model = Prophet()
model.fit(df_train)

prefix = f"lab_7_{args.source_service}_2_{args.destination_service}"
# Define Prometheus metrics to expose
anomaly_gauge = Gauge(prefix + 'anomaly_count', 'Number of detected anomalies')
mae_gauge = Gauge(prefix + 'mae_score', 'Mean Absolute Error (MAE)')
mape_gauge = Gauge(prefix + 'mape_score', 'Mean Absolute Percentage Error (MAPE)')
y_min_gauge = Gauge(prefix + 'y_min', 'Minimum value of the metric')
y_max_gauge = Gauge(prefix + 'y_max', 'Maximum value of the metric')
y_gauge = Gauge(prefix + 'current_value', 'Current value of the metric')
yhat_gauge = Gauge(prefix + 'predicted_value', 'Predicted value by Prophet')

# Start the Prometheus server to expose metrics on port 8000
start_http_server(args.prometheus_port)
logger.info("Prometheus server started on port " + str(args.prometheus_port))

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
        return float(timestamp), float(value) if value not in ('NaN', 'null') else None
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
            logger.warning("Data fetch failed. Pick a god and pray")
            time.sleep(60)  # Wait and try again if data fetch failed
            continue

        # Create test datapoint
        current_time = time.time() - test_start_time
        df_test = pd.DataFrame({'ds': [datetime.fromtimestamp(current_time)], 
                               'y': [value]})
        
        # Make a prediction using the Prophet model
        forecast = model.predict(df_test)
        predicted_value = forecast['yhat'].iloc[0]
        y_min = forecast['yhat_lower'].iloc[0]
        y_max = forecast['yhat_upper'].iloc[0]

        logger.info(f"Actual value: {value}, Predicted value: {predicted_value}")
                
        # Calculate residual (error) between actual and predicted
        residual = abs(value - predicted_value)
        
        # Determine if this is an anomaly
        threshold = df_train['y'].std() * 2  # 2 standard deviations as an example threshold
        is_anomaly = residual > threshold

        # Count anomalies in current iteration
        anomaly_count = 1 if is_anomaly else 0

        # Update Prometheus metrics
        anomaly_gauge.set(anomaly_count)
        y_gauge.set(value)
        yhat_gauge.set(predicted_value)
        y_min_gauge.set(y_min)
        y_max_gauge.set(y_max)

        # Only calculate MAE and MAPE if values are valid
        if not(pd.isna(value) or pd.isna(predicted_value)):
            # Calculate MAE and MAPE
            mae = mean_absolute_error([value], [predicted_value])
            mape = mean_absolute_percentage_error([value], [predicted_value])
            mae_gauge.set(mae)
            mape_gauge.set(mape)
            
            # Log the MAE and MAPE
            logger.info(f"MAE: {mae}, MAPE: {mape}, Anomaly: {anomaly_count}")

            # Add the new row to the results list
            results.append({
                'Timestamp': datetime.now(),
                'Anomalies': anomaly_count,
                'MAE': mae,
                'MAPE': mape
            })
        else:
            # Handle case where MAE and MAPE are not calculated
            results.append({
                'Timestamp': datetime.now(),
                'Anomalies': anomaly_count,
                'MAE': 'N/A',
                'MAPE': 'N/A'
            })

        # Print the results in tabular form
        if len(results) > 0:
            df_results = pd.DataFrame(results)
            print(tabulate(df_results, headers='keys', tablefmt='grid', showindex=False))
        else:
            logger.warning("Results list is empty!")

        # Wait before the next iteration
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    monitor()