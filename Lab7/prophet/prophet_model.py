import json
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
import pandas as pd
from prophet import Prophet
from prometheus_client import Gauge, start_http_server
import time
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tabulate import tabulate

# Load training data from boutique_training.json
with open("/app/new_boutique_training.json") as f:
    prom = json.load(f)

# Extract values based on the known structure
metric_data = prom['data']['result'][0]['values']

# Convert to DataFrame
df_train = pd.DataFrame(metric_data, columns=['ds', 'y'])

# Convert 'y' column to numeric
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
df_train.dropna(subset=['y'], inplace=True)

# Reset training data to 0 origin before HMS conversion
df_train['ds'] = df_train['ds'] - df_train['ds'].iloc[0]
df_train['ds'] = df_train['ds'].apply(lambda sec: datetime.fromtimestamp(sec))

# Train the Prophet model on the training data
model = Prophet()
model.fit(df_train)

# Define Prometheus metrics to expose
anomaly_gauge = Gauge('anomaly_count', 'Number of detected anomalies')
mae_gauge = Gauge('mae_score', 'Mean Absolute Error (MAE)')
mape_gauge = Gauge('mape_score', 'Mean Absolute Percentage Error (MAPE)')
y_min_gauge = Gauge('y_min', 'Minimum value of the metric')
y_max_gauge = Gauge('y_max', 'Maximum value of the metric')
y_gauge = Gauge('current_value', 'Current value of the metric')
yhat_gauge = Gauge('predicted_value', 'Predicted value by Prophet')



results = []

def fetch_current_data():
    """Fetch current metric data from Prometheus."""
    query = "histogram_quantile(0.5, sum by (le) (rate(istio_request_duration_milliseconds_bucket{app='frontend', destination_app='shippingservice', reporter='source'}[1m])))"
    try:
        response = requests.get('http://prometheus.istio-system:9090/api/v1/query', 
                                params={'query': query}, 
                                timeout=10)  # Add a timeout
        response.raise_for_status()
        data = response.json()['data']['result']
        if not data:
            print("No data returned from Prometheus.", flush=True)
            return None, None
        timestamp, value = data[0]['value']
        return float(timestamp), float(value)
    except Timeout:
        print("Connection to Prometheus timed out. Check if the server is reachable.", flush=True)
    except ConnectionError:
        print("Failed to connect to Prometheus. Check network connectivity.", flush=True)
    except RequestException as e:
        print(f"Error fetching data from Prometheus: {e}", flush=True)
    except (KeyError, IndexError) as e:
        print(f"Unexpected data format received from Prometheus: {e}", flush=True)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", flush=True)
    return None, None

def monitor():
    """Continuously fetch data, make predictions, and update Prometheus metrics."""
    test_start_time = time.time()
    while True:
        try:
            # Fetch the current data point
            timestamp, value = fetch_current_data()
            if value is None:
                print("Data fetch failed. Retrying in 60 seconds...", flush=True)
                time.sleep(60)
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

            print(f"Actual value: {value}" , flush=True)
            print(f"Predicted value: {predicted_value}", flush=True)
            
            # Calculate residual (error) between actual and predicted
            residual = abs(value - predicted_value)
            
            # Determine if this is an anomaly
            threshold = df_train['y'].std() * 3
            is_anomaly = residual > threshold

            # Count anomalies in current iteration
            anomaly_count = 1 if is_anomaly else 0

            # Update Prometheus metrics
            anomaly_gauge.set(anomaly_count)
            y_gauge.set(value)
            yhat_gauge.set(predicted_value)
            y_min_gauge.set(y_min)
            y_max_gauge.set(y_max)

            # Calculate MAE and MAPE
            if not(pd.isna(value) or pd.isna(predicted_value)):
                mae = mean_absolute_error([value], [predicted_value])
                mape = mean_absolute_percentage_error([value], [predicted_value])
                if mae is not None and mape is not None:
                    mae_gauge.set(mae)
                    mape_gauge.set(mape)
            else:
                mae = None
                mape = None

            # Add the new row to the results list
            results.append({
                'Timestamp': datetime.now(),
                'Anomalies': anomaly_count,
                'MAE': mae if mae is not None else 'N/A',
                'MAPE': mape if mape is not None else 'N/A'
            })

            # Print the results in tabular form
            df_results = pd.DataFrame(results)
            print(tabulate(df_results, headers='keys', tablefmt='grid', showindex=False), flush=True)

        except Exception as e:
            print(f"An error occurred in the monitor loop: {e}", flush=True)
            print("Retrying in 60 seconds...", flush=True)
        
        time.sleep(60)

if __name__ == "__main__":
    # Start the Prometheus server to expose metrics on port 8080
    start_http_server(8080)
    monitor()