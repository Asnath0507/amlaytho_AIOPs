from prometheus_client import start_http_server, Gauge, Histogram
import random
import time


# Adding the new train and test metrics
g = Gauge('demo_gauge', 'Description of demo gauge')
train_gauge = Gauge('train_gauge', 'Train Gauge Metric')
test_gauge= Gauge('test_gauge', 'Test Gauge Metric')
train_histogram = Histogram('train_histogram', 'Train Histogram Metric')
test_histogram = Histogram('test_histogram', 'Test Histogram Metric') 


def emit_data():
    """Emit fake data"""
    # time.sleep(t)
    # g.set(t)
    # time.sleep(5) 
   
    # Generating random values
    random_val_0_1 = random.random()
    random_val_0_6 = random.random() * 0.6

    # Setting random values for 'train_gauge' and 'test_gauge' metrics 
    g.set(random_val_0_1)
    train_gauge.set(random_val_0_6)
    test_gauge.set(random_val_0_1)

   # For observing the random values in the histograms
    train_histogram.observe(random_val_0_6)
    test_histogram.observe(random_val_0_1)
      


if __name__ == '__main__':
    start_http_server(8000)
    while True:
        emit_data()
