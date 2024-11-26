# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Patch the prophet library to replace np.float_ with np.float64
RUN sed -i 's/np.float_/np.float64/g' $(pip show prophet | grep Location | cut -d' ' -f2)/prophet/forecaster.py

# Expose port for Prometheus metrics
EXPOSE 8080

# Run the application
CMD ["python", "prophet_model.py"]
