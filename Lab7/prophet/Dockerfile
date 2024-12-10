# Use an official Python runtime as a parent image
FROM python:3.9.6

# Install system-level dependencies for Prophet and other Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libatlas-base-dev \
    gfortran \
    && apt-get clean

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the working directory
COPY requirements.txt /app/

# Install Python dependencies with a longer timeout to avoid timeouts
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Fix compatibility issues in Prophet with NumPy versions
RUN sed -i 's/np.float_/np.float64/g' $(pip show prophet | grep Location | cut -d" " -f2)/prophet/forecaster.py

# Copy the current directory contents into the container at /app
COPY . /app

# Run your monitoring application
CMD ["python", "prophet_model.py"]