# Use a Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /usr/src/app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

RUN sed -i 's/np.float_/np.float64/g' $(pip show prophet | grep Location | cut -d' ' -f2)/prophet/forecaster.py 

# Expose the application port (if applicable)
# EXPOSE 9095

# Run the application
CMD ["python", "updated_prophet_model.py"]
