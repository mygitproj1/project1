# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy DVC-tracked scaler and model dependencies (for model loading)
COPY src/scaler.joblib src/

# Copy the application code
COPY src /app/src
COPY app /app/app
COPY monitoring /app/monitoring

# Set environment variable for the MLflow server URL
# **CHANGE: Replace with the actual host IP if not running locally**
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000 

# Expose the FastAPI port
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]