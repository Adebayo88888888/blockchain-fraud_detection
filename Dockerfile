# Use a lightweight Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Copy dependency files first for caching
COPY Pipfile Pipfile.lock* ./

# Install dependencies system-wide from Pipfile.lock
RUN pipenv install --system --deploy --ignore-pipfile

# Copy application files
COPY predict_service_api.py .
COPY trained_xgb_model.joblib .

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI service
CMD ["uvicorn", "predict_service_api:app", "--host", "0.0.0.0", "--port", "8000"]
