FROM python:3.12-slim

WORKDIR /app

RUN pip install mlflow

# Jalankan MLflow serve langsung dari DagsHub
CMD mlflow models serve -m "models:/student-performance/Production" --no-conda --host 0.0.0.0 --port 5000
