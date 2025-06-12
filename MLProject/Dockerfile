ARG RUN_ID
FROM python:3.12-slim

WORKDIR /app

COPY mlruns /app/mlruns

RUN pip install mlflow

ENV MLFLOW_TRACKING_URI=file:/app/mlruns

CMD ["sh", "-c", "mlflow models serve -m mlruns/0/${RUN_ID}/artifacts/model -p 5000"]
