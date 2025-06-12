FROM python:3.12-slim

WORKDIR /app

RUN pip install mlflow

# Jalankan model langsung dari DagsHub (sesuaikan jika perlu)
CMD mlflow models serve -m "runs:/f5c77a366573403bb706e6cb96c3078e/model" --no-conda --host 0.0.0.0 --port 5000
