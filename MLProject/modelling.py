# modelling.py

import mlflow
import mlflow.sklearn
import pandas as pd
import argparse
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Parsing argumen dari MLProject
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='students_performance_preprocessing.csv')
args = parser.parse_args()

# Aktifkan autologging
mlflow.sklearn.autolog()

# Load dataset
df = pd.read_csv(args.data_path)

# Fitur dan target
X = df.drop(columns='average_score').astype(float)
y = df['average_score']

# Standarisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Buat direktori artifacts jika belum ada
script_dir = os.path.dirname(os.path.abspath(__file__))
artifacts_dir = os.path.join(script_dir, "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

# Mulai run MLflow eksplisit
with mlflow.start_run():
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error (MSE):", mse)

    # Simpan model
    model_path = os.path.join(artifacts_dir, "model.pkl")
    joblib.dump(model, model_path)

    # Simpan scaler
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # Simpan report JSON
    report = {"mse": mse, "r2_score": r2}
    report_path = os.path.join(artifacts_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    # Confusion matrix bukan untuk regresi, jadi plot residual distribusi
    plt.figure(figsize=(6, 4))
    sns.histplot(y_test - y_pred, kde=True)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.tight_layout()
    cm_path = os.path.join(artifacts_dir, "residual_plot.png")
    plt.savefig(cm_path)
    plt.close()

    # Logging manual artifacts ke MLflow
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(scaler_path)
    mlflow.log_artifact(report_path)
    mlflow.log_artifact(cm_path)
