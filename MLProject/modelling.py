# modelling.py

import mlflow
import mlflow.sklearn
import pandas as pd
import argparse
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mulai run MLflow eksplisit
with mlflow.start_run():
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error (MSE):", mse)

    # Logging metrik eksplisit
    mlflow.log_metric("mse", mse)

    # Simpan model ke file di direktori saat ini (MLProject)
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    joblib.dump(model, model_path)

    # Upload model ke MLflow artifact juga
    mlflow.log_artifact(model_path)
