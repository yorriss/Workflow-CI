# modelling.py

import mlflow
import mlflow.sklearn
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Parsing argumen dari MLProject
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='students_performance_preprocessing.csv')
args = parser.parse_args()

# Aktifkan autolog
mlflow.sklearn.autolog()

# Load dataset dari argumen
df = pd.read_csv(args.data_path)

# Fitur dan target
X = df.drop(columns='average_score').astype('float')
y = df['average_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediksi dan evaluasi
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Log manual (opsional karena autolog juga log)
mlflow.log_metric("mse", mse)
