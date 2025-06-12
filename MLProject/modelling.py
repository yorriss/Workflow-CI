import mlflow
import mlflow.sklearn
import pandas as pd
import argparse
import joblib
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def main(data_path: str):
    # Setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(script_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(data_path)
    X = df.drop(columns='average_score').astype(float)
    y = df['average_score']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        # Log model in MLflow format
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Save scaler locally and log as artifact
        scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        # Generate and save residual plot
        plt.figure(figsize=(6, 4))
        residuals = y_test - y_pred
        plt.hist(residuals, density=True, bins=30, alpha=0.7)
        plt.title("Residual Distribution")
        plt.xlabel("Residual")
        plt.tight_layout()
        plot_path = os.path.join(artifacts_dir, "residual_plot.png")
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

        # Save JSON report and log
        report = {"mse": mse, "r2_score": r2}
        report_path = os.path.join(artifacts_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact(report_path)

    print("Training and logging finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and log RandomForest model with MLflow")
    parser.add_argument(
        "--data_path", "--data-path",
        dest="data_path",
        type=str,
        default="students_performance_preprocessing.csv",
        help="Path to preprocessed CSV dataset"
    )
    args = parser.parse_args()
    main(args.data_path)
