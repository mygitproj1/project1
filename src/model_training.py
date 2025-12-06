import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.sklearn
import os

# --- Configuration ---
PROCESSED_DATA_DIR = "data/processed"
MLFLOW_TRACKING_URI = "http://0.0.0.0:5000"
EXPERIMENT_NAME = "IsolationForest_FraudDetection"
CONTAMINATION_RATE = 0.0017 # Based on typical fraud rate (~0.17%)

def train_isolation_forest():
    # 1. MLflow Setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2. Load Data
    X_train_normal = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_train_normal.csv"))
    X_test_mixed = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_test_mixed.csv"))
    y_test_mixed = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_test_mixed.csv")).squeeze()

    with mlflow.start_run(run_name="IsolationForest_Run"):
        # 3. Model Definition and Training
        iso_forest = IsolationForest(
            n_estimators=100, 
            contamination=CONTAMINATION_RATE, 
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(X_train_normal)

        # 4. Evaluation
        # Isolation Forest uses decision_function for anomaly scores: lower is more anomalous.
        # We negate the score so higher values correspond to higher fraud risk (like a probability).
        anomaly_scores = -iso_forest.decision_function(X_test_mixed)
        
        # Calculate AUC-ROC (y_true must be 0/1, anomaly_scores are the probabilities/scores)
        # Note: Fraud is class 1.
        auc_roc = roc_auc_score(y_test_mixed, anomaly_scores)

        # 5. MLflow Logging
        mlflow.log_param("contamination", CONTAMINATION_RATE)
        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_metric("AUC_ROC", auc_roc)
        
        # Log model and register it
        mlflow.sklearn.log_model(
            sk_model=iso_forest,
            artifact_path="isolation_forest_model",
            registered_model_name="FraudDetector_IF"
        )
        print(f"Trained Isolation Forest model with AUC-ROC: {auc_roc:.4f}")
        
        # **CHANGE: Manually promote the best model to 'Production' via MLflow UI or API**

if __name__ == "__main__":
    train_isolation_forest()