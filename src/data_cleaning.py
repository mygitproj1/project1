import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os

# --- Configuration ---
DATA_PATH = "data/creditcard.csv" # **CHANGE: Ensure this file is downloaded and placed here**
OUTPUT_DIR = "data/processed"
SCALER_PATH = "src/scaler.joblib"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def preprocess_data(data_path: str = DATA_PATH, output_dir: str = OUTPUT_DIR) -> str:
    """Loads, preprocesses, splits, and saves the data/scaler."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    # Remove the 'Time' column (as it's often anonymized or not predictive here)
    # **NOTE: For real projects, you would convert 'Time' to Hour of Day/Day of Week.**
    df = df.drop('Time', axis=1)

    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Train-Test Split (use the full dataset for a mixed test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Filter Training Data to contain ONLY Normal (Class 0) transactions for Isolation Forest
    X_train_normal = X_train[y_train == 0].copy()
    
    # Fit StandardScaler only on Normal training data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X_train_normal)
    X_train_scaled = pd.DataFrame(scaled_features, columns=X_train_normal.columns)

    # Transform the mixed test set using the fitted scaler
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Save the necessary artifacts
    X_train_scaled.to_csv(os.path.join(output_dir, "X_train_normal.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, "X_test_mixed.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test_mixed.csv"), index=False)
    
    joblib.dump(scaler, SCALER_PATH)

    print(f"Data saved. Normal training samples: {len(X_train_scaled)}")
    return output_dir

if __name__ == "__main__":
    preprocess_data()