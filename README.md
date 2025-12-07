💳 Financial Anomaly Detection: MLOps End-to-End Pipeline
This repository contains a complete, production-ready MLOps pipeline for detecting fraudulent financial transactions using Isolation Forest. The project is designed strictly using open-source tools and is configured to run entirely in a local environment.

Shutterstock

🎯 Project Objective
The goal is to identify anomalous (fraudulent) transactions in a financial dataset. Unlike standard classification, this pipeline uses an Unsupervised Anomaly Detection approach, where the model is trained only on "normal" transactions to define a security boundary.

🛠️ The Tech Stack
Orchestration: Prefect (Local)

Experiment Tracking: MLflow (Local)

Data Versioning: DVC

Model Serving: FastAPI

Containerization: Docker

Monitoring: Evidently AI

Dashboard: Streamlit

CI/CD: GitHub Actions (Simulation)

🏗️ Project Structure
Plaintext

/mlops-fraud-detection/
|-- /data/                # Raw and processed data (Tracked by DVC)
|-- /src/                 # Core logic: cleaning, training, evaluation
|-- /app/                 # FastAPI application for model serving
|-- /monitoring/          # Evidently AI drift reports and live logs
|-- /streamlit_app/       # Visualization dashboard
|-- Dockerfile            # Container definition
|-- dvc.yaml              # Data version control stages
|-- .github/workflows/    # CI/CD definition
|-- requirements.txt      # Python dependencies
🚀 Getting Started
1. Prerequisites
Python 3.10+

Docker Desktop

Git & DVC

2. Installation & Setup
Bash

# Clone the repository
git clone https://github.com/your-username/mlops-fraud-detection.git
cd mlops-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc pull
3. Running the Pipeline
Ensure your local MLflow and Prefect servers are running in separate terminals:

Bash

# Terminal 1: MLflow UI
mlflow ui --port 5000

# Terminal 2: Prefect Orion
prefect orion start
Now, execute the end-to-end training pipeline:

Bash

python src/pipeline.py
4. Local Deployment (Docker)
Build and run the FastAPI service to serve predictions:

Bash

docker build -t fraud-detector .
docker run -d -p 8000:8000 --add-host=host.docker.internal:host-gateway fraud-detector
5. Monitoring & Dashboard
Trigger a data drift report and view the results on the dashboard:

Bash

# Generate report
python monitoring/monitor_app.py

# Launch Dashboard
streamlit run streamlit_app/dashboard.py
📊 Pipeline Components
A. Data Tracking (DVC)
We track creditcard.csv and the scaler.joblib artifact. Only the small .dvc metadata files are committed to Git, keeping the repository lightweight.

B. Experiment Tracking (MLflow)
Every run logs:

Parameters: contamination, n_estimators.

Metrics: AUC-ROC, Precision, Recall.

Artifacts: The trained Isolation Forest model is registered in the Model Registry.

C. Drift Monitoring (Evidently AI)
We monitor Data Drift by comparing the statistical distributions of incoming live transactions against the baseline training data. If significant drift is detected in feature columns (like Amount), a model retraining is triggered.

🤖 CI/CD Flow
The GitHub Actions workflow (ci_cd.yaml) ensures:

Python environment builds correctly.

Unit tests pass (if present).

The Docker image builds successfully.

DVC metadata is valid.
