import streamlit as st
import pandas as pd
import os

st.set_page_config(layout="wide")

## --- Configuration ---
REPORT_PATH = "monitoring/data_drift_report.html"
LOG_FILE = "monitoring/live_data.csv"

st.title("💳 Financial Anomaly Detector Dashboard")
st.markdown("---")

# 1. Prediction Log / Live Data View
st.header("1. Live Transaction Feed")
if os.path.exists(LOG_FILE):
    df_live = pd.read_csv(LOG_FILE)
    st.dataframe(df_live.tail(10), use_container_width=True)
    st.markdown(f"**Total Transactions Logged:** {len(df_live)}")
    st.markdown(f"**Fraud Flagged (Last 100):** {df_live['is_fraud'].tail(100).sum()}")
else:
    st.info("No live transaction data yet. Run the FastAPI service and make test calls.")

st.markdown("---")

# 2. Data Drift Monitoring Report
st.header("2. Evidently AI Data Drift Monitoring")
st.info("Run `python monitoring/monitor_app.py` to generate the latest report.")
if os.path.exists(REPORT_PATH):
    with open(REPORT_PATH, 'r') as f:
        html_code = f.read()
    
    # Display the HTML report using an iframe
    st.components.v1.html(
        html_code, 
        height=800, 
        scrolling=True
    )
else:
    st.warning("Data Drift Report not found. Please run the monitoring script.")