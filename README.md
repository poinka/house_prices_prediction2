# Extended House Prices Prediction web site with Pipelines

This repository implements an automated machine learning pipeline with three stages—Data Engineering, Model Engineering, and Deployment—using Airflow, MLflow, DVC, and Docker. The pipeline is scheduled to run every 5 minutes (configurable).
## Features
This project automates a complete ML workflow:

- **Data Engineering**: Loads raw data, cleans it, and splits it into training and testing sets.
- **Model Engineering**: Performs feature engineering, trains a RandomForestRegressor model, evaluates it with multiple metrics, and logs results using MLflow.
- **Deployment**: Deploys a FastAPI-based model API and a Streamlit app in separate Docker containers.

The pipeline is managed by Airflow Standalone, which integrates scheduling, web UI, and API services in a single process.

## Repository structure
  ├── code
  │   ├── datasets
  │   │   └── process_data.py          # Data cleaning and splitting script
  │   │   └── save_raw.py              # Save dataset using sklearn.datasets
  │   └── deployment
  │       ├── api
  │       │   ├── main.py             # FastAPI model API
  │       │   └── Dockerfile          # Docker configuration for API
  │       ├── app
  │       │   ├── app.py              # Streamlit application
  │       │   └── Dockerfile          # Docker configuration for app
  │       └ docker-compose.yml        # Docker Compose configuration
  ├── .gitignore                       # Git ignore file
  │   └── models
  │       └── train.py                # Model training and evaluation script
  ├── data
  │   ├── processed                    # Processed train/test data (tracked by DVC)
  │   └── raw                          # Raw data (tracked by DVC)
  ├── dags
  │   └── mlops_pipeline.py    # Airflow DAG definition
  │  
  ├── models                           # Trained model files (tracked by DVC)
  ├── mlruns                           # MLflow tracking directory
  ├── requirements.txt                 # Python dependencies
  ├── airflow.cfg                      # Airflow configuration
  └── README.md                        # This file

  
  ## Requirements
- Python 3.10
- Docker
- Git
- Internet connection for package installation
- 
## Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd house_prices_prediction2
   ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ``` 
