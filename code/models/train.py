import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
import joblib
import os
from datetime import datetime

# Set MLflow tracking URI to local server
mlflow.set_tracking_uri("http://localhost:5001")

# Define project root for file paths relative to the script location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Define paths for training and testing data
X_train_path = os.path.join(project_root, 'data', 'processed', 'X_train.csv')
y_train_path = os.path.join(project_root, 'data', 'processed', 'y_train.csv')
X_test_path = os.path.join(project_root, 'data', 'processed', 'X_test.csv')
y_test_path = os.path.join(project_root, 'data', 'processed', 'y_test.csv')

# Define path for saving the trained model
models_dir = os.path.join(project_root, 'models')
model_path = os.path.join(models_dir, 'model.pkl')

# Load data from processed files
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).values.ravel()  # Ensure y is 1D array
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).values.ravel()

# Feature Engineering: Create meaningful features to improve model performance
# Calculate rooms per bedroom and add it as a new feature
X_train['RoomsPerBedroom'] = X_train['AveRooms'] / X_train['AveBedrms'].replace(0, 1e-6)  # Avoid division by zero
X_test['RoomsPerBedroom'] = X_test['AveRooms'] / X_test['AveBedrms'].replace(0, 1e-6)

# Normalize features to improve convergence
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# Define hyperparameter grid for GridSearchCV to optimize model
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20]
}

# Initialize and train model with hyperparameter tuning
print("Started Greed Search")
with mlflow.start_run(run_name=f"RF_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    # Use GridSearchCV to find the best hyperparameters
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate multiple metrics for comprehensive evaluation
    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    r2 = r2_score(y_test, y_pred)  # R-squared for goodness of fit

    # Log parameters, metrics, and model to MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics({
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2
    })
    mlflow.sklearn.log_model(best_model, "random_forest_model", input_example=X_train.head())
    print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# Save the best model to disk for deployment
os.makedirs(models_dir, exist_ok=True)  # Create models directory if it doesn't exist
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")

# Log model path to MLflow for traceability
mlflow.log_artifact(model_path)