import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import joblib
import os

# Настройка трекинг URI
mlflow.set_tracking_uri("http://localhost:5001")

# Относительный путь к файлам от корня проекта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

X_train_path = os.path.join(project_root, 'data', 'processed', 'X_train.csv')
y_train_path = os.path.join(project_root, 'data', 'processed', 'y_train.csv')
X_test_path = os.path.join(project_root, 'data', 'processed', 'X_test.csv')
y_test_path = os.path.join(project_root, 'data', 'processed', 'y_test.csv')

models_dir = os.path.join(project_root, 'models')
model_path = os.path.join(models_dir, 'model.pkl')

# Загрузка данных
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).values.ravel()
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).values.ravel()

# Фича-инжиниринг
X_train['RoomsPerBedroom'] = X_train['AveRooms'] / X_train['AveBedrms']
X_test['RoomsPerBedroom'] = X_test['AveRooms'] / X_test['AveBedrms']

# Обучение и логирование в MLflow
with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model", input_example=X_train.head())
    print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

# Сохранение модели
os.makedirs(models_dir, exist_ok=True)
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")