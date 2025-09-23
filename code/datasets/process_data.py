import pandas as pd
from sklearn.model_selection import train_test_split
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
raw_path = os.path.join(project_root, 'data', 'raw', 'housing.csv')
processed_dir = os.path.join(project_root, 'data', 'processed')

X_train_path = os.path.join(processed_dir, 'X_train.csv')
y_train_path = os.path.join(processed_dir, 'y_train.csv')
X_test_path = os.path.join(processed_dir, 'X_test.csv')
y_test_path = os.path.join(processed_dir, 'y_test.csv')

df = pd.read_csv(raw_path)

# Cleaning
df = df.dropna()  # Delete nulls
q1, q3 = df['MedInc'].quantile([0.25, 0.75])
df = df[(df['MedInc'] >= q1 - 1.5 * (q3 - q1)) & (df['MedInc'] <= q3 + 1.5 * (q3 - q1))] # Remove outliers

# Data split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
os.makedirs(processed_dir, exist_ok=True)
X_train.to_csv(X_train_path, index=False)
y_train.to_csv(y_train_path, index=False)
X_test.to_csv(X_test_path, index=False)
y_test.to_csv(y_test_path, index=False)
print(f"Processed data saved to {processed_dir}")