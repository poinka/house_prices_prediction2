from sklearn.datasets import fetch_california_housing
import pandas as pd
import ssl
import certifi
import urllib.request
import os

context = ssl.create_default_context(cafile=certifi.where())
https_handler = urllib.request.HTTPSHandler(context=context)
opener = urllib.request.build_opener(https_handler)
urllib.request.install_opener(opener)

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))
os.makedirs(raw_dir, exist_ok=True)
df.to_csv(os.path.join(raw_dir, 'housing.csv'), index=False)