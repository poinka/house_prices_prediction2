from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import pandas as pd
import os

app = FastAPI()

model = joblib.load(os.path.join(os.path.dirname(__file__), '../../../models/model.pkl'))
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'RoomsPerBedroom']

class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict(features: HousingFeatures):
    data = np.array([[features.MedInc, features.HouseAge, features.AveRooms, features.AveBedrms,
                      features.Population, features.AveOccup, features.Latitude, features.Longitude]])
    data_df = pd.DataFrame(data, columns=feature_names[:-1])  # Add RoomsPerBedroom
    data_df['RoomsPerBedroom'] = data_df['AveRooms'] / data_df['AveBedrms']
    prediction = model.predict(data_df)[0]
    return {"prediction": float(prediction)}