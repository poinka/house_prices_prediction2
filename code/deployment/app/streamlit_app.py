import streamlit as st
import requests

st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("House Price Predictor")
st.markdown("Enter housing features to predict the median house value in dollars.")

col1, col2 = st.columns(2)
with col1:
    MedInc = st.number_input("Median Income ($10,000)", min_value=0.5, max_value=15.0, value=3.0, step=0.1)
with col1:
    HouseAge = st.number_input("House Age (years)", min_value=0, max_value=100, value=20, step=1)
with col1:
    AveRooms = st.number_input("Average Rooms", min_value=1, max_value=10, value=5, step=1)
with col1:
    AveBedrms = st.number_input("Average Bedrooms", min_value=1, max_value=6, value=2, step=1)
with col2:
    Population = st.number_input("Population (hundreds)", min_value=10, max_value=5000, value=100, step=10)
with col2:
    AveOccup = st.number_input("Average Occupancy", min_value=1, max_value=10, value=3, step=1)
with col2:
    Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.0, step=0.1)
with col2:
    Longitude = st.number_input("Longitude", min_value=-124.0, max_value=-114.0, value=-118.0, step=0.1)

if st.button("Predict"):
    data = {
        "MedInc": MedInc, "HouseAge": HouseAge, "AveRooms": AveRooms, "AveBedrms": AveBedrms,
        "Population": Population, "AveOccup": AveOccup, "Latitude": Latitude, "Longitude": Longitude
    }
    response = requests.post("http://api:8090/predict", json=data)
    prediction = response.json()["prediction"]
    st.success(f"Predicted House Value: ${prediction * 100000:.2f}")

st.markdown("<style> .stApp {background-color: #f0f2f6;} </style>", unsafe_allow_html=True)