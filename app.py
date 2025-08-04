import streamlit as st 
import joblib
import numpy as np

# Load the trained model
model = joblib.load('Iris.py')

st.title("Crop Yield Prediction Web Beta  (Indonesia Only)")

# Inputs
rain = st.number_input("Average Rainfall (mm/year)", min_value=0.0)
temp = st.number_input("Average Temperature (°C)", min_value=0.0)
pest = st.number_input("Pesticides Used (tonnes)", min_value=0.0)
item = st.selectbox("Crop Type", [
    'cassava',
    'maize',
    'potatoes',
    'rice, paddy',
    'soybeans',
    'sweet potatoes'
])

# Map item input to model column
item_map = {
    'cassava': 'Item_cassava',
    'maize': 'Item_maize',
    'potatoes': 'Item_potatoes',
    'rice, paddy': 'Item_rice, paddy',
    'soybeans': 'Item_soybeans',
    'sweet potatoes': 'Item_sweet potatoes'
}

# Model column order (must match training)
model_columns = [
    'Area',
    'average_rain_fall_mm_per_year',
    'pesticides_tonnes',
    'avg_temp',
    'Item_cassava',
    'Item_maize',
    'Item_potatoes',
    'Item_rice, paddy',
    'Item_soybeans',
    'Item_sweet potatoes'
]

# Prepare input
input_data = {col: 0 for col in model_columns}
input_data['average_rain_fall_mm_per_year'] = np.log(rain + 1)
input_data['avg_temp'] = np.log(temp + 1)
input_data['pesticides_tonnes'] = np.log(pest + 1)
input_data['Area'] = 1  # Hardcoded to Indonesia
input_data[item_map[item]] = 1  # One-hot for crop

# Final input array
X = np.array([input_data[col] for col in model_columns]).reshape(1, -1)

# Predict
if st.button("Predict", key="predict_button"):

    prediction = model.predict(X)[0]
    st.success(f"🌾 Predicted Yield: {prediction:.2f} hg/ha")

