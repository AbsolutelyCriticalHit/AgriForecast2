import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
st.title("🌾 Farm Yield Prediction App")

@st.cache_data
def load_data():
    return pd.read_csv("yield_df.csv")

farmdata = load_data()
st.subheader("📊 Dataset Preview")
st.dataframe(farmdata)

# Prepare features and labels
X = farmdata[["average_rain_fall_mm_per_year", "avg_temp"]]
y = farmdata["hg/ha_yield"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# User Input
st.subheader("🧮 Predict Crop Yield")
rainfall = st.number_input("Average Rainfall (mm/year)", value=2702.0, min_value=0.0, step=10.0)
temperature = st.number_input("Average Temperature (°C)", value=27.5, step=0.1)

if st.button("Predict Yield"):
    prediction = model.predict([[rainfall, temperature]])[0]
    st.success(f"🌱 Predicted Yield: {prediction:.2f} hg/ha")

    # Display R² score
    score = model.score(X_test, y_test)
    st.info(f"Model R² Score: {score:.2f}")

# Optionally show the test set and actual values
with st.expander("📌 Show y_test values"):
    st.write(y_test.reset_index(drop=True))
