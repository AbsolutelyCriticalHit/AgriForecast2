import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="🌾 Crop Yield Predictor", layout="centered")
st.title("🌾 Crop Yield Predictor")

# Load the dataset (no upload needed)
DATA_PATH = "yield_df.csv"

try:
    df = pd.read_csv(DATA_PATH)
    st.success("✅ Dataset loaded successfully!")
    st.write("Preview of the dataset:")
    st.dataframe(df.head())

    # Features and label (same as Iris.py)
    X = df[["average_rain_fall_mm_per_year", "avg_temp"]]
    y = df["hg/ha_yield"]

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # User inputs
    st.header("📥 Enter Rainfall and Temperature")
    rainfall = st.number_input("Average Rainfall (mm/year)", value=2702.0)
    temperature = st.number_input("Average Temperature (°C)", value=27.5)

    if st.button("🚀 Predict"):
        pred = model.predict([[rainfall, temperature]])[0]
        score = model.score(X_test, y_test)

        st.subheader(f"🌿 Predicted Yield: `{pred:.2f} hg/ha`")
        st.caption(f"📊 Model R² Score (Accuracy): `{score * 100:.2f}%`")

except FileNotFoundError:
    st.error(f"❌ File `{DATA_PATH}` not found.")
except Exception as e:
    st.exception(e)
