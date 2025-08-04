import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="🌾 Crop Yield Predictor", layout="centered")
st.title("🌾 Crop Yield Predictor")

# Load CSV directly from the repo
DATA_PATH = "yield_df.csv"
try:
    farmdata = pd.read_csv(DATA_PATH)
    st.success("✅ Dataset loaded successfully!")

    st.subheader("📄 Data Preview")
    st.dataframe(farmdata.head())

    target_col = "hg/ha_yield"
    feature_cols = [col for col in farmdata.columns if col != target_col and pd.api.types.is_numeric_dtype(farmdata[col])]

    # Train the model
    X = farmdata[feature_cols]
    y = farmdata[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    st.success("✅ Model trained on numeric columns")

    # Input features
    st.header("📥 Input Values for Prediction")
    input_data = {}
    for col in feature_cols:
        default_val = float(farmdata[col].mean())
        input_data[col] = st.number_input(f"{col}", value=default_val)

    if st.button("🚀 Predict Yield"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        score = model.score(X_test, y_test)

        st.subheader(f"🌾 Predicted Yield: `{prediction:.2f} hg/ha`")
        st.caption(f"Model Accuracy: `{score * 100:.2f}%`")

except FileNotFoundError:
    st.error(f"❌ Could not find `{DATA_PATH}` in the repo.")
except Exception as e:
    st.exception(e)
