import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Crop Yield Predictor", layout="centered")
st.title("🌾 Crop Yield Predictor")

# Upload CSV file
st.header("1. Upload Farm Data")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    farmdata = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(farmdata.head())

    target_col = "hg/ha_yield"
    feature_cols = [col for col in farmdata.columns if col != target_col]

    # Train model
    X = farmdata[feature_cols]
    y = farmdata[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    st.success("✅ Model trained successfully!")

    # Input fields for each feature
    st.header("2. Enter Input Values")
    input_data = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(farmdata[col]):
            val = st.number_input(f"{col}", value=float(farmdata[col].mean()))
            input_data[col] = val
        else:
            val = st.text_input(f"{col}")
            input_data[col] = val

    # Predict yield
    if st.button("🚀 Predict Yield"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        score = model.score(X_test, y_test)

        st.subheader(f"🌿 Predicted Yield: `{prediction:.2f} hg/ha`")
        st.caption(f"Model Accuracy: {score * 100:.2f}%")
