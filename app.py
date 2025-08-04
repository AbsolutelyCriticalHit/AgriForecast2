import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("🌾 Crop Yield Predictor (All Inputs)")

# Step 1: Upload CSV
st.header("1. Upload Farm Data")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load CSV
    farmdata = pd.read_csv(uploaded_file)
    st.subheader("Preview of Data")
    st.write(farmdata.head())

    # Automatically detect feature columns (exclude target)
    target_col = "hg/ha_yield"
    feature_cols = [col for col in farmdata.columns if col != target_col]

    # Show detected features
    st.write("Using the following columns as input features:")
    st.write(feature_cols)

    # Train model
    X = farmdata[feature_cols]
    y = farmdata[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    st.success("✅ Model trained successfully!")

    # Step 2: Create dynamic input fields for each feature
    st.header("2. Enter Feature Values for Prediction")
    user_inputs = {}
    for col in feature_cols:
        dtype = farmdata[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            default = float(farmdata[col].mean())
            user_inputs[col] = st.number_input(f"{col}", value=default)
        else:
            user_inputs[col] = st.text_input(f"{col}")

    # Predict
    if st.button("Predict"):
        # Turn user input into DataFrame
        input_df = pd.DataFrame([user_inputs])
        prediction = model.predict(input_df)[0]
        score = model.score(X_test, y_test)

        st.subheader(f"🌾 Predicted Yield: `{prediction:.2f} hg/ha`")
        st.caption(f"Model Accuracy: {score*100:.2f}%")
