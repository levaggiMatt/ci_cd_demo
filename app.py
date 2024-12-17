
#!pip install streamlit
import streamlit as st
import joblib
import pandas as pd
import os

# Function to detect the latest model version
def get_latest_model_version():
    model_dirs = [d for d in os.listdir() if os.path.isdir(d) and d.startswith("v")]
    latest_version = sorted(model_dirs, reverse=True)[0] if model_dirs else "v1"
    return latest_version

# Detect the latest model version and load it
latest_version = get_latest_model_version()
model_path = f"{latest_version}/model_{latest_version}.pkl"
model = joblib.load(model_path)

# Streamlit app title
st.title("Student Performance Prediction")

# Display the loaded model version
st.sidebar.write(f"Loaded Model Version: {latest_version}")

# Input form for school directors
st.header("Enter Student Attributes")

# Inputs for features
G1 = st.number_input("First Period Grade (G1)", min_value=0.0, max_value=20.0, step=0.1)
studytime = st.selectbox("Study Time (1: <2h, 2: 5h, 3: 5-10h, 4: >10h)", options=[1, 2, 3, 4])
famsup = st.radio("Family Support", options=["Yes", "No"])

# Encode famsup into two columns
famsup_yes = 1 if famsup == "Yes" else 0
famsup_no = 1 if famsup == "No" else 0

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame({
        "G1": [G1],
        "studytime": [studytime],
        "famsup_no": [famsup_no],
        "famsup_yes": [famsup_yes]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.success(f"Predicted Final Grade (G3): {prediction[0]:.2f}")
