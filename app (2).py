
import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Heart Attack Risk Predictor (Demo)")

st.markdown("""
> ⚠️ **Note:** This is a **demo** model using sample data.  
> Replace this with real medical data for production use.
""")

# Four input features (same as Iris dataset used in dummy model)
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk Detected (demo prediction)")
    else:
        st.success("✅ Low Risk Detected (demo prediction)")
