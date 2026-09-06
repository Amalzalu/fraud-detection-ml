import streamlit as st
import pandas as pd
import joblib
import os

# Build an absolute path based on THIS script's own location, not the current working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'fraud_detection_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'models', 'scaler.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


feature_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

st.title("💳 Fraud Detection System")
st.write("Enter transaction details below to check if it's likely fraudulent.")
st.info("Note: V1-V28 are anonymized (PCA-transformed) features from the original dataset. "
        "For this demo, they default to typical values — adjust if you have specific values to test.")

# User inputs for the two interpretable features
time_input = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=50000.0)
amount_input = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)

# Default V1-V28 values (using a realistic sample - e.g., mean values from training data)
with st.expander("Advanced: Adjust anonymized features (V1-V28)"):
    v_values = {}
    for i in range(1, 29):
        v_values[f'V{i}'] = st.slider(f'V{i}', -20.0, 20.0, 0.0, step=0.1)

if st.button("Check Transaction"):
    transaction = {'Time': time_input, **v_values, 'Amount': amount_input}
    
    input_df = pd.DataFrame([transaction])[feature_order]
    input_scaled = scaler.transform(input_df)
    
    fraud_probability = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]
    
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ Prediction: FRAUDULENT")
    else:
        st.success(f"✅ Prediction: LEGITIMATE")
    
    st.metric("Fraud Probability", f"{fraud_probability * 100:.2f}%")