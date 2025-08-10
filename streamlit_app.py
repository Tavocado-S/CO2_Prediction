import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load models and scalers ---
rf_model = joblib.load("models/best_model_Random_Forest.pkl")
xgb_model = joblib.load("models/best_model_XGBoost.pkl")

lin_model = joblib.load("models/linear_regression_model.pkl")
scaler_lin = joblib.load("models/scaler_lin.pkl")

lasso_model = joblib.load("models/lasso_model.pkl")
scaler_lasso = joblib.load("models/scaler_lasso.pkl")

dl_model = joblib.load("models/dl_model.pkl")
scaler_dl = joblib.load("models/scaler_dl.pkl")

st.title("CO2 Emission Prediction App")

# --- Feature inputs on main page ---
mass = st.number_input("Mass in Running Order (kg)", min_value=200, max_value=5000, value=1500)
engine_capacity = st.number_input("Engine Capacity (cm³)", min_value=500, max_value=8000, value=1600)
engine_power = st.number_input("Engine Power (kW)", min_value=20, max_value=1200, value=85)
fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel"])

fuel_type_petrol = 1 if fuel_type == "Petrol" else 0

input_dict = {
    'Mass_in_Running_Order(kg)': [mass],
    'Engine_Capacity(cm3)': [engine_capacity],
    'Engine_Power(kW)': [engine_power],
    'Fuel_Type_petrol': [fuel_type_petrol]
}
input_df = pd.DataFrame(input_dict)

# --- Helper function to scale inputs only on scaler's expected features ---
def scale_input(input_df, scaler):
    features = list(scaler.feature_names_in_)
    df_scaled = input_df.copy()
    df_scaled[features] = scaler.transform(input_df[features])
    return df_scaled

# --- MAIN PREDICTION with Random Forest ---
if st.button("Predict with Random Forest (Main Model)"):
    pred_rf = rf_model.predict(input_df)
    pred_rf_val = pred_rf[0] if hasattr(pred_rf, "__len__") else pred_rf
    st.success(f"Random Forest Model Prediction: {pred_rf_val:.2f} g/km CO2 emissions")

# --- OTHER MODELS SECTION ---
st.markdown("---")
st.subheader("Predict with Other Models")

if st.button("Predict with Linear Regression, Lasso, XGBoost, and Deep Learning"):
    lin_scaled = scale_input(input_df, scaler_lin)
    pred_lin = lin_model.predict(lin_scaled)
    pred_lin_val = pred_lin[0] if hasattr(pred_lin, "__len__") else pred_lin

    lasso_scaled = scale_input(input_df, scaler_lasso)
    pred_lasso = lasso_model.predict(lasso_scaled)
    pred_lasso_val = pred_lasso[0] if hasattr(pred_lasso, "__len__") else pred_lasso

    pred_xgb = xgb_model.predict(input_df)
    pred_xgb_val = pred_xgb[0] if hasattr(pred_xgb, "__len__") else pred_xgb

    dl_scaled = scale_input(input_df, scaler_dl)
    pred_dl = dl_model.predict(dl_scaled)
    pred_dl_val = pred_dl[0] if hasattr(pred_dl, "__len__") else pred_dl

    st.write(f"Linear Regression Prediction: {pred_lin_val:.2f} g/km CO2 emissions")
    st.write(f"Lasso Prediction: {pred_lasso_val:.2f} g/km CO2 emissions")
    st.write(f"XGBoost Prediction: {pred_xgb_val:.2f} g/km CO2 emissions")
    st.write(f"Deep Learning Prediction: {pred_dl_val.item():.2f} g/km CO2 emissions")

# --- Notes ---
st.markdown("""
---
**Note:**

- Inputs are now on the main page (not sidebar) to allow sidebar usage as navigation.
- Predictions might return arrays, so values are extracted before formatting.
""")
