import streamlit as st
import pandas as pd
import joblib

# Load models and scalers
model_rf = joblib.load("models/best_model_Random_Forest.pkl")
model_lin = joblib.load("models/linear_regression_model.pkl")
scaler_lin = joblib.load("models/scaler_lin.pkl")
model_lasso = joblib.load("models/lasso_model.pkl")
scaler_lasso = joblib.load("models/scaler_lasso.pkl")
model_xgb = joblib.load("models/best_model_XGBoost.pkl")
model_dl = joblib.load("models/dl_model.pkl")
scaler_dl = joblib.load("models/scaler_dl.pkl")

# Initialize session state
if "main_pred" not in st.session_state:
    st.session_state.main_pred = None
if "other_preds" not in st.session_state:
    st.session_state.other_preds = None

# Feature inputs
st.header("Vehicle Features")

mass = st.number_input("Mass in Running Order (kg)", min_value=200, max_value=5000, value=1500)
engine_capacity = st.number_input("Engine Capacity (cm³)", min_value=500, max_value=8000, value=1600)
engine_power = st.slider("Engine Power (kW)", min_value=20, max_value=1200, value=85)
fuel_type = st.radio("Fuel Type", ["Petrol", "Diesel"])
fuel_type_petrol = 1 if fuel_type == "Petrol" else 0

# --- MAIN MODEL SECTION ---
st.subheader("Main Model Prediction (Random Forest)")
if st.button("Predict with Main Model"):
    input_df = pd.DataFrame({
        "Mass_in_Running_Order(kg)": [mass],
        "Engine_Capacity(cm3)": [engine_capacity],
        "Engine_Power(kW)": [engine_power],
        "Fuel_Type_petrol": [fuel_type_petrol]
    })
    pred_rf = model_rf.predict(input_df)
    st.session_state.main_pred = float(pred_rf[0])
    st.session_state.other_preds = None  # reset other models

if st.session_state.main_pred is not None:
    st.success(f"Random Forest Prediction: {st.session_state.main_pred:.2f} g/km CO2 emissions")

# --- OTHER MODELS SECTION ---
st.subheader("Predict with Other Models")
if st.button("Predict with Other Models"):
    preds = {}

    # Linear Regression
    lin_input = pd.DataFrame({
        "Mass_in_Running_Order(kg)": [mass],
        "Engine_Capacity(cm3)": [engine_capacity],
        "Engine_Power(kW)": [engine_power]
    })
    lin_input_scaled = scaler_lin.transform(lin_input)
    lin_input_scaled_df = pd.DataFrame(lin_input_scaled, columns=scaler_lin.feature_names_in_)
    lin_input_scaled_df["Fuel_Type_petrol"] = fuel_type_petrol
    preds["Linear Regression"] = float(model_lin.predict(lin_input_scaled_df)[0])

    # Lasso
    lasso_input_scaled = scaler_lasso.transform(lin_input)
    lasso_input_scaled_df = pd.DataFrame(lasso_input_scaled, columns=scaler_lasso.feature_names_in_)
    lasso_input_scaled_df["Fuel_Type_petrol"] = fuel_type_petrol
    preds["Lasso"] = float(model_lasso.predict(lasso_input_scaled_df)[0])

    # XGBoost
    preds["XGBoost"] = float(model_xgb.predict(pd.DataFrame({
        "Mass_in_Running_Order(kg)": [mass],
        "Engine_Capacity(cm3)": [engine_capacity],
        "Engine_Power(kW)": [engine_power],
        "Fuel_Type_petrol": [fuel_type_petrol]
    }))[0])

    # Deep Learning
    dl_input = pd.DataFrame({
        "Mass_in_Running_Order(kg)": [mass],
        "Engine_Capacity(cm3)": [engine_capacity],
        "Engine_Power(kW)": [engine_power],
        "Fuel_Type_petrol": [fuel_type_petrol]
    })
    dl_input_scaled = scaler_dl.transform(dl_input)
    preds["Deep Learning"] = float(model_dl.predict(dl_input_scaled)[0])

    st.session_state.other_preds = preds

if st.session_state.other_preds is not None:
    for model_name, pred in st.session_state.other_preds.items():
        st.info(f"{model_name} Prediction: {pred:.2f} g/km CO2 emissions")
