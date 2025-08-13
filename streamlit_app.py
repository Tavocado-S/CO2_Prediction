import streamlit as st
import pandas as pd
import joblib
import base64
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="CO₂ Emissions Dashboard", layout="wide")

# Sidebar menu
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "1. Project Introduction",
    "2. Data Overview",
    "3. Features Selection",
    "4. Model",
    "5. Prediction"
])

# === Helper function for background image ===
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    css = f"""
    <style>
    .section {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        padding: 2rem;
        border-radius: 10px;
    }}
    .overlay {{
        background-color: rgba(0, 0, 0, 0.6);
        padding: 2rem;
        border-radius: 10px;
        color: white;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# === Section 1: Project Introduction ===
if section == "1. Project Introduction":
    set_background("streamlit_support/foto1.jpg")

    st.markdown('<div class="section"><div class="overlay">', unsafe_allow_html=True)

    st.title("Project Introduction, Context and Objective")

    st.markdown("### Introduction")
    st.write("""
    The transportation sector is a major contributor to greenhouse gas emissions worldwide.
    Passenger cars, in particular, release significant amounts of CO₂.
    This project uses real data to investigate and model these emissions.
    """)

    st.markdown("### Context")
    st.write("""
    The dataset is provided by the **European Environment Agency (EEA)**.
    It contains technical specifications for all new vehicles registered in the EU in 2023:
    mass, engine size, power, fuel type, consumption, and CO₂ output.
    """)

    st.markdown("### Objective")
    st.write("""
    - Identify technical features that influence CO₂ emissions.
    - Model the relationship between those features and CO₂ output (WLTP standard).
    - Enable predictive tools to estimate emissions before testing.
    """)

    st.markdown('</div></div>', unsafe_allow_html=True)

# === Section 2: Data Overview ===
elif section == "2. Data Overview":
    st.title("Dataset Overview")

    st.subheader("🇫🇷 France 2013 Dataset")
    st.write("""
    Used for initial exploration. Contains historical vehicle registration data for France.
    Less complete and consistent than the EEA 2023 dataset.
    """)

    st.subheader("🇪🇺 EEA 2023 Dataset")
    st.write("""
    Main dataset for this project. Includes:
    - Vehicle mass (kg)
    - Engine capacity (cm3)
    - Engine power (KW)
    - Fuel type (petrol, diesel, hybrid, etc.)
    - Fuel mode (ICE, PHEV, etc.)
    - CO₂ emissions under WLTP standard
    - Country and manufacturer

    The data was cleaned, translated, and filtered to focus on ICE vehicles.
    """)

# === Section 3: Features Selection ===
elif section == "3. Features Selection":
    st.title("Feature Selection")

    st.write("""
    Several steps were taken to select relevant features for modeling:
    - Removed columns with >30% missing values or low interpretability
    - Excluded highly correlated columns (e.g. fuel consumption & mass)
    - Focused on measurable, interpretable technical variables:
        - Mass
        - Engine power
        - Engine capacity
        - Fuel type
    """)

    st.info("Correlation analysis and distribution plots were used to assess predictive power.")

# === Section 4: Model ===
elif section == "4. Model":
    import matplotlib.pyplot as plt

    st.title("Modeling")
    
    st.subheader("Model Types")
    st.write("""
    Various machine learning models were tested to predict CO₂ emissions:
    - Linear Regression
    - Lasso Model
    - Random Forest
    - XGBoost
    - Deeplearning Neural Network
             
    Then we optimized our models (Lasso, Random Forest, XGBoost) with hyperparameter tuning.
    
    We empirically tried different architectures and parameters for the neural network model to reach the best performance.
    """)

    st.subheader("Model Evaluation")        
    st.write("""
    Evaluation was done using:
    - Train/test split (80/20)
    - R², RMSE, MSE and MAE scores
    - Cross-validation check for robustness and potential overfitting.
    """)

    # === Data for charts ===
    data = {
        "Metric": ["MSE", "MAE", "RMSE", "R² Score"],
        "Linear Regression": [373.25, 12.02, 19.32, 0.811],
        "Lasso Model": [373.32, 12.02, 19.32, 0.811],
        "Random Forest": [128.42, 5.05, 11.33, 0.935],
        "XGBoost": [144.96, 5.59, 12.04, 0.927],
        "DL model": [245.97, 9.04, 15.68, 0.876],
    }
    df = pd.DataFrame(data)
    df.set_index("Metric", inplace=True)
    models = df.columns

    model_colors = {
        "Linear Regression": '#1f77b4',
        "Lasso Model": '#ff7f0e',
        "Random Forest": '#2ca02c',
        "XGBoost": '#d62728',
        "DL model": '#9467bd'
    }

    # === Function to plot chart ===
    def plot_metric(metric):
        fig, ax = plt.subplots(figsize=(8, 5))
        values = df.loc[metric]
        bars = ax.bar(models, values, color=[model_colors[m] for m in models])

        # Highlight best model
        if metric != "R² Score":
            best_model = values.idxmin()
        else:
            best_model = values.idxmax()

        best_idx = models.get_loc(best_model)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)

        # Value labels
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}",
                    ha='center', va='bottom', fontsize=10)

        ax.set_title(f'{metric} Comparison Across Models')
        ax.set_ylabel(metric)
        ax.set_xticklabels(models, rotation=30)
        plt.tight_layout()
        return fig

    # === Always visible: R² Score ===
    st.subheader("Model Comparisons & Results")
    st.pyplot(plot_metric("R² Score"))

    # === Collapsible sections for other metrics ===
    with st.expander("Show MSE Comparison"):
        st.pyplot(plot_metric("MSE"))

    with st.expander("Show MAE Comparison"):
        st.pyplot(plot_metric("MAE"))

    with st.expander("Show RMSE Comparison"):
        st.pyplot(plot_metric("RMSE"))

    st.success("Best model: Random Forest, with RMSE ≈ 11.33 and R² ≈ 0.94 on test set.")

    # === Section: Feature Importance & SHAP Values ===
    st.subheader("Feature Importance & SHAP Values Analysis")

    # Load model and test data
    model_rf = joblib.load("models/best_model_Random_Forest.pkl")
    X_test = joblib.load("../data/X_test.pkl") 

    # Compute SHAP values
    explainer = shap.TreeExplainer(model_rf)
    shap_values = explainer.shap_values(X_test)

    # --- SHAP Bar Plot (Feature Importance) ---
    st.markdown("**Feature Importance (Average Impact)**")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig1)

    # --- SHAP Beeswarm Plot (Feature Impact & Direction) ---
    st.markdown("**Detailed Feature Impact (Beeswarm Plot)**")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig2)

    # Interpretation
    st.markdown("""
    **Interpretation:**  
    The SHAP summary plots confirm that vehicle **mass** is by far the most influential feature in predicting CO₂ emissions, with a significantly higher average impact than all other inputs.  
    **Engine power** and **fuel type** also contribute meaningfully, while **engine capacity** has a more moderate influence overall.  

    The color-coded beeswarm plot shows that higher values for mass and engine power strongly increase predicted emissions, whereas petrol fuel type typically lowers them compared to diesel.
    """)


# === Section 5: Prediction ===
elif section == "5. Prediction":
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
        st.success(f"Random Forest Prediction: {st.session_state.main_pred:.2f} g/km CO₂ emissions")

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
            st.info(f"{model_name} Prediction: {pred:.2f} g/km CO₂ emissions")
