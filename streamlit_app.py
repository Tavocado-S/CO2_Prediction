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
    "3. Feature Engineering",
    "4. Models Exploration",
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


# Hero function definition to set the background
def hero(image_path, title_text):
    import base64, pathlib
    encoded = base64.b64encode(pathlib.Path(image_path).read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        .hero {{
          position: relative;
          min-height: 220px;
          border-radius: 16px;
          overflow: hidden;
          margin-bottom: 25px;
          display: flex;              /* enable flexbox */
          align-items: center;        /* vertical center */
          justify-content: center;    /* horizontal center */
          text-align: center;         /* text center for multiline */
        }}
        .hero::before {{
          content: "";
          position: absolute;
          inset: 0;
          background-image: url("data:image/jpg;base64,{encoded}");
          background-size: cover;
          background-position: center;
          filter: brightness(0.55);
        }}
        .hero-content {{
          position: relative;
          z-index: 1;
          color: white;
          font-size: 60px;          /* bigger text */
          font-weight: 800;         /* extra bold */
          padding: 0 20px;
        }}
        </style>
        <div class="hero">
          <div class="hero-content">{title_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
# === Section 1: Project Introduction ===
if section == "1. Project Introduction":
    hero("images/contamination.webp", "Introduction")

    st.markdown('<div class="section"><div class="overlay">', unsafe_allow_html=True)

    st.markdown("### Motivation")
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
    hero("images/contamination.webp", "Data Overview")
    
    st.subheader("France 2013 Dataset")
    st.write("""
    Used for initial exploration. Contains historical vehicle registration data for France.
    Less complete and consistent than the EEA 2023 dataset.
    """)

    st.subheader("EEA 2023 Dataset")
    st.write("""
    This data set offers 
    - Modern Data Base and car characteristics
    - Measures are after “Dieselgate” scandal, based on standarized WLTP
    
    The car characteristics at disposal in the data base includes:
    - Vehicle mass (kg)
    - Engine capacity (cm3)
    - Engine power (KW)
    - Fuel type (petrol, diesel, hybrid, etc.)
    - Fuel mode (ICE, PHEV, etc.)
    - CO₂ emissions under WLTP standard
    - Country and manufacturer
    """)
    
    st.subheader("Exploration of Target variable and car characteristics")
    st.write(""" This first explorations gave us some clues. We concluded with statistical tests, that the numerical characteristics mentioned above have a positive correlation and have a significant relationship to CO2. The categorical values also have a significant relationship with CO2. 
    """)
    st.image("images/first_explo.png", caption="Traffic pollution", use_container_width=True)



# === Section 3: Features Engineering ===
elif section == "3. Feature Engineering":
    hero("images/contamination.webp", "Feature Engineering")
    st.markdown("""
    ## Characteristics Selection
    - Deleting non-essential identification features (maker, registration date, approval number, etc.)
    - Removing columns with too many missing values
    - Handling multicolinearities:
    - Fuel consumption vs CO₂
    - Mass vs Mass in running order
    """)
    

    st.markdown("""
    ## Filtering rows
    - Filtering Internal Combustion Cars
    - deleting zeros and outliers
    - GETTING UNIQUE CARS

    How did we get UNIQUE CARS?
    first we explored the variation (with Coefficient of Variation) of Mass, Engine Power, Engine Capacity and CO2 Emissions within Type - Variant - Version groups. If the there is a variation within the groups, the characteristic is used as subset in our drop duplicates code.
    - Mass and CO2 Emissions varied within the groups
    - Engine Capacity and Power did not varied.

    
    
    """)

    st.image("images/Unique_cars_ilustration.png", caption="Traffic pollution", use_container_width=True)

    st.markdown("""Our drop duplicates code to find unique cars then was:

    ```python
        df_unique = df_a.drop_duplicates(subset=
    ['Version', 'Variant', 'Type','CO2_Emissions_WLTP(g/km)',
    'Mass_in_Running_Order(kg)','Fuel_Type'])
    """)





# === Section 4: Model ===
elif section == "4. Models Exploration":
    hero("images/contamination.webp", "Models Exploration")
    import matplotlib.pyplot as plt
    from PIL import Image
    import os
    
    st.subheader("Tested Models")
    st.write("""
    Various machine learning models were tested to predict CO₂ emissions:
    - Linear Regression
    - Lasso Model
    - Random Forest
    - XGBoost
    - Deep Learning Neural Network
             
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

    from io import BytesIO

    MAX_WIDTH = 800  # max width for all images

    def st_plot(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        st.image(buf, use_container_width=False, width=MAX_WIDTH)
        plt.close(fig)

    st_plot(plot_metric("R² Score"))

    # === Collapsible sections for other metrics ===
    with st.expander("Show MSE Comparison"):
        st_plot(plot_metric("MSE"))

    with st.expander("Show MAE Comparison"):
        st_plot(plot_metric("MAE"))

    with st.expander("Show RMSE Comparison"):
        st_plot(plot_metric("RMSE"))


    st.success("Best model: Random Forest, with RMSE ≈ 11.33 and R² ≈ 0.94 on test set.")

    # === Section: Feature Importance & SHAP Values ===
    st.header("Feature Importance & SHAP Values Analysis")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    bar_plot_path = os.path.join(BASE_DIR, "plots", "shap_summary_bar.png")
    detailed_plot_path = os.path.join(BASE_DIR, "plots", "shap_summary_detailed.png")

    # Display images with max width to prevent oversized display
    MAX_WIDTH = 800

    if os.path.exists(bar_plot_path):
        st.subheader("Feature Importance")
        bar_img = Image.open(bar_plot_path)
        st.image(bar_img, use_container_width=False, width=MAX_WIDTH)
    else:
        st.error(f"Bar plot not found at {bar_plot_path}")

    if os.path.exists(detailed_plot_path):
        st.subheader("SHAP Values")
        detailed_img = Image.open(detailed_plot_path)
        st.image(detailed_img, use_container_width=False, width=MAX_WIDTH)
    else:
        st.error(f"Detailed SHAP plot not found at {detailed_plot_path}")

    st.markdown("""
    **Interpretation:**  
    The SHAP summary plots confirm that vehicle mass is by far the most influential feature in predicting CO₂ emissions, with a significantly higher average impact than all other inputs. Engine power and fuel type also contribute meaningfully, while engine capacity has a more moderate influence overall.  

    """)


# === Section 5: Prediction ===
elif section == "5. Prediction":
    hero("images/contamination.webp", "Prediction")

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
    st.markdown(
        """
        <style>
        div[data-testid="stSlider"] {
            max-width: 500px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.header("Vehicle Features")

    mass = st.slider("Mass in Running Order (kg)", min_value=200, max_value=5000, value=1500)
    engine_capacity = st.slider("Engine Capacity (cm³)", min_value=500, max_value=8000, value=1600)
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
