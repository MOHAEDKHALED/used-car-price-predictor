from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ── File paths ─────────────────────────────────────────────────────────────────
project_directory = Path(__file__).resolve().parent
model_path = project_directory / "car_price_model.pkl"
data_path = project_directory / "cars_cleaned.csv"

FEATURES = ["vehicle_age", "km_driven", "fuel_type",
            "transmission_type", "brand", "model"]
TARGET = "selling_price"


# ── Build model from CSV (used when .pkl is missing or invalid) ────────────────
def build_model(df: pd.DataFrame):
    """Train a Random Forest pipeline and save it to disk."""
    train_df = df[FEATURES + [TARGET]].dropna()
    X, y = train_df[FEATURES], train_df[TARGET]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_cols = ["fuel_type", "transmission_type", "brand", "model"]
    preprocessor = ColumnTransformer(
        [("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, model_path)
    return pipeline


# ── Load data used for dropdown options ────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(data_path)


# ── Load or build the model ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_build_model(df):
    """Load the .pkl if it exists and is valid; otherwise train from scratch."""
    if model_path.exists():
        try:
            loaded = joblib.load(model_path)
            # Quick sanity-check: try a dummy prediction
            dummy = pd.DataFrame([{
                "vehicle_age": 5, "km_driven": 50000,
                "fuel_type": "Petrol", "transmission_type": "Manual",
                "brand": "Maruti", "model": "Swift",
            }])
            loaded.predict(dummy)
            return loaded, False          # (model, was_built_now)
        except Exception:
            pass                          # Fall through to rebuild
    return build_model(df), True          # Train and return


# ── Bootstrap ──────────────────────────────────────────────────────────────────
try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ **cars_cleaned.csv** not found. Make sure it is in the same folder as app.py.")
    st.stop()

with st.spinner("🔧 Loading model — this may take a minute on first run…"):
    model, just_built = load_or_build_model(df)

if just_built:
    st.toast("✅ Model trained successfully!", icon="🎉")


# ── Page heading ───────────────────────────────────────────────────────────────
st.title("🚗 Used Car Price Predictor")
st.write(
    "Enter the vehicle information below to receive an "
    "estimated selling price."
)


# ── Dropdown options ───────────────────────────────────────────────────────────
brand_options       = sorted(df["brand"].dropna().unique())
fuel_options        = sorted(df["fuel_type"].dropna().unique())
transmission_options = sorted(df["transmission_type"].dropna().unique())

# Brand selection
brand = st.selectbox("Brand", options=brand_options)

# Models filtered by brand
model_options = sorted(
    df.loc[df["brand"] == brand, "model"].dropna().unique()
)
car_model = st.selectbox("Model", options=model_options)


# ── Prediction form ────────────────────────────────────────────────────────────
with st.form("car_prediction_form"):
    first_column, second_column = st.columns(2)

    with first_column:
        vehicle_age = st.number_input(
            "Vehicle age (years)", min_value=0, max_value=50, value=5, step=1
        )
        fuel_type = st.selectbox("Fuel type", options=fuel_options)

    with second_column:
        km_driven = st.number_input(
            "Kilometres driven", min_value=0, max_value=5_000_000,
            value=50_000, step=5_000
        )
        transmission_type = st.selectbox(
            "Transmission type", options=transmission_options
        )

    predict_button = st.form_submit_button(
        "Predict selling price", use_container_width=True
    )


# ── Make prediction ────────────────────────────────────────────────────────────
if predict_button:
    new_car = pd.DataFrame([{
        "vehicle_age":       vehicle_age,
        "km_driven":         km_driven,
        "fuel_type":         fuel_type,
        "transmission_type": transmission_type,
        "brand":             brand,
        "model":             car_model,
    }])

    predicted_price = model.predict(new_car)[0]

    st.success("Prediction completed")
    st.metric(
        label="Estimated selling price",
        value=f"₹{predicted_price:,.0f}"
    )
    st.caption(
        "This is a machine-learning estimate and may differ "
        "from the actual market price."
    )


# ── About the model ────────────────────────────────────────────────────────────
with st.expander("About the model"):
    st.write(
        """
        The prediction is generated using a Random Forest model.

        **Test results:**

        - R²: 0.8994
        - MAE: 111,668.45
        - RMSE: 275,214.64

        The model uses vehicle age, kilometres driven, fuel type,
        transmission type, brand and model.
        """
    )