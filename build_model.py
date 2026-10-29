"""
build_model.py — Rebuilds car_price_model.pkl from cars_cleaned.csv
Run once with: python build_model.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import numpy as np

project_dir = Path(__file__).resolve().parent
data_path   = project_dir / "cars_cleaned.csv"
model_path  = project_dir / "car_price_model.pkl"

print("📂  Loading dataset …")
df = pd.read_csv(data_path)

FEATURES = ["vehicle_age", "km_driven", "fuel_type", "transmission_type", "brand", "model"]
TARGET   = "selling_price"

df = df[FEATURES + [TARGET]].dropna()
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cat_cols = ["fuel_type", "transmission_type", "brand", "model"]
num_cols = ["vehicle_age", "km_driven"]

preprocessor = ColumnTransformer([
    ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols),
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
])

print("🏋️  Training model …")
pipeline.fit(X_train, y_train)

print("📊  Evaluating …")
y_pred = pipeline.predict(X_test)
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"   R²   = {r2:.4f}")
print(f"   MAE  = {mae:,.2f}")
print(f"   RMSE = {rmse:,.2f}")

joblib.dump(pipeline, model_path)
print(f"✅  Model saved → {model_path}")
