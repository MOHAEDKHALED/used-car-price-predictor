"""
Flask REST API for Used Car Price Predictor
Run with: python api.py
"""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
project_dir = Path(__file__).resolve().parent
model_path = project_dir / "car_price_model.pkl"
data_path = project_dir / "cars_cleaned.csv"

# ── Load model & data once at startup ─────────────────────────────────────────
model = joblib.load(model_path)
df = pd.read_csv(data_path)

REQUIRED_FIELDS = [
    "vehicle_age",
    "km_driven",
    "fuel_type",
    "transmission_type",
    "brand",
    "model",
]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Health-check / welcome endpoint."""
    return jsonify({
        "status": "ok",
        "message": "Used Car Price Predictor API is running 🚗",
        "endpoints": {
            "GET  /": "This help message",
            "GET  /health": "Health check",
            "GET  /options": "Valid dropdown values (brands, models, fuel types, transmission types)",
            "POST /predict": "Predict car selling price",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    """Simple liveness probe."""
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route("/options", methods=["GET"])
def options():
    """Return valid values for categorical fields."""
    brands = sorted(df["brand"].dropna().unique().tolist())
    fuel_types = sorted(df["fuel_type"].dropna().unique().tolist())
    transmission_types = sorted(df["transmission_type"].dropna().unique().tolist())

    # Optionally filter models by brand
    brand_filter = request.args.get("brand")
    if brand_filter:
        models = sorted(
            df.loc[df["brand"] == brand_filter, "model"].dropna().unique().tolist()
        )
    else:
        models = sorted(df["model"].dropna().unique().tolist())

    return jsonify({
        "brands": brands,
        "models": models,
        "fuel_types": fuel_types,
        "transmission_types": transmission_types,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the selling price of a used car.

    Expected JSON body:
    {
        "vehicle_age":       5,
        "km_driven":         50000,
        "fuel_type":         "Petrol",
        "transmission_type": "Manual",
        "brand":             "Maruti",
        "model":             "Swift"
    }

    Returns:
    {
        "predicted_price": 450000.0,
        "currency":        "INR",
        "input":           { ...echoed input... }
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON (Content-Type: application/json)"}), 415

    data = request.get_json()

    # ── Validate required fields ───────────────────────────────────────────────
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing,
            "required_fields": REQUIRED_FIELDS,
        }), 400

    # ── Type coercion & basic validation ──────────────────────────────────────
    try:
        vehicle_age = int(data["vehicle_age"])
        km_driven = int(data["km_driven"])
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"vehicle_age and km_driven must be integers: {exc}"}), 400

    if vehicle_age < 0 or km_driven < 0:
        return jsonify({"error": "vehicle_age and km_driven must be non-negative"}), 400

    # ── Build input DataFrame ─────────────────────────────────────────────────
    input_df = pd.DataFrame([{
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "fuel_type": str(data["fuel_type"]),
        "transmission_type": str(data["transmission_type"]),
        "brand": str(data["brand"]),
        "model": str(data["model"]),
    }])

    # ── Run model ─────────────────────────────────────────────────────────────
    try:
        predicted_price = float(model.predict(input_df)[0])
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify({
        "predicted_price": round(predicted_price, 2),
        "currency": "INR",
        "input": input_df.iloc[0].to_dict(),
    })


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚗  Starting Used Car Price Predictor API ...")
    print("    Docs → http://127.0.0.1:5000/")
    app.run(host="0.0.0.0", port=5000, debug=True)
