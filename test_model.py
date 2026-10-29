"""
test_model.py — Direct unit tests for the car_price_model.pkl
Run with: python test_model.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
project_dir = Path(__file__).resolve().parent
model_path = project_dir / "car_price_model.pkl"
data_path = project_dir / "cars_cleaned.csv"

PASS = "\033[92m✔ PASS\033[0m"
FAIL = "\033[91m✘ FAIL\033[0m"
SECTION = "\033[94m{}\033[0m"

results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"  {status}  {name}"
    if detail:
        msg += f"\n         → {detail}"
    print(msg)
    results.append((name, condition))


# ══════════════════════════════════════════════════════════════════════════════
print(SECTION.format("\n══════════════════════════════════════════"))
print(SECTION.format("   🚗  Used Car Price Predictor — Model Tests"))
print(SECTION.format("══════════════════════════════════════════\n"))

# ── 1. Load model ─────────────────────────────────────────────────────────────
print(SECTION.format("[ 1 ]  Model Loading"))

try:
    model = joblib.load(model_path)
    check("Model file loads without error", True, str(model_path))
except Exception as exc:
    check("Model file loads without error", False, str(exc))
    print("\n⛔  Cannot continue — model failed to load.")
    sys.exit(1)

check("Model object is not None", model is not None)

# ── 2. Load data ──────────────────────────────────────────────────────────────
print(SECTION.format("\n[ 2 ]  Dataset Loading"))

try:
    df = pd.read_csv(data_path)
    check("Dataset loads without error", True, str(data_path))
except Exception as exc:
    check("Dataset loads without error", False, str(exc))
    df = None

if df is not None:
    expected_columns = ["vehicle_age", "km_driven", "fuel_type",
                        "transmission_type", "brand", "model"]
    missing_cols = [c for c in expected_columns if c not in df.columns]
    check(
        "All required columns present",
        len(missing_cols) == 0,
        f"Missing: {missing_cols}" if missing_cols else f"Columns OK: {expected_columns}",
    )
    check("Dataset has rows", len(df) > 0, f"{len(df):,} rows loaded")

# ── 3. Single prediction ──────────────────────────────────────────────────────
print(SECTION.format("\n[ 3 ]  Single Prediction"))

SAMPLE = {
    "vehicle_age": 5,
    "km_driven": 50_000,
    "fuel_type": "Petrol",
    "transmission_type": "Manual",
    "brand": "Maruti",
    "model": "Swift",
}

try:
    sample_df = pd.DataFrame([SAMPLE])
    prediction = model.predict(sample_df)

    check("predict() returns a result", True)
    check("Result has exactly one element", len(prediction) == 1,
          f"Got {len(prediction)} element(s)")

    price = float(prediction[0])
    check("Predicted price is a finite number", np.isfinite(price),
          f"Value = {price:,.2f}")
    check("Predicted price is positive", price > 0,
          f"₹{price:,.0f}")

    # Sanity range: cars in the dataset range roughly ₹20k – ₹10 cr
    check("Predicted price is in plausible range (₹10,000 – ₹10,00,00,000)",
          10_000 <= price <= 100_000_000,
          f"₹{price:,.0f}")

except Exception as exc:
    check("Single prediction runs without error", False, str(exc))

# ── 4. Batch prediction ───────────────────────────────────────────────────────
print(SECTION.format("\n[ 4 ]  Batch Predictions"))

BATCH_SAMPLES = [
    {"vehicle_age": 1,  "km_driven": 10_000,  "fuel_type": "Diesel",   "transmission_type": "Automatic", "brand": "Hyundai", "model": "Creta"},
    {"vehicle_age": 10, "km_driven": 200_000, "fuel_type": "Petrol",   "transmission_type": "Manual",    "brand": "Maruti",  "model": "Alto"},
    {"vehicle_age": 3,  "km_driven": 30_000,  "fuel_type": "Electric", "transmission_type": "Automatic", "brand": "Tata",    "model": "Nexon"},
]

try:
    batch_df = pd.DataFrame(BATCH_SAMPLES)
    batch_preds = model.predict(batch_df)

    check("Batch predict() runs without error", True)
    check("Returns correct number of predictions", len(batch_preds) == len(BATCH_SAMPLES),
          f"Expected {len(BATCH_SAMPLES)}, got {len(batch_preds)}")

    all_positive = all(p > 0 for p in batch_preds)
    all_finite = all(np.isfinite(p) for p in batch_preds)
    check("All batch predictions are positive", all_positive)
    check("All batch predictions are finite", all_finite)

    for i, (sample, pred) in enumerate(zip(BATCH_SAMPLES, batch_preds)):
        print(f"         Sample {i+1} — {sample['brand']} {sample['model']} "
              f"({sample['vehicle_age']}yr, {sample['km_driven']:,}km) → ₹{pred:,.0f}")

except Exception as exc:
    check("Batch prediction runs without error", False, str(exc))

# ── 5. Edge cases ─────────────────────────────────────────────────────────────
print(SECTION.format("\n[ 5 ]  Edge Cases"))

EDGE_CASES = [
    ("Brand-new car",       {"vehicle_age": 0,  "km_driven": 0,         "fuel_type": "Petrol", "transmission_type": "Manual",    "brand": "Maruti",  "model": "Swift"}),
    ("Very old high km",    {"vehicle_age": 30, "km_driven": 999_999,   "fuel_type": "Diesel", "transmission_type": "Manual",    "brand": "Maruti",  "model": "Eeco"}),
    ("Luxury automatic",    {"vehicle_age": 2,  "km_driven": 20_000,    "fuel_type": "Petrol", "transmission_type": "Automatic", "brand": "BMW",     "model": "5 Series"}),
]

for label, sample in EDGE_CASES:
    try:
        pred = float(model.predict(pd.DataFrame([sample]))[0])
        check(f"Edge case: '{label}' → ₹{pred:,.0f}", np.isfinite(pred) and pred > 0)
    except Exception as exc:
        check(f"Edge case: '{label}'", False, str(exc))

# ── 6. Model attributes ───────────────────────────────────────────────────────
print(SECTION.format("\n[ 6 ]  Model Attributes"))

check("Model has predict method",    hasattr(model, "predict"))
check("Model has fit method",        hasattr(model, "fit"))

# Try to find the final estimator (works for pipelines or plain models)
final_estimator = model
if hasattr(model, "steps"):          # sklearn Pipeline
    final_estimator = model.steps[-1][1]
elif hasattr(model, "named_steps"):  # also Pipeline
    final_estimator = list(model.named_steps.values())[-1]

estimator_name = type(final_estimator).__name__
check("Model type reported",         True, estimator_name)

if hasattr(final_estimator, "n_estimators"):
    check("n_estimators attribute readable", True,
          f"{final_estimator.n_estimators} trees")

# ── Summary ───────────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print(SECTION.format("\n══════════════════════════════════════════"))
print(f"   Result: {passed}/{total} tests passed", end="")
if failed:
    print(f"  |  \033[91m{failed} failed\033[0m")
else:
    print("  🎉")
print(SECTION.format("══════════════════════════════════════════\n"))

sys.exit(0 if failed == 0 else 1)
