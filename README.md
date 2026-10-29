# Used Car Price Predictor

A Streamlit app that predicts the selling price of a used car from vehicle details such as brand, model, age, kilometres driven, fuel type, and transmission type.

## Project Overview

This project uses a trained Random Forest regression pipeline saved as `car_price_model.pkl`. The app loads the trained model and the cleaned dataset, then provides dropdowns and number inputs for making price predictions through a simple web interface.

## Files

- `app.py`: Streamlit web app.
- `cars_analysis.ipynb`: Notebook used for data analysis, preprocessing, training, and evaluation.
- `cars.csv`: Original dataset.
- `cars_cleaned.csv`: Cleaned dataset used by the app for dropdown values.
- `car_price_model.pkl`: Saved machine learning pipeline.
- `requirements.txt`: Python dependencies with pinned versions.

## Model Features

The model uses these input columns:

- `vehicle_age`
- `km_driven`
- `fuel_type`
- `transmission_type`
- `brand`
- `model`

The target column is:

- `selling_price`

## Model Performance

The app reports the following test results:

- R2: 0.9023
- MAE: 110,675.52
- RMSE: 271,129.42

## Environment

Use the project virtual environment at:

```bash
/Users/najwaaouaj/cars_project/.venv
```

This environment uses Python 3.11 and the pinned package versions from `requirements.txt`. This matters because the saved model was trained with `scikit-learn==1.6.1`; using a newer scikit-learn version can make the pickle fail to load.

Verified environment:

```text
Python 3.11.14
streamlit 1.64.0
pandas 2.2.3
joblib 1.4.2
scikit-learn 1.6.1
```

## Setup

If the virtual environment already exists, activate it:

```bash
cd /Users/najwaaouaj/cars_project
source .venv/bin/activate
```

If you need to recreate it:

```bash
cd /Users/najwaaouaj/cars_project
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run The App

From the project folder:

```bash
cd /Users/najwaaouaj/cars_project
.venv/bin/streamlit run app.py
```

If the default Streamlit port `8501` is busy, run it on another port:

```bash
.venv/bin/streamlit run app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

## Quick Verification

Run this command to confirm that the app environment can load the dataset and model:

```bash
cd /Users/najwaaouaj/cars_project
.venv/bin/python - <<'PY'
from pathlib import Path
import joblib
import pandas as pd
import sklearn

project = Path(".")
df = pd.read_csv(project / "cars_cleaned.csv")
model = joblib.load(project / "car_price_model.pkl")
sample = df[["vehicle_age", "km_driven", "fuel_type", "transmission_type", "brand", "model"]].head(1)
prediction = model.predict(sample)[0]

print("scikit-learn:", sklearn.__version__)
print("dataset shape:", df.shape)
print("model type:", type(model).__name__)
print("sample prediction:", round(float(prediction), 2))
PY
```

Expected result includes:

```text
scikit-learn: 1.6.1
dataset shape: (15411, 12)
model type: Pipeline
```

## Notes

- Do not run this project with the system `python3` unless it points to the same `.venv`.
- The old mismatched Python 3.14 environment was moved to `.venv-py314-mismatch`.
- The predicted price is an estimate and may differ from real market prices.
