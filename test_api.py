"""
test_api.py — Integration tests for the Flask API (api.py)

Usage:
  1. Start the API in a separate terminal:
       python api.py

  2. Then run this test script:
       python test_api.py

  Or run both in one command (Windows):
       start python api.py && timeout /t 3 && python test_api.py
"""

import sys
import time
import json

import requests

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = "http://127.0.0.1:5000"
TIMEOUT  = 10  # seconds per request

PASS    = "\033[92m✔ PASS\033[0m"
FAIL    = "\033[91m✘ FAIL\033[0m"
SKIP    = "\033[93m⚠ SKIP\033[0m"
SECTION = "\033[94m{}\033[0m"

results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"  {status}  {name}"
    if detail:
        msg += f"\n         → {detail}"
    print(msg)
    results.append((name, condition))
    return condition


def get(path: str, **kwargs):
    return requests.get(f"{BASE_URL}{path}", timeout=TIMEOUT, **kwargs)


def post(path: str, json_body=None, **kwargs):
    return requests.post(f"{BASE_URL}{path}", json=json_body, timeout=TIMEOUT, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
print(SECTION.format("\n══════════════════════════════════════════"))
print(SECTION.format("   🚗  Used Car Price Predictor — API Tests"))
print(SECTION.format(f"   Target: {BASE_URL}"))
print(SECTION.format("══════════════════════════════════════════\n"))

# ── 0. Connectivity check ─────────────────────────────────────────────────────
print(SECTION.format("[ 0 ]  Connectivity"))

try:
    r = get("/")
    api_reachable = True
    check("API is reachable", True, f"HTTP {r.status_code}")
except requests.exceptions.ConnectionError:
    check("API is reachable", False,
          f"Could not connect to {BASE_URL}. Is 'python api.py' running?")
    print("\n⛔  Cannot continue — API is not running.")
    sys.exit(1)

# ── 1. GET / ──────────────────────────────────────────────────────────────────
print(SECTION.format("\n[ 1 ]  GET /  (index)"))

r = get("/")
check("Status 200", r.status_code == 200, f"Got {r.status_code}")
try:
    body = r.json()
    check("Response is valid JSON", True)
    check("'status' == 'ok'", body.get("status") == "ok", str(body.get("status")))
    check("'endpoints' key present", "endpoints" in body)
except ValueError as exc:
    check("Response is valid JSON", False, str(exc))

# ── 2. GET /health ────────────────────────────────────────────────────────────
print(SECTION.format("\n[ 2 ]  GET /health"))

r = get("/health")
check("Status 200", r.status_code == 200, f"Got {r.status_code}")
body = r.json()
check("'status' == 'healthy'", body.get("status") == "healthy")
check("'model_loaded' is True", body.get("model_loaded") is True)

# ── 3. GET /options ───────────────────────────────────────────────────────────
print(SECTION.format("\n[ 3 ]  GET /options"))

r = get("/options")
check("Status 200", r.status_code == 200, f"Got {r.status_code}")
body = r.json()
for key in ("brands", "models", "fuel_types", "transmission_types"):
    ok = key in body and isinstance(body[key], list) and len(body[key]) > 0
    check(f"'{key}' is non-empty list", ok,
          f"{len(body.get(key, []))} items" if key in body else "key missing")

print(SECTION.format("\n[ 3b ]  GET /options?brand=Maruti"))

r = get("/options", params={"brand": "Maruti"})
check("Status 200 with brand filter", r.status_code == 200)
body = r.json()
check("Filtered models list is non-empty", len(body.get("models", [])) > 0,
      f"{len(body.get('models', []))} models for Maruti")

# ── 4. POST /predict — valid payload ─────────────────────────────────────────
print(SECTION.format("\n[ 4 ]  POST /predict — valid payload"))

VALID_PAYLOAD = {
    "vehicle_age": 5,
    "km_driven": 50000,
    "fuel_type": "Petrol",
    "transmission_type": "Manual",
    "brand": "Maruti",
    "model": "Swift",
}

r = post("/predict", json_body=VALID_PAYLOAD)
check("Status 200", r.status_code == 200, f"Got {r.status_code}")
body = r.json()
check("Response is valid JSON", True)
check("'predicted_price' key present", "predicted_price" in body)
check("'currency' == 'INR'",           body.get("currency") == "INR")
check("'input' echoed back",           "input" in body)

if "predicted_price" in body:
    price = body["predicted_price"]
    check("Predicted price is a number", isinstance(price, (int, float)),
          f"Type: {type(price).__name__}")
    check("Predicted price is positive", price > 0, f"₹{price:,.0f}")
    check("Predicted price is plausible (₹10k – ₹10cr)",
          10_000 <= price <= 100_000_000, f"₹{price:,.0f}")

# ── 5. POST /predict — multiple valid samples ─────────────────────────────────
print(SECTION.format("\n[ 5 ]  POST /predict — multiple valid samples"))

SAMPLES = [
    ("Diesel Hyundai Creta (1yr)",  {"vehicle_age": 1,  "km_driven": 10000,  "fuel_type": "Diesel",   "transmission_type": "Automatic", "brand": "Hyundai", "model": "Creta"}),
    ("Old Maruti Alto (10yr)",      {"vehicle_age": 10, "km_driven": 200000, "fuel_type": "Petrol",   "transmission_type": "Manual",    "brand": "Maruti",  "model": "Alto"}),
    ("Electric Tata Nexon (3yr)",   {"vehicle_age": 3,  "km_driven": 30000,  "fuel_type": "Electric", "transmission_type": "Automatic", "brand": "Tata",    "model": "Nexon"}),
]

for label, payload in SAMPLES:
    r = post("/predict", json_body=payload)
    ok = r.status_code == 200
    price = r.json().get("predicted_price", "N/A") if ok else "ERROR"
    price_fmt = f"₹{price:,.0f}" if isinstance(price, (int, float)) else price
    check(f"Sample: {label}  →  {price_fmt}", ok, f"HTTP {r.status_code}")

# ── 6. POST /predict — error cases ───────────────────────────────────────────
print(SECTION.format("\n[ 6 ]  POST /predict — error / edge-case handling"))

# 6a. Missing fields
r = post("/predict", json_body={"vehicle_age": 5, "km_driven": 10000})
check("Missing fields → 400 Bad Request", r.status_code == 400,
      f"Got {r.status_code}")
body = r.json()
check("Error response has 'missing_fields' key", "missing_fields" in body,
      str(body))

# 6b. Wrong Content-Type (send plain string instead of JSON)
r = requests.post(
    f"{BASE_URL}/predict",
    data="not json",
    headers={"Content-Type": "text/plain"},
    timeout=TIMEOUT,
)
check("Non-JSON Content-Type → 415", r.status_code == 415,
      f"Got {r.status_code}")

# 6c. Negative km_driven
r = post("/predict", json_body={**VALID_PAYLOAD, "km_driven": -1})
check("Negative km_driven → 400", r.status_code == 400,
      f"Got {r.status_code}")

# 6d. Negative vehicle_age
r = post("/predict", json_body={**VALID_PAYLOAD, "vehicle_age": -5})
check("Negative vehicle_age → 400", r.status_code == 400,
      f"Got {r.status_code}")

# 6e. Non-numeric vehicle_age
r = post("/predict", json_body={**VALID_PAYLOAD, "vehicle_age": "five"})
check("Non-numeric vehicle_age → 400", r.status_code == 400,
      f"Got {r.status_code}")

# ── 7. Response time ──────────────────────────────────────────────────────────
print(SECTION.format("\n[ 7 ]  Performance — Response Time"))

N_CALLS = 5
times = []
for _ in range(N_CALLS):
    t0 = time.time()
    post("/predict", json_body=VALID_PAYLOAD)
    times.append(time.time() - t0)

avg_ms  = (sum(times) / N_CALLS) * 1000
max_ms  = max(times) * 1000
check(f"Average response time < 2000 ms", avg_ms < 2000,
      f"avg={avg_ms:.0f} ms  max={max_ms:.0f} ms ({N_CALLS} calls)")

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
