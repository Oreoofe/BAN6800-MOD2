"""API demo script (Module 5): starts the FastAPI service, calls every endpoint, prints the results.
Usage:  python api_demo.py          (from the package root, after artifacts/xgboost.joblib exists)
"""
import subprocess, sys, time, json, requests
PROC = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app", "--port", "8000", "--log-level", "warning"])
BASE = "http://127.0.0.1:8000"
try:
    for _ in range(40):
        try:
            if requests.get(f"{BASE}/health", timeout=1).ok: break
        except Exception: time.sleep(0.5)
    print("GET /health      ->", requests.get(f"{BASE}/health").json())
    print("GET /model_info  ->", json.dumps(requests.get(f"{BASE}/model_info").json(), indent=2))
    base = dict(Gender="F", Neighbourhood="JARDIM DA PENHA", Scholarship=0, Hipertension=0, Diabetes=0, Alcoholism=0,
                Handcap=0, appointment_dow=2, appointment_month=5)
    patients = {
      "A - long lead time, past misses": dict(base, Age=26, age_band="young_adult", lead_time_days=35, SMS_received=1, prior_appointments=4, prior_noShows=3, prior_noShow_rate=0.75),
      "B - mid lead time, one past miss": dict(base, Age=45, age_band="adult", lead_time_days=10, SMS_received=1, prior_appointments=2, prior_noShows=1, prior_noShow_rate=0.5),
      "C - same-day booking, reliable":   dict(base, Age=62, age_band="senior", lead_time_days=0, SMS_received=0, prior_appointments=5, prior_noShows=0, prior_noShow_rate=0.0),
    }
    for name, f in patients.items():
        r = requests.post(f"{BASE}/predict", json={"features": f}); print(f"\nPOST /predict [{name}]\n", json.dumps(r.json(), indent=2))
    r = requests.post(f"{BASE}/predict", json={"features": {"Age": 40}}); print("\nPOST /predict with missing fields ->", r.status_code, r.json()["detail"][:80], "...")
finally:
    PROC.terminate()
