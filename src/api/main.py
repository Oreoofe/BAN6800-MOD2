"""Module 4/5 — FastAPI application with /predict endpoint (updated with real operating point + live SHAP reason codes).

Run locally:   uvicorn src.api.main:app --port 8000
Demo script:   python api_demo.py
"""
import os, json, joblib
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="CareConnect Predictive Engagement API",
              description="No-show risk scoring with plain-language reason codes (Module 4/5)",
              version="1.1.0")

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/xgboost.joblib")
OP_PATH = os.environ.get("OP_PATH", "artifacts/operating_point.json")
FEATURES = ["Gender", "Age", "Neighbourhood", "Scholarship", "Hipertension", "Diabetes", "Alcoholism",
            "Handcap", "SMS_received", "lead_time_days", "appointment_dow", "appointment_month",
            "age_band", "prior_appointments", "prior_noShows", "prior_noShow_rate"]
PLAIN = {"lead_time_days": "Days between booking and appointment", "Age": "Patient age",
         "prior_noShow_rate": "Share of past appointments missed", "prior_noShows": "Number of past missed appointments",
         "prior_appointments": "Number of past appointments", "SMS_received": "Reminder SMS status",
         "Neighbourhood": "Neighbourhood", "age_band": "Age group", "Scholarship": "Welfare scholarship status",
         "appointment_month": "Month of appointment", "appointment_dow": "Day of week", "Gender": "Gender",
         "Alcoholism": "Alcoholism flag", "Diabetes": "Diabetes flag", "Hipertension": "Hypertension flag", "Handcap": "Disability level"}
model, OP = None, {"flag_threshold": 0.25, "high_threshold": 0.40}

@app.on_event("startup")
def startup():
    global model, OP
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(OP_PATH):
        OP = json.load(open(OP_PATH))

class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    return_reasons: bool = True

class PredictionResponse(BaseModel):
    no_show_probability: float
    risk_tier: str
    flagged_for_outreach: bool
    reason_codes: List[str] = []
    model_version: str = "careconnect_xgb_module4"
    note: str = "Decision support only. A human coordinator decides what, if anything, to offer."

def _tier(p: float) -> str:
    if p >= OP["high_threshold"]: return "high"
    if p >= OP["flag_threshold"]: return "medium"
    return "low"

def _group(name: str) -> str:
    b = name.split("__", 1)[1]
    for g in ["Neighbourhood", "age_band", "Gender"]:
        if b.startswith(g): return g
    return b

def reasons(df: pd.DataFrame, k: int = 3) -> List[str]:
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    names = list(pre.get_feature_names_out())
    c = clf.get_booster().predict(xgb.DMatrix(pre.transform(df), feature_names=names), pred_contribs=True)[0][:-1]
    agg = {}
    for n, v in zip(names, c): agg[_group(n)] = agg.get(_group(n), 0.0) + float(v)
    top = sorted(agg.items(), key=lambda x: -abs(x[1]))[:k]
    return [f"{PLAIN.get(f, f)}: {'raises' if v > 0 else 'lowers'} risk" for f, v in top]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check MODEL_PATH.")
    missing = [f for f in FEATURES if f not in req.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")
    df = pd.DataFrame([{f: req.features[f] for f in FEATURES}])
    p = float(model.predict_proba(df)[0, 1])
    return PredictionResponse(no_show_probability=round(p, 4), risk_tier=_tier(p),
                              flagged_for_outreach=p >= OP["flag_threshold"],
                              reason_codes=reasons(df) if req.return_reasons else [])

@app.get("/model_info")
def model_info():
    return {"model_name": "careconnect_no_show_predictor", "model_type": "XGBoost classifier in an sklearn pipeline",
            "target": "noshow_flag", "operating_point": OP,
            "intended_use": "Identify appointments at higher risk of a missed visit so a person can offer support",
            "non_intended_use": "Deny care, penalise patients, or automate scheduling without human review",
            "fairness_constraint": "Disparity <= 0.05 across Gender and age_band (Ethical Charter Principle 2)"}
