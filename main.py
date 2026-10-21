"""
main.py  -  FastAPI backend for the churn model
Run locally:  uvicorn main:app --reload
Docs page  :  http://127.0.0.1:8000/docs
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import preprocessing  # noqa: F401  (the saved pipeline needs this module to be importable)

load_dotenv()  # reads a local .env file if present (on Render use the dashboard env vars)

MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
MODEL_PATH = os.getenv("MODEL_PATH", f"model/churn_model_v{MODEL_VERSION}.joblib")
METADATA_PATH = os.getenv("METADATA_PATH", "model/model_metadata.json")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model ONCE when the server starts (not on every request)."""
    try:
        state["model"] = joblib.load(MODEL_PATH)
        with open(METADATA_PATH) as f:
            state["metadata"] = json.load(f)
        logger.info("Model loaded: %s", MODEL_PATH)
    except Exception as exc:
        logger.exception("Could not load model: %s", exc)
        state["model"] = None
    yield
    state.clear()


app = FastAPI(title="Customer Churn Prediction API", version=MODEL_VERSION, lifespan=lifespan)


class CustomerInput(BaseModel):
    """Input validation: wrong type / out-of-range values are rejected automatically (HTTP 422)."""
    PlanType: Literal["Purple Plan", "Green Plan", "Blue Plan", "Red Plan"]
    Promotion: Literal["yes", "no"]
    HasRenewed: Literal["yes", "no"]
    DeviceType: Literal["-", "iPhone", "Galaxy"] = "-"
    Territory: str = Field("CA", min_length=2, max_length=2)
    StartMonth: Literal["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] = "Jan"
    BaseFee: float = Field(..., ge=0, le=100)
    NumberOfPenalties: int = Field(..., ge=0, le=50)
    CurrentPeriodUsage: float = Field(..., gt=0, le=1000)
    PriorPeriodUsage: float = Field(..., ge=0, le=1000)
    ServiceRating: float = Field(..., ge=0, le=20)
    ServiceTickets: int = Field(..., ge=0, le=100)
    CustomerTenure: int = Field(..., ge=0, le=500)


@app.get("/")
def root():
    return {"message": "Churn API is running. Open /docs for the interactive page."}


@app.get("/health")
def health():
    """Used by Render / Docker to check the service is alive."""
    ok = state.get("model") is not None
    return {"status": "ok" if ok else "model_not_loaded", "model_version": MODEL_VERSION}


@app.get("/model-info")
def model_info():
    if state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state["metadata"]


@app.post("/predict")
def predict(customer: CustomerInput):
    if state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        row = pd.DataFrame([customer.model_dump()])
        # the training data had floats for these columns
        row["NumberOfPenalties"] = row["NumberOfPenalties"].astype(float)
        row["ServiceTickets"] = row["ServiceTickets"].astype(float)
        row["CustomerTenure"] = row["CustomerTenure"].astype(float)

        prob = float(state["model"].predict_proba(row)[0, 1])
        label = "Churn" if prob >= THRESHOLD else "No churn"
        risk = "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"
        return {
            "prediction": label,
            "churn_probability": round(prob, 4),
            "risk_level": risk,
            "model_version": MODEL_VERSION,
        }
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
