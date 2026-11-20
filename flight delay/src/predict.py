"""
Prediction Utilities for Flight Delay Prediction
Provides inference functions for the Streamlit web application.
"""

import os
import joblib
import pandas as pd
from datetime import date, time

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "flight_delay_model.pkl")

def get_risk_level(prob: float) -> str:
    """
    Returns risk categorization according to project threshold:
    - Low: < 30%
    - Medium: 30% - 60%
    - High: >= 60%
    """
    if prob < 0.30:
        return "Low"
    elif prob < 0.60:
        return "Medium"
    else:
        return "High"

def format_prediction_input(
    airline: str,
    origin: str,
    destination: str,
    travel_date: date,
    departure_time: time,
    distance: int,
    duration: int
) -> pd.DataFrame:
    """
    Converts user input form values into the exact 8-feature DataFrame expected
    by the ColumnTransformer pipeline.
    """
    return pd.DataFrame([{
        "DISTANCE": float(distance),
        "SCHEDULED_TIME": float(duration),
        "SCHEDULED_DEP_HOUR": int(departure_time.hour),
        "MONTH": str(travel_date.month),
        "DAY_OF_WEEK": str(travel_date.isoweekday()),
        "AIRLINE": str(airline).strip().upper(),
        "ORIGIN_AIRPORT": str(origin).strip().upper(),
        "DESTINATION_AIRPORT": str(destination).strip().upper()
    }])

def predict_flight_delay(model, input_df: pd.DataFrame):
    """
    Runs model inference on the validated DataFrame.
    Returns:
    - prob: float delay probability
    - risk: str ("Low", "Medium", "High")
    - pred_class: str ("Delayed" or "On time")
    """
    prob = float(model.predict_proba(input_df)[0, 1])
    risk = get_risk_level(prob)
    pred_class = "Delayed" if prob >= 0.5 else "On time"
    return prob, risk, pred_class
