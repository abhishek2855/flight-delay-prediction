"""
Preprocessing Pipeline for Flight Delay Prediction
Based on Chapter 3, Code Snippet 3.3
Maharishi University of Information Technology, Noida
Author: Abhishek (Roll No: 2402031117003)
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Feature groupings as defined in Table 2.1 & Code Snippet 3.2
NUMERIC_COLS = ["DISTANCE", "SCHEDULED_TIME", "SCHEDULED_DEP_HOUR"]
TIME_COLS = ["MONTH", "DAY_OF_WEEK"]
NOMINAL_COLS = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
FEATURE_COLS = NUMERIC_COLS + TIME_COLS + NOMINAL_COLS

# 14 major airlines studied in the project report
AIRLINES = ["AA", "AS", "B6", "DL", "EV", "F9", "HA",
            "MQ", "NK", "OO", "UA", "US", "VX", "WN"]

def clean_flight_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw flight records as specified in Code Snippet 3.1:
    - Drops cancelled and diverted flights (operated flights only)
    - Drops null ARRIVAL_DELAY and duplicate records
    - Derives DELAYED target (1 if ARRIVAL_DELAY >= 15 else 0)
    - Derives SCHEDULED_DEP_HOUR from scheduled departure time (hhmm // 100)
    """
    # Keep only flights that were actually operated
    df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)].copy()
    df = df.dropna(subset=["ARRIVAL_DELAY"]).drop_duplicates()
    
    # Target: delayed if the flight arrives 15 minutes or more after schedule
    df["DELAYED"] = (df["ARRIVAL_DELAY"] >= 15).astype(int)
    
    # Derived time feature from scheduled departure time (hhmm)
    df["SCHEDULED_DEP_HOUR"] = df["SCHEDULED_DEPARTURE"] // 100
    
    # Cast temporal and nominal columns to string for OneHotEncoder
    for col in TIME_COLS + NOMINAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    return df

def build_preprocessor() -> ColumnTransformer:
    """
    Assembles the 3-branch ColumnTransformer (Code Snippet 3.3):
    Branch 1: Numeric (StandardScaler)
    Branch 2: Time (OneHotEncoder)
    Branch 3: Nominal (OneHotEncoder with min_frequency=100)
    """
    numeric_pipeline = Pipeline(steps=[
        ("scale", StandardScaler())
    ])

    time_pipeline = Pipeline(steps=[
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])

    nominal_pipeline = Pipeline(steps=[
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=100))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("Numeric", numeric_pipeline, NUMERIC_COLS),
        ("Time", time_pipeline, TIME_COLS),
        ("Nominal", nominal_pipeline, NOMINAL_COLS)
    ])

    return preprocessor
