"""
Model Training, Tuning and Evaluation Script
Based on Chapter 3, Code Snippets 3.4 - 3.7
Maharishi University of Information Technology, Noida
Author: Abhishek (Roll No: 2402031117003)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from preprocessing import (build_preprocessor, clean_flight_data,
                           FEATURE_COLS, NUMERIC_COLS, TIME_COLS, NOMINAL_COLS, AIRLINES)

def generate_sample_dataset(n_samples: int = 15000) -> pd.DataFrame:
    """
    Generates a realistic sample dataset reflecting BTS statistical patterns
    if data/flights.csv is not yet placed in the data/ folder.
    """
    np.random.seed(42)
    airports = ["JFK", "LAX", "ORD", "DFW", "ATL", "SFO", "DEN", "MIA", "CLT", "LAS", "SEA", "PHX", "BOS", "EWR"]
    
    origin = np.random.choice(airports, size=n_samples)
    destination = [np.random.choice([a for a in airports if a != o]) for o in origin]
    airline = np.random.choice(AIRLINES, size=n_samples)
    
    month = np.random.choice(range(1, 13), size=n_samples)
    day_of_week = np.random.choice(range(1, 8), size=n_samples)
    dep_hour = np.random.choice(range(5, 24), size=n_samples)
    scheduled_departure = dep_hour * 100 + np.random.choice(range(0, 60, 5), size=n_samples)
    
    distance = np.random.randint(200, 2600, size=n_samples)
    scheduled_time = (distance / 7.5 + np.random.randint(30, 75, size=n_samples)).astype(int)
    
    # Calculate baseline delay probability based on historical factors
    # Delays peak in evening hours, summer/winter, and for certain carriers
    hour_factor = (dep_hour - 5) / 19.0 * 0.20  # +0% at 5am up to +20% at night
    carrier_weights = {
        "NK": 0.12, "F9": 0.10, "B6": 0.08, "UA": 0.05, "WN": 0.04,
        "AA": 0.02, "DL": -0.04, "AS": -0.06, "HA": -0.07
    }
    carrier_factor = np.array([carrier_weights.get(a, 0.0) for a in airline])
    month_factor = np.array([0.08 if m in [6, 7, 12] else (0.04 if m in [1, 8] else -0.03) for m in month])
    hub_factor = np.array([0.05 if o in ["ORD", "EWR", "SFO", "ATL"] else 0.0 for o in origin])
    
    base_prob = 0.14 + hour_factor + carrier_factor + month_factor + hub_factor
    base_prob = np.clip(base_prob, 0.05, 0.85)
    
    is_delayed = (np.random.rand(n_samples) < base_prob).astype(int)
    arrival_delay = np.where(is_delayed == 1, np.random.randint(15, 180, size=n_samples), np.random.randint(-25, 14, size=n_samples))
    
    df = pd.DataFrame({
        "CANCELLED": 0,
        "DIVERTED": 0,
        "ARRIVAL_DELAY": arrival_delay,
        "SCHEDULED_DEPARTURE": scheduled_departure,
        "MONTH": month.astype(str),
        "DAY_OF_WEEK": day_of_week.astype(str),
        "AIRLINE": airline,
        "ORIGIN_AIRPORT": origin,
        "DESTINATION_AIRPORT": destination,
        "DISTANCE": distance,
        "SCHEDULED_TIME": scheduled_time
    })
    return df

def evaluate_model(pipe, X_test, y_test):
    """Consolidated model evaluation matching Code Snippet 3.5"""
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred)),
        "F1-score": float(f1_score(y_test, y_pred)),
        "ROC-AUC": float(roc_auc_score(y_test, y_prob)),
    }

def train_and_export_pipeline():
    print("=== Phase 1 & 2: Dataset Loading & Preprocessing ===")
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "flights.csv")
    
    if os.path.exists(data_path):
        print(f"Loading dataset from {data_path}...")
        df = pd.read_csv(data_path, low_memory=False)
        df = clean_flight_data(df)
        if len(df) > 500_000:
            df = df.sample(n=500_000, random_state=42)
    else:
        print("Data file not found at data/flights.csv; generating representative sample dataset...")
        df = generate_sample_dataset(n_samples=25000)
        df = clean_flight_data(df)
        
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['DELAYED'].value_counts(normalize=True)}")
    
    X = df[FEATURE_COLS]
    y = df["DELAYED"]
    
    # 70/30 Stratified Split with random_state=42 (Code Snippet 3.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    preprocessor = build_preprocessor()
    
    print("\n=== Phase 3: Model Training (LR, DT, RF) ===")
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42
        )
    }
    
    pipelines = {}
    for name, clf in models.items():
        print(f"Training {name}...")
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
        
    print("\n=== Model Evaluation Results ===")
    results = {}
    for name, p in pipelines.items():
        results[name] = evaluate_model(p, X_test, y_test)
        
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    
    print("\n=== Phase 3.6: Hyperparameter Tuning via RandomizedSearchCV ===")
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10, 15, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    }
    
    random_search = RandomizedSearchCV(
        estimator=pipelines["Random Forest"],
        param_distributions=param_grid,
        n_iter=5, cv=3, scoring="roc_auc",
        random_state=42, n_jobs=-1
    )
    print("Fitting RandomizedSearchCV on training data...")
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    print(f"Best hyperparameters: {random_search.best_params_}")
    
    best_metrics = evaluate_model(best_model, X_test, y_test)
    print(f"Tuned Random Forest Performance: {best_metrics}")
    
    # Save model pipeline
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "flight_delay_model.pkl")
    
    joblib.dump(best_model, out_path)
    print(f"\nModel pipeline successfully saved to: {out_path}")
    
    # Sanity check reload
    loaded = joblib.load(out_path)
    test_sample = X_test.head(3)
    sample_preds = loaded.predict_proba(test_sample)[:, 1]
    print(f"Sanity check predictions on first 3 test rows: {sample_preds}")
    print("Model ready for Streamlit web application.")

if __name__ == "__main__":
    train_and_export_pipeline()
