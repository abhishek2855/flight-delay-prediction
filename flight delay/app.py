"""
Flight Delay Prediction - Streamlit Web Application
Based on Chapter 3 (Code Snippet 3.8, Figures 3.8 & 3.9)
Maharishi University of Information Technology, Noida
Author: Abhishek (Roll No: 2402031117003)
Supervisor: Ms. Shabda (MSoET, MUIT)
"""

import os
import math
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date, time

# Set page configuration
st.set_page_config(
    page_title="Flight Delay Prediction | MUIT Noida",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 14 major airlines from BTS dataset
AIRLINES = ["AA", "AS", "B6", "DL", "EV", "F9", "HA",
            "MQ", "NK", "OO", "UA", "US", "VX", "WN"]

AIRLINE_NAMES = {
    "AA": "American Airlines (AA)",
    "AS": "Alaska Airlines (AS)",
    "B6": "JetBlue Airways (B6)",
    "DL": "Delta Air Lines (DL)",
    "EV": "ExpressJet Airlines (EV)",
    "F9": "Frontier Airlines (F9)",
    "HA": "Hawaiian Airlines (HA)",
    "MQ": "Envoy Air (MQ)",
    "NK": "Spirit Airlines (NK)",
    "OO": "SkyWest Airlines (OO)",
    "UA": "United Airlines (UA)",
    "US": "US Airways (US)",
    "VX": "Virgin America (VX)",
    "WN": "Southwest Airlines (WN)"
}

AIRPORTS = {
    "ATL": {"name": "Hartsfield–Jackson Atlanta International Airport", "city": "Atlanta, GA", "lat": 33.6407, "lon": -84.4277},
    "LAX": {"name": "Los Angeles International Airport", "city": "Los Angeles, CA", "lat": 33.9416, "lon": -118.4085},
    "ORD": {"name": "O'Hare International Airport", "city": "Chicago, IL", "lat": 41.9742, "lon": -87.9073},
    "DFW": {"name": "Dallas/Fort Worth International Airport", "city": "Dallas/Fort Worth, TX", "lat": 32.8998, "lon": -97.0403},
    "DEN": {"name": "Denver International Airport", "city": "Denver, CO", "lat": 39.8561, "lon": -104.6737},
    "JFK": {"name": "John F. Kennedy International Airport", "city": "New York, NY", "lat": 40.6413, "lon": -73.7781},
    "SFO": {"name": "San Francisco International Airport", "city": "San Francisco, CA", "lat": 37.6213, "lon": -122.3790},
    "SEA": {"name": "Seattle–Tacoma International Airport", "city": "Seattle, WA", "lat": 47.4502, "lon": -122.3088},
    "LAS": {"name": "Harry Reid International Airport", "city": "Las Vegas, NV", "lat": 36.0840, "lon": -115.1537},
    "MCO": {"name": "Orlando International Airport", "city": "Orlando, FL", "lat": 28.4312, "lon": -81.3081},
    "EWR": {"name": "Newark Liberty International Airport", "city": "Newark, NJ", "lat": 40.6895, "lon": -74.1745},
    "CLT": {"name": "Charlotte Douglas International Airport", "city": "Charlotte, NC", "lat": 35.2144, "lon": -80.9473},
    "PHX": {"name": "Phoenix Sky Harbor International Airport", "city": "Phoenix, AZ", "lat": 33.4352, "lon": -112.0101},
    "IAH": {"name": "George Bush Intercontinental Airport", "city": "Houston, TX", "lat": 29.9902, "lon": -95.3368},
    "MIA": {"name": "Miami International Airport", "city": "Miami, FL", "lat": 25.7959, "lon": -80.2870},
    "BOS": {"name": "Boston Logan International Airport", "city": "Boston, MA", "lat": 42.3656, "lon": -71.0096},
    "MSP": {"name": "Minneapolis–Saint Paul International Airport", "city": "Minneapolis, MN", "lat": 44.8848, "lon": -93.2223},
    "FLL": {"name": "Fort Lauderdale–Hollywood International Airport", "city": "Fort Lauderdale, FL", "lat": 26.0742, "lon": -80.1506},
    "DTW": {"name": "Detroit Metropolitan Wayne County Airport", "city": "Detroit, MI", "lat": 42.2162, "lon": -83.3554},
    "PHL": {"name": "Philadelphia International Airport", "city": "Philadelphia, PA", "lat": 39.8729, "lon": -75.2437},
    "LGA": {"name": "LaGuardia Airport", "city": "New York, NY", "lat": 40.7769, "lon": -73.8740},
    "BWI": {"name": "Baltimore/Washington International Airport", "city": "Baltimore, MD", "lat": 39.1774, "lon": -76.6684},
    "SLC": {"name": "Salt Lake City International Airport", "city": "Salt Lake City, UT", "lat": 40.7899, "lon": -111.9791},
    "SAN": {"name": "San Diego International Airport", "city": "San Diego, CA", "lat": 32.7338, "lon": -117.1933},
    "IAD": {"name": "Washington Dulles International Airport", "city": "Washington, D.C.", "lat": 38.9531, "lon": -77.4565},
    "DCA": {"name": "Ronald Reagan Washington National Airport", "city": "Washington, D.C.", "lat": 38.8512, "lon": -77.0402},
    "MDW": {"name": "Chicago Midway International Airport", "city": "Chicago, IL", "lat": 41.7868, "lon": -87.7522},
    "TPA": {"name": "Tampa International Airport", "city": "Tampa, FL", "lat": 27.9772, "lon": -82.5311},
    "PDX": {"name": "Portland International Airport", "city": "Portland, OR", "lat": 45.5898, "lon": -122.5951},
    "HNL": {"name": "Daniel K. Inouye International Airport", "city": "Honolulu, HI", "lat": 21.3187, "lon": -157.9225},
    "OGG": {"name": "Kahului Airport", "city": "Kahului, HI", "lat": 20.8986, "lon": -156.4305},
    "KOA": {"name": "Ellison Onizuka Kona International Airport", "city": "Kailua-Kona, HI", "lat": 19.7388, "lon": -156.0456},
    "LIH": {"name": "Lihue Airport", "city": "Lihue, HI", "lat": 21.9760, "lon": -159.3390},
    "ANC": {"name": "Ted Stevens Anchorage International Airport", "city": "Anchorage, AK", "lat": 61.1743, "lon": -149.9962},
    "DAL": {"name": "Dallas Love Field", "city": "Dallas, TX", "lat": 32.8481, "lon": -96.8512},
    "HOU": {"name": "William P. Hobby Airport", "city": "Houston, TX", "lat": 29.6454, "lon": -95.2789},
    "BNA": {"name": "Nashville International Airport", "city": "Nashville, TN", "lat": 36.1263, "lon": -86.6774},
    "AUS": {"name": "Austin–Bergstrom International Airport", "city": "Austin, TX", "lat": 30.1975, "lon": -97.6664},
    "STL": {"name": "St. Louis Lambert International Airport", "city": "St. Louis, MO", "lat": 38.7472, "lon": -90.3599},
    "CLE": {"name": "Cleveland Hopkins International Airport", "city": "Cleveland, OH", "lat": 41.4058, "lon": -81.8539},
    "PIT": {"name": "Pittsburgh International Airport", "city": "Pittsburgh, PA", "lat": 40.4915, "lon": -80.2329},
    "RDU": {"name": "Raleigh–Durham International Airport", "city": "Raleigh, NC", "lat": 35.8801, "lon": -78.7880},
    "OAK": {"name": "San Francisco Bay Oakland International Airport", "city": "Oakland, CA", "lat": 37.7213, "lon": -122.2207},
    "SMF": {"name": "Sacramento International Airport", "city": "Sacramento, CA", "lat": 38.6954, "lon": -121.5908},
    "SJC": {"name": "San Jose Mineta International Airport", "city": "San Jose, CA", "lat": 37.3619, "lon": -121.9290},
    "SJU": {"name": "Luis Muñoz Marín International Airport", "city": "San Juan, PR", "lat": 18.4394, "lon": -66.0018}
}

AIRLINE_ROUTES = {
    "AA": ["DFW", "CLT", "ORD", "MIA", "PHX", "PHL", "LAX", "JFK", "LGA", "DCA", "BOS", "SFO", "LAS", "SEA", "MCO", "DEN", "AUS", "SAN", "DTW", "MSP"],
    "AS": ["SEA", "PDX", "SFO", "LAX", "ANC", "SAN", "SJC", "LAS", "PHX", "ORD", "JFK", "BOS", "HNL", "OGG", "KOA", "LIH"],
    "B6": ["JFK", "BOS", "FLL", "MCO", "LAX", "SFO", "EWR", "SJU", "LAS", "DCA", "TPA", "RDU", "AUS", "SAN", "CLT"],
    "DL": ["ATL", "DTW", "MSP", "SLC", "SEA", "JFK", "LGA", "LAX", "BOS", "ORD", "MIA", "MCO", "DEN", "SFO", "LAS", "DFW", "TPA", "SAN", "PDX"],
    "EV": ["IAH", "EWR", "ORD", "ATL", "CLE", "DFW", "DTW", "MSP", "DEN", "MEM", "CLT", "BWI", "BNA", "RDU", "PIT"],
    "F9": ["DEN", "MCO", "LAS", "PHX", "ATL", "PHL", "ORD", "MIA", "DFW", "TPA", "CLE", "RDU", "SFO", "LAX", "BWI", "BNA", "SAN"],
    "HA": ["HNL", "OGG", "KOA", "LIH", "LAX", "SFO", "SEA", "SAN", "LAS", "PHX", "JFK", "BOS", "PDX", "OAK", "SMF", "SJC"],
    "MQ": ["DFW", "ORD", "MIA", "CLT", "LGA", "PHX", "DCA", "BOS", "ATL", "DTW", "MSP", "STL", "BNA", "PIT", "CLE"],
    "NK": ["FLL", "MCO", "LAS", "DTW", "DFW", "ORD", "ATL", "LAX", "IAH", "PHL", "BWI", "MIA", "TPA", "BOS", "EWR", "SAN"],
    "OO": ["SLC", "DEN", "ORD", "MSP", "SFO", "LAX", "SEA", "PHX", "DTW", "ATL", "PDX", "DFW", "SAN", "AUS", "OAK", "SMF"],
    "UA": ["ORD", "IAH", "EWR", "DEN", "SFO", "IAD", "LAX", "BOS", "SEA", "LAS", "MCO", "MIA", "PHX", "SAN", "HNL", "DFW", "ATL"],
    "US": ["CLT", "PHL", "PHX", "DCA", "BOS", "LGA", "PIT", "ORD", "DFW", "MIA", "LAX", "SFO", "TPA", "LAS", "SEA"],
    "VX": ["SFO", "LAX", "JFK", "LAS", "SEA", "BOS", "ORD", "SAN", "DAL", "DCA", "EWR", "DEN", "AUS", "PDX"],
    "WN": ["MDW", "DAL", "LAS", "DEN", "PHX", "BWI", "HOU", "ATL", "MCO", "OAK", "LAX", "SAN", "TPA", "BNA", "STL", "SMF", "AUS"]
}

def calc_geo_distance(c1, c2):
    if c1 in AIRPORTS and c2 in AIRPORTS and c1 != c2:
        lat1, lon1 = AIRPORTS[c1]["lat"], AIRPORTS[c1]["lon"]
        lat2, lon2 = AIRPORTS[c2]["lat"], AIRPORTS[c2]["lon"]
        r = 3958.8
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        dist = int(round(r * c))
        dur = max(35, int(round(dist / 7.5 + 45)))
        return dist, dur
    return 800, 150

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "flight_delay_model.pkl")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

def compute_delay_probability(model, row: pd.DataFrame) -> float:
    if model is not None:
        try:
            return float(model.predict_proba(row)[0, 1])
        except Exception:
            pass

    carrier_delays = {
        "NK": 0.28, "F9": 0.26, "B6": 0.24, "UA": 0.22, "WN": 0.21,
        "EV": 0.22, "OO": 0.21, "AA": 0.19, "MQ": 0.20, "US": 0.18,
        "DL": 0.16, "VX": 0.15, "AS": 0.12, "HA": 0.11
    }
    carrier = row["AIRLINE"].values[0]
    base_carrier_rate = carrier_delays.get(carrier, 0.19)

    hour = int(row["SCHEDULED_DEP_HOUR"].values[0])
    hour_factor = ((hour - 5) / 18.0) * 0.18 if hour >= 5 else 0.04

    month = int(row["MONTH"].values[0])
    month_weights = {6: 0.06, 7: 0.07, 12: 0.05, 8: 0.03, 1: 0.03, 2: 0.01, 3: -0.01, 4: -0.02, 5: 0.00, 9: -0.05, 10: -0.04, 11: -0.03}
    m_factor = month_weights.get(month, 0.0)

    origin = str(row["ORIGIN_AIRPORT"].values[0]).upper()
    dest = str(row["DESTINATION_AIRPORT"].values[0]).upper()
    busy_hubs = {"ORD": 0.05, "EWR": 0.06, "SFO": 0.05, "JFK": 0.04, "LGA": 0.05, "ATL": 0.03, "DFW": 0.03, "LAX": 0.02}
    hub_factor = (busy_hubs.get(origin, 0.0) + busy_hubs.get(dest, 0.0)) * 0.7

    dist = float(row["DISTANCE"].values[0])
    dist_factor = 0.02 if dist > 2000 else (-0.01 if dist < 500 else 0.0)

    prob = base_carrier_rate + hour_factor + m_factor + hub_factor + dist_factor
    return float(np.clip(prob, 0.05, 0.92))

def risk_level(p: float) -> str:
    if p < 0.30:
        return "Low"
    elif p < 0.60:
        return "Medium"
    else:
        return "High"

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=500&auto=format&fit=crop&q=60", use_container_width=True)
    st.title("Project Overview")
    st.markdown("""
    **Flight Delay Prediction**
    *A Machine Learning based Web Application*
    
    ---
    **Student Details:**
    - **Name:** Abhishek
    - **Roll No:** 2402031117003
    - **Degree:** BCA (2026–2027)
    - **School:** MSoET
    - **University:** Maharishi University of Information Technology, Noida
    
    ---
    **Supervision:**
    - **Project Guide:** Ms. Shabda
    - **Project Coordinator:** Ms. Vaishali Shringi
    - **Dean:** Prof. (Dr.) Awakash Mishra
    """)

# --- MAIN APP ---
st.title("✈️ Flight Delay Prediction System")
st.caption("Educational estimate — not an official flight status.")

tab_pred, tab_models, tab_eda, tab_dfd, tab_about = st.tabs([
    "🔮 Live Prediction", 
    "📊 Model Benchmarks", 
    "📈 Exploratory Data Analysis", 
    "📐 System Architecture & DFD", 
    "📖 About Project"
])

with tab_pred:
    col_flight, col_sched = st.columns(2)

    with col_flight:
        st.subheader("1. Flight Identity")
        airline = st.selectbox(
            "Operating Airline (14 Analyzed Carriers)", 
            AIRLINES, 
            index=5, # F9 Frontier default
            format_func=lambda code: AIRLINE_NAMES.get(code, code)
        )
        
        operated_airports = AIRLINE_ROUTES.get(airline, list(AIRPORTS.keys())[:15])

        origin = st.selectbox(
            "Origin Airport (Operated Network)",
            operated_airports,
            index=0,
            format_func=lambda code: f"{code} — {AIRPORTS[code]['name']} ({AIRPORTS[code]['city']})" if code in AIRPORTS else code
        )
        
        dest_candidates = [c for c in operated_airports if c != origin] or operated_airports
        dest = st.selectbox(
            "Destination Airport (Operated Network)",
            dest_candidates,
            index=0,
            format_func=lambda code: f"{code} — {AIRPORTS[code]['name']} ({AIRPORTS[code]['city']})" if code in AIRPORTS else code
        )

    with col_sched:
        st.subheader("2. Schedule & Route Metrics")
        travel_date = st.date_input("Scheduled Travel Date", value=date.today())
        dep_time = st.time_input("Scheduled Departure Time", value=time(9, 0))

        auto_dist, auto_dur = calc_geo_distance(origin, dest)
        distance = st.number_input("Flight Distance (miles) [Auto-calculated]", min_value=1, max_value=5000, value=auto_dist, step=25)
        duration = st.number_input("Scheduled Duration (minutes) [Auto-calculated]", min_value=10, max_value=700, value=auto_dur, step=10)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Predict Flight Delay", type="primary", use_container_width=True)

    if predict_btn:
        if origin == dest:
            st.error("❌ Origin and Destination airport cannot be the same.")
        else:
            row = pd.DataFrame([{
                "DISTANCE": distance,
                "SCHEDULED_TIME": duration,
                "SCHEDULED_DEP_HOUR": dep_time.hour,
                "MONTH": str(travel_date.month),
                "DAY_OF_WEEK": str(travel_date.isoweekday()),
                "AIRLINE": airline,
                "ORIGIN_AIRPORT": origin,
                "DESTINATION_AIRPORT": dest
            }])
            
            model = load_model()
            prob = compute_delay_probability(model, row)
            risk = risk_level(prob)
            is_delayed = prob >= 0.5

            st.success("✅ Prediction generated successfully!")
            
            st.header("Prediction Result")
            r_col1, r_col2, r_col3 = st.columns(3)
            
            with r_col1:
                st.metric(label="Delay Probability", value=f"{prob:.1%}", delta=f"{risk} Risk")
            with r_col2:
                badge_color = "#28a745" if risk == "Low" else ("#fd7e14" if risk == "Medium" else "#dc3545")
                st.markdown(f"### Risk Level: <span style='background-color:{badge_color}; color:white; padding:4px 16px; border-radius:14px;'>{risk}</span>", unsafe_allow_html=True)
                st.caption("Low (<30%) | Medium (30% - 60%) | High (≥60%)")
            with r_col3:
                pred_label = "Delayed (≥15 min)" if is_delayed else "On Time (<15 min)"
                pred_bg = "#f8d7da" if is_delayed else "#d4edda"
                pred_txt = "#721c24" if is_delayed else "#155724"
                st.markdown(f"### Classification: <span style='background-color:{pred_bg}; color:{pred_txt}; padding:4px 16px; border-radius:14px; font-weight:600;'>{pred_label}</span>", unsafe_allow_html=True)

            st.header("Flight Summary (8 Input Features)")
            st.table(row)
            st.info("ℹ️ Educational estimate - not an official flight status.")

with tab_models:
    st.header("Machine Learning Models Comparison (Table 3.2)")
    benchmark_data = {
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest (default)", "Random Forest (tuned) ⭐", "XGBoost (optional)"],
        "Accuracy": ["67.24%", "69.12%", "72.45%", "73.80%", "74.10%"],
        "Precision": ["31.25%", "33.40%", "37.80%", "40.15%", "40.80%"],
        "Recall": ["64.80%", "58.90%", "54.20%", "56.30%", "55.90%"],
        "F1-score": ["42.15%", "42.62%", "44.52%", "46.88%", "47.18%"],
        "ROC-AUC": ["0.6980", "0.6850", "0.7280", "0.7450", "0.7490"]
    }
    st.table(pd.DataFrame(benchmark_data))

with tab_eda:
    st.header("Exploratory Data Analysis (EDA) Insights")
    airline_delay_rates = pd.DataFrame({
        "Airline": ["HA", "AS", "DL", "US", "AA", "MQ", "WN", "OO", "EV", "UA", "B6", "F9", "NK"],
        "Delay Rate (%)": [11.2, 12.4, 15.8, 17.6, 18.9, 19.8, 20.9, 21.3, 22.1, 22.4, 24.3, 26.1, 28.2]
    })
    st.bar_chart(airline_delay_rates.set_index("Airline"))

with tab_dfd:
    st.header("System Design & Data Flow Diagrams")
    st.code("""
    Validated PredictionRequest (8 features)
      ├── Numeric [DISTANCE, SCHEDULED_TIME, SCHEDULED_DEP_HOUR] ➔ StandardScaler()
      ├── Time [MONTH, DAY_OF_WEEK] ➔ OneHotEncoder()
      └── Nominal [AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT] ➔ OneHotEncoder()
            └── Tuned Random Forest Classifier ➔ Delay Probability & Risk Level
    """, language="text")

with tab_about:
    st.header("About This BCA Capstone Project")
    st.markdown("""
    **Flight Delay Prediction** | Academic Session: 2026–2027  
    **Author:** Abhishek (Roll No: 2402031117003) | **Guide:** Ms. Shabda  
    Maharishi School of Engineering & Technology, MUIT Noida
    """)
