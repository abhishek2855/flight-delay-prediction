# Flight Delay Prediction System ✈️
### A Machine Learning based Web Application for Predicting Flight Delays

**Degree Programme:** Bachelor of Computer Application (BCA)  
**Academic Session:** 2026–2027  
**Institution:** Maharishi School of Engineering & Technology (MSoET), Maharishi University of Information Technology (MUIT), Noida  
**Submitted By:** Abhishek (Roll No.: 2402031117003)  
**Project Guide:** Ms. Shabda  
**Project Coordinator:** Ms. Vaishali Shringi  
**Dean:** Prof. (Dr.) Awakash Mishra  

---

## 📖 Project Overview

Air travel punctuality is essential to global transportation networks. When one flight departs late, crew duty hours expire, airport gates stay occupied, and connecting passengers miss subsequent flights. 

Following the convention of the **U.S. Bureau of Transportation Statistics (BTS)**, a flight is classified as **Delayed (`DELAYED = 1`)** if its arrival delay is **15 minutes or more** past the scheduled arrival time.

This project delivers:
1. **Pre-departure Feature Extraction**: Strictly eliminates data leakage by considering only schedule-level parameters available prior to departure.
2. **Multi-Branch Preprocessing Pipeline**: Built using Scikit-learn's `ColumnTransformer` (Numeric `StandardScaler`, Time `OneHotEncoder`, Nominal `OneHotEncoder`).
3. **Machine Learning Benchmarking**: Logistic Regression (Baseline), Decision Tree, Random Forest (Tuned best estimator), and XGBoost.
4. **Interactive Web Application**:
   - **`index.html`**: Zero-dependency, modern interactive web interface ready to open directly in any browser (Chrome, Edge, Opera) or deploy to static hosting.
   - **`app.py`**: Python Streamlit application matching Code Snippet 3.8 from the project report.

---

## 🗂️ Project Directory Architecture (Chapter 5, Code Snippet 5.1)

```text
Flight-Delay-Prediction/
|-- data/
|   |-- flights.csv              # BTS 2015 dataset (~5.8M rows, Kaggle)
|-- notebooks/
|   |-- model_training.ipynb     # EDA, preprocessing, training & tuning
|-- src/
|   |-- preprocessing.py         # Reusable ColumnTransformer & cleaning
|   |-- train.py                 # Training, RandomizedSearchCV & model export
|   |-- predict.py               # Inference and probability scoring
|-- models/
|   |-- flight_delay_model.pkl   # Serialized best_estimator_ pipeline
|-- app.py                       # Streamlit web application
|-- index.html                   # Interactive standalone web interface
|-- requirements.txt             # Python dependencies
|-- README.md                    # Project documentation
```

---

## 🚀 How to Run the Website

### Method 1: Instant Browser Launch (Recommended - Zero Setup)
Simply double-click **`index.html`** in this folder, or right-click and choose **Open with Chrome / Edge / Opera**.
- Live interactive delay probability predictions.
- Dynamic gauge animations, Low/Medium/High risk indicators (<30%, 30–60%, ≥60%).
- Quick route presets (JFK ➔ LAX, ATL ➔ ORD, DEN ➔ SFO).
- Table 3.2 benchmark tables, Table 4.1 test runner, and DFD diagrams.

### Method 2: Streamlit Web Application (`app.py`)
If you have Python installed:
```bash
# 1. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-train the model pipeline
python src/train.py

# 4. Launch the Streamlit application
streamlit run app.py
```

---

## 📊 Model Evaluation Results (Table 3.2)

Evaluated on the held-out 30% test set using stratified sampling (`random_state=42`):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 67.24% | 31.25% | 64.80% | 42.15% | 0.6980 | Baseline |
| **Decision Tree** | 69.12% | 33.40% | 58.90% | 42.62% | 0.6850 | max_depth=12 |
| **Random Forest (Default)** | 72.45% | 37.80% | 54.20% | 44.52% | 0.7280 | n_estimators=100 |
| **Random Forest (Tuned)** | **73.80%** | **40.15%** | **56.30%** | **46.88%** | **0.7450** | **Best Model** |
| **XGBoost (Optional)** | 74.10% | 40.80% | 55.90% | 47.18% | 0.7490 | Gradient Boosted |

---

## 📐 System Pipeline Architecture (DFD Level 2)

```text
Raw Validated Flight Record (8 Features)
  │
  ├── 1. Numeric Pipeline [DISTANCE, SCHEDULED_TIME, SCHEDULED_DEP_HOUR]
  │      └── StandardScaler()
  │
  ├── 2. Time Pipeline [MONTH, DAY_OF_WEEK]
  │      └── OneHotEncoder(handle_unknown='ignore')
  │
  └── 3. Nominal Pipeline [AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT]
         └── OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=100)
               │
               ▼
       Combined Feature Vector
               │
               ▼
       Tuned Random Forest Classifier ──► Delay Probability & Risk Level
```

---

## ⚠️ Important Educational Disclaimer

*This application has been built only for learning and for exploring data. It is NOT an official flight-status service, and its output should never replace the information issued by an airline, an airport or an aviation authority.*
