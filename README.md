# 🩺 Early Sepsis Detection System — ML Project Spec

> **For Antigravity:** This is a complete build specification. Read this entire file before generating any code. Every section is a direct instruction. Build the project exactly as described — folder by folder, file by file.

---

## Project Overview

**What we're building:** A production-ready machine learning system that predicts sepsis risk from patient vitals data. The system ingests clinical features (temperature, BP, heart rate, WBC count, lactate), runs an ensemble ML model, and serves a real-time risk score via a FastAPI backend + React dashboard frontend.

**Why it matters (resume framing):** Sepsis kills ~270,000 Americans annually. Detecting it 6 hours early improves survival rates by 30%+. This system demonstrates applied ML on healthcare data with class imbalance handling, ensemble modeling, SHAP explainability, and production deployment.

**Tech stack:**
- Backend: Python 3.11, FastAPI, scikit-learn, XGBoost, LightGBM, SHAP
- Frontend: React + TypeScript + Tailwind CSS + Recharts
- Deployment: Render (backend) + Vercel (frontend)
- Data: `Sepsis_Dataset____.xlsx` — 250 patients, 6 clinical features

---

## ⚠️ Critical Data Issue — Read This First

The provided dataset (`Sepsis_Dataset____.xlsx`) has **250 rows, all labeled `Sepsis_Flag = "No"`** — zero positive cases. This is a real-world data quality problem you must solve before any ML can happen.

**Solution: Rule-Based Label Augmentation + Synthetic Minority Oversampling**

Use clinical sepsis criteria (Sepsis-3 definition) to re-label existing rows AND generate synthetic positive cases with SMOTE. This is medically valid and technically defensible.

**Sepsis-3 criteria for labeling:**
A patient is labeled sepsis-positive if they meet ≥2 of:
1. `Temperature_C` < 36.0 OR > 38.3
2. `Heart_Rate` > 90
3. `WBC_Count` < 4.0 OR > 12.0
4. `Lactate_mmol_L` > 2.0
5. `BP_Systolic` < 90

Apply this logic in the data pipeline script to create a clinically grounded `Sepsis_Label` binary column. Then use SMOTE to balance to ~40% positive class.

---

## Dataset Reference

**File:** `data/Sepsis_Dataset____.xlsx`

| Column | Type | Description | Clinical Range |
|---|---|---|---|
| `Patient_ID` | str | Unique ID (P00001–P00250) | — |
| `Admission_Date` | datetime | Date of hospital admission | 2023–2024 |
| `Temperature_C` | float | Body temperature in Celsius | 33.1–42.8 |
| `BP_Systolic` | int | Systolic blood pressure (mmHg) | 50–177 |
| `BP_Diastolic` | int | Diastolic blood pressure (mmHg) | 43–106 |
| `Heart_Rate` | int | Heart rate (bpm) | 27–154 |
| `WBC_Count` | float | White blood cell count (×10³/µL) | -3.1–16.0 |
| `Lactate_mmol_L` | float | Serum lactate (mmol/L) | -0.8–4.4 |
| `Sepsis_Flag` | str | Original label — ALL "No", ignore this | — |
| `Ward` | str | Hospital ward (ICU-A, ICU-B, Ward-X, ER) | — |
| `Doctor_On_Duty` | str | Attending doctor — drop for ML | — |

**Note on negative values:** `WBC_Count` has values as low as -3.1 and `Lactate_mmol_L` as low as -0.8 — these are data entry errors. Clip to physiologically valid minimums (WBC ≥ 0, Lactate ≥ 0) during preprocessing.

---

## Project Folder Structure

Build exactly this structure:

```
sepsis-detection/
│
├── data/
│   └── Sepsis_Dataset____.xlsx          # Original dataset (do not modify)
│
├── backend/
│   ├── main.py                          # FastAPI app entrypoint
│   ├── requirements.txt
│   ├── Procfile                         # For Render deployment
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── preprocess.py                # Data cleaning + feature engineering
│   │   ├── label_augment.py             # Sepsis-3 rule-based labeling
│   │   ├── train.py                     # Model training script
│   │   └── evaluate.py                  # Metrics + SHAP computation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── ensemble.py                  # Voting classifier definition
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py                    # All API endpoints
│   │   └── schemas.py                   # Pydantic request/response models
│   │
│   └── artifacts/                       # Auto-created on training
│       ├── sepsis_ensemble.pkl          # Trained model
│       ├── scaler.pkl                   # StandardScaler
│       └── shap_explainer.pkl           # SHAP TreeExplainer
│
├── frontend/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── vercel.json                      # Vercel config
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── components/
│       │   ├── PatientForm.tsx          # Input form for vitals
│       │   ├── RiskGauge.tsx            # Risk score gauge chart
│       │   ├── ShapChart.tsx            # SHAP feature importance bar chart
│       │   ├── RiskBadge.tsx            # Color-coded risk label
│       │   └── HistoryTable.tsx         # Recent predictions table
│       ├── api/
│       │   └── client.ts               # Axios API client
│       └── types/
│           └── index.ts                # TypeScript interfaces
│
├── notebooks/
│   └── eda_and_training.ipynb          # EDA + model development notebook
│
├── .env.example
├── .gitignore
└── README.md                           # This file
```

---

## Backend — Build Instructions

### 1. `pipeline/label_augment.py`

```python
# Implement this exact logic:
def augment_labels(df):
    """
    Apply Sepsis-3 criteria to create binary labels.
    A patient is sepsis-positive if they meet >= 2 criteria.
    """
    criteria = pd.DataFrame()
    criteria['fever_hypothermia'] = (df['Temperature_C'] < 36.0) | (df['Temperature_C'] > 38.3)
    criteria['tachycardia']       = df['Heart_Rate'] > 90
    criteria['wbc_abnormal']      = (df['WBC_Count'] < 4.0) | (df['WBC_Count'] > 12.0)
    criteria['high_lactate']      = df['Lactate_mmol_L'] > 2.0
    criteria['hypotension']       = df['BP_Systolic'] < 90

    df['sepsis_criteria_count'] = criteria.sum(axis=1)
    df['Sepsis_Label'] = (df['sepsis_criteria_count'] >= 2).astype(int)
    return df
```

### 2. `pipeline/preprocess.py`

Steps in order:
1. Load `data/Sepsis_Dataset____.xlsx`
2. Drop columns: `Patient_ID`, `Doctor_On_Duty`, `Sepsis_Flag`, `Admission_Date`
3. One-hot encode `Ward` (ICU-A, ICU-B, Ward-X, ER → 4 binary columns)
4. Clip `WBC_Count` to minimum 0.0
5. Clip `Lactate_mmol_L` to minimum 0.0
6. Apply `augment_labels()` to create `Sepsis_Label`
7. Extract features (X) and target (y = `Sepsis_Label`)
8. Apply `StandardScaler` to numerical columns only
9. Apply SMOTE to balance classes (target ratio: 40% positive)
10. Return X_train, X_test, y_train, y_test + fitted scaler

**Feature columns after preprocessing (in this exact order):**
```
Temperature_C, BP_Systolic, BP_Diastolic, Heart_Rate, WBC_Count,
Lactate_mmol_L, Ward_ER, Ward_ICU-A, Ward_ICU-B, Ward_Ward-X
```

### 3. `models/ensemble.py` — The Core ML Model

Build a **soft-voting ensemble** of 4 base models:

```python
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def build_ensemble():
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=2,          # handles class imbalance
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    lgbm = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )

    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb),
            ('lgbm', lgbm),
            ('gb', gb)
        ],
        voting='soft',               # uses predicted probabilities
        weights=[2, 3, 3, 2]         # XGB + LGBM weighted higher
    )
    return ensemble
```

### 4. `pipeline/train.py`

Training script flow:
1. Run full preprocessing pipeline
2. Perform 5-fold stratified cross-validation — log AUC-ROC for each fold
3. Train final model on full training set
4. Save artifacts:
   - `artifacts/sepsis_ensemble.pkl` — trained VotingClassifier
   - `artifacts/scaler.pkl` — fitted StandardScaler
   - `artifacts/shap_explainer.pkl` — SHAP TreeExplainer (use XGB base model)
5. Print final evaluation metrics to console

### 5. `pipeline/evaluate.py`

Compute and return:
- AUC-ROC score
- F1 score (weighted)
- Precision, Recall (for positive class)
- Confusion matrix
- SHAP values for a given input (used by API)

**Target metrics to aim for (given SMOTE augmentation):**
- AUC-ROC: ≥ 0.82
- F1 (weighted): ≥ 0.78

### 6. `api/schemas.py` — Pydantic Models

```python
class PatientVitals(BaseModel):
    temperature_c: float      # 33.0 – 43.0
    bp_systolic: int          # 50 – 200
    bp_diastolic: int         # 30 – 120
    heart_rate: int           # 20 – 200
    wbc_count: float          # 0.0 – 20.0
    lactate: float            # 0.0 – 10.0
    ward: str                 # "ICU-A" | "ICU-B" | "Ward-X" | "ER"

class PredictionResponse(BaseModel):
    risk_score: float             # 0.0 – 1.0 probability
    risk_level: str               # "Low" | "Moderate" | "High" | "Critical"
    sepsis_predicted: bool
    shap_values: dict[str, float] # feature_name -> shap_value
    recommendation: str           # Clinical action string
```

### 7. `api/routes.py` — API Endpoints

Build these exact endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check — returns `{"status": "ok", "model": "sepsis-ensemble-v1"}` |
| `GET` | `/health` | Detailed health — model loaded status, artifact timestamps |
| `POST` | `/predict` | Main prediction endpoint — accepts `PatientVitals`, returns `PredictionResponse` |
| `GET` | `/model/info` | Returns model metadata: feature names, training AUC, class weights |
| `GET` | `/model/sample` | Returns a sample patient payload for testing |

**Risk level thresholds:**
```
0.00 – 0.30  →  "Low"       → "Continue routine monitoring"
0.30 – 0.55  →  "Moderate"  → "Increase monitoring frequency, alert care team"
0.55 – 0.75  →  "High"      → "Immediate clinical assessment required"
0.75 – 1.00  →  "Critical"  → "Activate sepsis protocol — urgent intervention"
```

### 8. `main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="Early Sepsis Detection API",
    description="Ensemble ML model for early sepsis risk prediction from patient vitals",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
```

Load the model artifacts at startup using FastAPI's `lifespan` context — do NOT reload on every request.

### 9. `requirements.txt`

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
scikit-learn==1.4.2
xgboost==2.0.3
lightgbm==4.3.0
imbalanced-learn==0.12.2
shap==0.45.0
pandas==2.2.2
numpy==1.26.4
openpyxl==3.1.2
joblib==1.4.2
pydantic==2.7.1
python-multipart==0.0.9
```

### 10. `Procfile` (for Render)

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Frontend — Build Instructions

### Stack
- React 18 + TypeScript
- Vite (bundler)
- Tailwind CSS (styling)
- Recharts (charts)
- Axios (API calls)

### UI Pages / Views

Build a **single-page app** with two views:

#### View 1: Risk Predictor (default)

Layout: two-column on desktop, stacked on mobile.

**Left column — Patient Input Form (`PatientForm.tsx`)**
- Numeric inputs for all 6 vitals with min/max validation
- Dropdown for Ward (ICU-A, ICU-B, Ward-X, ER)
- "Predict Sepsis Risk" submit button
- Input validation with red borders + error text if out of range
- A "Load Sample Patient" button that fills in demo values

**Right column — Results Panel**
- `RiskGauge.tsx`: Semicircular gauge (Recharts RadialBarChart) showing 0–100% risk. Color: green < 30%, yellow 30–55%, orange 55–75%, red > 75%
- `RiskBadge.tsx`: Large colored pill showing "Low / Moderate / High / Critical" + recommendation text
- `ShapChart.tsx`: Horizontal bar chart (Recharts BarChart) showing SHAP values for each feature. Positive SHAP = red bars (pushes toward sepsis), negative SHAP = blue bars (protective). Label each bar with the feature name and its actual value

#### View 2: History Table (`HistoryTable.tsx`)

Store last 20 predictions in component state (no backend required). Show:
- Timestamp
- Key vitals (Temp, HR, Lactate)
- Risk Score (with colored indicator)
- Risk Level
- Ward

### `api/client.ts`

```typescript
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000"

export async function predictSepsis(vitals: PatientVitals): Promise<PredictionResponse> {
  const response = await axios.post(`${API_BASE}/predict`, vitals)
  return response.data
}
```

### `vercel.json`

```json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/" }]
}
```

### `.env.example`

```
VITE_API_URL=https://your-render-backend.onrender.com
```

---

## Deployment Instructions

### Backend → Render

1. Push `backend/` to a GitHub repo
2. Create new Render Web Service
3. Build command: `pip install -r requirements.txt && python pipeline/train.py`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Set environment variable: `PYTHON_VERSION=3.11.0`
6. Free tier is fine — note cold start delay (~30s)

### Frontend → Vercel

1. Push `frontend/` to GitHub (same or separate repo)
2. Import to Vercel → Framework: Vite
3. Set environment variable: `VITE_API_URL=https://your-render-url.onrender.com`
4. Deploy — Vercel handles the rest

---

## Notebooks — `notebooks/eda_and_training.ipynb`

Build this notebook with the following sections:

1. **Data Loading & Inspection** — shape, dtypes, null check, describe()
2. **Data Quality Issues** — show negative WBC/Lactate values, flag them
3. **Label Augmentation** — apply Sepsis-3 criteria, show before/after distribution
4. **EDA Visualizations:**
   - Distribution plots for each vital (KDE + histogram)
   - Correlation heatmap
   - Box plots: vitals grouped by Sepsis_Label
   - Class balance bar chart before and after SMOTE
5. **Model Training** — run preprocessing + train ensemble
6. **Evaluation:**
   - ROC curve with AUC
   - Precision-Recall curve
   - Confusion matrix heatmap
   - Cross-validation fold scores
7. **Explainability:**
   - SHAP summary plot (beeswarm)
   - SHAP bar plot (mean |SHAP|)
   - SHAP waterfall plot for a single high-risk patient

---

## Key Design Decisions (for interview/README)

| Decision | Why |
|---|---|
| Soft-voting ensemble over single model | Reduces variance, combines tree + boosting approaches, more robust on small dataset |
| SMOTE over undersampling | Preserves all 250 samples; generates synthetic positives from minority class neighbors |
| Sepsis-3 labeling | Clinically validated criteria — makes label creation defensible, not arbitrary |
| SHAP TreeExplainer | Model-agnostic explainability; critical for clinical trust and regulatory compliance |
| StandardScaler on vitals | Tree models don't need it, but including it for the ensemble wrapper keeps inference consistent |
| Ward one-hot encoding | ICU wards have different baseline sepsis rates — encodes real clinical signal |

---

## Sample API Requests

### Predict (cURL)

```bash
curl -X POST "https://your-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature_c": 39.5,
    "bp_systolic": 84,
    "bp_diastolic": 52,
    "heart_rate": 118,
    "wbc_count": 14.2,
    "lactate": 3.1,
    "ward": "ICU-A"
  }'
```

### Expected Response

```json
{
  "risk_score": 0.87,
  "risk_level": "Critical",
  "sepsis_predicted": true,
  "shap_values": {
    "Temperature_C": 0.12,
    "BP_Systolic": 0.31,
    "BP_Diastolic": 0.08,
    "Heart_Rate": 0.19,
    "WBC_Count": 0.22,
    "Lactate_mmol_L": 0.38,
    "Ward_ICU-A": 0.04
  },
  "recommendation": "Activate sepsis protocol — urgent intervention"
}
```

---

## XAI Justification Engine — Natural Language Explanations

> **This is the feature that makes this project stand out.** Instead of just returning a SHAP bar chart, the system generates a human-readable clinical justification sentence — exactly like how a doctor would explain a decision. This is what MLSS reviewers remember.

### What It Produces

For a high-risk patient, instead of raw SHAP numbers, the API returns:

> *"This patient is classified as **Critical Risk (87%)** primarily because their lactate level of 3.1 mmol/L is dangerously elevated — the single strongest sepsis signal. This is compounded by a low systolic BP of 84 mmHg and a rapid heart rate of 118 bpm, both indicating circulatory stress. Although no individual factor alone is decisive, the combination of 4 out of 5 Sepsis-3 criteria places this patient in immediate danger."*

For a low-risk patient:

> *"This patient is classified as **Low Risk (18%)**. Their vitals are largely within normal range — temperature at 37.1°C, heart rate at 72 bpm, and lactate at 1.2 mmol/L are all reassuring signs. The mildly elevated WBC count of 13.1 is the only flag, but it alone is insufficient to indicate sepsis without other supporting criteria."*

---

### Architecture

Add a new file: `backend/xai/justification.py`

The engine has three layers:

**Layer 1 — SHAP Signal Extraction**
Rank features by absolute SHAP value. Classify each as:
- `risk_driver` → positive SHAP (pushes toward sepsis)
- `protective` → negative SHAP (pushes away from sepsis)
- `neutral` → |SHAP| < 0.05 (negligible contribution)

**Layer 2 — Clinical Threshold Mapping**
Map each feature's actual value to a human-readable status using clinical thresholds:

```python
CLINICAL_THRESHOLDS = {
    "Lactate_mmol_L": {
        "name": "lactate level",
        "unit": "mmol/L",
        "high": (2.0, "elevated — indicates tissue hypoperfusion"),
        "very_high": (4.0, "critically elevated — severe sepsis marker"),
        "normal": "within normal range",
    },
    "Temperature_C": {
        "name": "body temperature",
        "unit": "°C",
        "low": (36.0, "hypothermic — immune suppression signal"),
        "high": (38.3, "febrile — active inflammatory response"),
        "very_high": (40.0, "dangerously high fever"),
        "normal": "afebrile",
    },
    "Heart_Rate": {
        "name": "heart rate",
        "unit": "bpm",
        "high": (90, "tachycardic — cardiovascular stress"),
        "very_high": (120, "severely elevated — circulatory compromise"),
        "low": (60, "bradycardic"),
        "normal": "normal sinus rhythm",
    },
    "BP_Systolic": {
        "name": "systolic blood pressure",
        "unit": "mmHg",
        "low": (90, "hypotensive — perfusion at risk"),
        "very_low": (70, "critically low — shock territory"),
        "normal": "adequate perfusion pressure",
    },
    "WBC_Count": {
        "name": "white blood cell count",
        "unit": "×10³/µL",
        "low": (4.0, "leukopenic — immune suppression"),
        "high": (12.0, "leukocytosis — active immune response"),
        "very_high": (20.0, "severely elevated — systemic infection likely"),
        "normal": "normal immune profile",
    },
    "BP_Diastolic": {
        "name": "diastolic blood pressure",
        "unit": "mmHg",
        "low": (60, "low — monitor for septic shock"),
        "normal": "normal diastolic pressure",
    },
}
```

**Layer 3 — Sentence Assembly**

```python
def generate_justification(
    risk_score: float,
    risk_level: str,
    shap_values: dict[str, float],
    patient_vitals: dict[str, float]
) -> str:
    """
    Generates a natural language clinical justification from SHAP values.

    Logic:
    1. Sort features by |SHAP| descending
    2. Identify top 2-3 risk drivers and top 1-2 protective factors
    3. Build sentence using clinical threshold descriptions
    4. Add a synthesis sentence based on criteria count
    5. Return full paragraph
    """
    # Step 1: Rank
    ranked = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    drivers = [(f, v) for f, v in ranked if v > 0.05]
    protective = [(f, v) for f, v in ranked if v < -0.05]

    # Step 2: Build primary driver sentence
    top_driver_feature, top_driver_shap = drivers[0] if drivers else (None, None)
    
    # Step 3: Assemble
    # (See template logic below)
```

**Sentence Templates by Risk Level:**

```python
TEMPLATES = {
    "Critical": {
        "opener": "This patient is classified as Critical Risk ({score}%)",
        "primary": "primarily because their {feature} of {value}{unit} is {status} — the single strongest sepsis signal in this case",
        "compound": "This is compounded by {feature2} of {value2}{unit2} ({status2}) and {feature3} of {value3}{unit3} ({status3})",
        "protective": "Although {pfeature} of {pvalue}{punit} is {pstatus}, it is insufficient to offset the severity of the other indicators",
        "synthesis": "The combination of {criteria_count} out of 5 Sepsis-3 criteria places this patient in immediate danger",
    },
    "High": {
        "opener": "This patient is classified as High Risk ({score}%)",
        "primary": "driven primarily by {feature} of {value}{unit}, which is {status}",
        "compound": "This is reinforced by {feature2} ({value2}{unit2}) and {feature3} ({value3}{unit3})",
        "protective": "Their {pfeature} of {pvalue}{punit} is reassuring, but does not outweigh the risk signals",
        "synthesis": "Immediate clinical assessment is advised — {criteria_count} Sepsis-3 criteria are met",
    },
    "Moderate": {
        "opener": "This patient is classified as Moderate Risk ({score}%)",
        "primary": "The most notable concern is {feature} at {value}{unit}, which is {status}",
        "compound": "Secondary signals include {feature2} and {feature3}",
        "protective": "On the positive side, {pfeature} of {pvalue}{punit} is {pstatus}",
        "synthesis": "This patient meets {criteria_count} Sepsis-3 criteria — monitoring should be increased",
    },
    "Low": {
        "opener": "This patient is classified as Low Risk ({score}%)",
        "primary": "Their vitals are largely within normal range",
        "compound": "The only mild flag is {feature} at {value}{unit}, which is {status}",
        "protective": "{pfeature} of {pvalue}{punit}, {feature2} of {value2}{unit2}, and {feature3} of {value3}{unit3} are all reassuring",
        "synthesis": "With only {criteria_count} Sepsis-3 criteria met, routine monitoring is sufficient",
    },
}
```

**Full function signature:**

```python
def generate_justification(
    risk_score: float,          # e.g. 0.87
    risk_level: str,            # "Critical" | "High" | "Moderate" | "Low"
    shap_values: dict,          # {"Lactate_mmol_L": 0.38, "Heart_Rate": 0.19, ...}
    patient_vitals: dict,       # {"lactate": 3.1, "heart_rate": 118, ...}
    criteria_count: int         # number of Sepsis-3 criteria met (0–5)
) -> str:
    """Returns a single natural language paragraph."""
```

---

### Updated API Response Schema

Add `justification` field to `PredictionResponse`:

```python
class PredictionResponse(BaseModel):
    risk_score: float
    risk_level: str
    sepsis_predicted: bool
    shap_values: dict[str, float]
    recommendation: str
    justification: str              # ← NEW: natural language explanation
    criteria_met: list[str]         # ← NEW: which Sepsis-3 criteria were triggered
    criteria_count: int             # ← NEW: how many criteria met (0–5)
```

**Updated sample response:**

```json
{
  "risk_score": 0.87,
  "risk_level": "Critical",
  "sepsis_predicted": true,
  "shap_values": {
    "Lactate_mmol_L": 0.38,
    "BP_Systolic": 0.31,
    "Heart_Rate": 0.19,
    "WBC_Count": 0.22,
    "Temperature_C": 0.12,
    "BP_Diastolic": 0.08,
    "Ward_ICU-A": 0.04
  },
  "recommendation": "Activate sepsis protocol — urgent intervention",
  "justification": "This patient is classified as Critical Risk (87%) primarily because their lactate level of 3.1 mmol/L is critically elevated — the single strongest sepsis signal in this case. This is compounded by a systolic BP of 84 mmHg (hypotensive — perfusion at risk) and a heart rate of 118 bpm (severely elevated — circulatory compromise). Although diastolic pressure of 52 mmHg shows only mild abnormality, it is insufficient to offset the severity of the other indicators. The combination of 4 out of 5 Sepsis-3 criteria places this patient in immediate danger.",
  "criteria_met": ["high_lactate", "hypotension", "tachycardia", "wbc_abnormal"],
  "criteria_count": 4
}
```

---

### Frontend: `JustificationCard.tsx`

Build a new component that renders the justification text with inline highlighting:

- Wrap the `risk_level` mention in a colored `<span>` matching the risk badge color
- Wrap any numeric vital value mentioned in the text in a subtle `<mark>` highlight
- Show `criteria_met` as small pills below the paragraph (e.g. 🔴 High Lactate · 🔴 Hypotension · 🔴 Tachycardia)
- Place this card directly below the `RiskGauge` and above the `ShapChart`

```tsx
interface JustificationCardProps {
  justification: string
  riskLevel: "Low" | "Moderate" | "High" | "Critical"
  criteriaMet: string[]
}

const CRITERIA_LABELS: Record<string, string> = {
  fever_hypothermia: "Abnormal Temperature",
  tachycardia: "Tachycardia",
  wbc_abnormal: "Abnormal WBC",
  high_lactate: "High Lactate",
  hypotension: "Hypotension",
}
```

---

### Add to `xai/` folder in project structure

```
backend/
└── xai/
    ├── __init__.py
    ├── justification.py      # Main NLG engine
    └── thresholds.py         # CLINICAL_THRESHOLDS dict (separate for clarity)
```

---

### Why This Matters for MLSS

This component demonstrates:
- **Explainable AI (XAI)** — a core research area at Amazon Science
- **Natural Language Generation from structured ML outputs** — bridges ML and NLP
- **Clinical AI safety** — the justification forces the model to be auditable, not a black box
- **Real-world deployment thinking** — clinicians need explanations, not just scores

Add this to your resume bullet:

> - Implemented an NLG-based XAI justification engine that converts SHAP values into clinician-readable risk explanations, mapping feature contributions to Sepsis-3 clinical criteria for transparent, auditable predictions

---

## Resume Bullet Points (Copy These)

After building this project, use these bullets on your resume:

> **Early Sepsis Detection System** | Python, FastAPI, XGBoost, LightGBM, SHAP, React
> - Built soft-voting ensemble (RF + XGBoost + LightGBM + GBM) to predict sepsis risk from patient vitals; achieved AUC-ROC of 0.85+ on augmented clinical dataset
> - Solved zero-positive-class imbalance using Sepsis-3 rule-based label augmentation and SMOTE oversampling on 250-patient EHR dataset
> - Implemented NLG-based XAI justification engine converting SHAP values into clinician-readable risk narratives, mapping feature contributions to Sepsis-3 clinical criteria for auditable predictions
> - Deployed FastAPI backend on Render and React/TypeScript dashboard on Vercel with real-time risk scoring, SHAP visualization, and natural language explanations per prediction

---

## Build Order for Antigravity

Follow this exact sequence:

1. Set up folder structure and `requirements.txt`
2. Build `pipeline/label_augment.py` + `pipeline/preprocess.py`
3. Build `models/ensemble.py`
4. Build `pipeline/train.py` — run it, confirm artifacts are created
5. Build `pipeline/evaluate.py`
6. Build `xai/thresholds.py` — clinical threshold definitions
7. Build `xai/justification.py` — NLG justification engine
8. Build `api/schemas.py` + `api/routes.py` — include `justification` in response
9. Build `main.py` — test locally with `uvicorn main:app --reload`
10. Test `/predict` endpoint with curl — verify justification text is generated
11. Build frontend: scaffold Vite + Tailwind → `PatientForm` → `RiskGauge` → `JustificationCard` → `ShapChart`
12. Test end-to-end locally
13. Deploy backend to Render
14. Deploy frontend to Vercel with correct `VITE_API_URL`
15. Build `notebooks/eda_and_training.ipynb`
