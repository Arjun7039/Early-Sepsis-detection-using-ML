# 🩺 Early Sepsis Detection System

A production-ready machine learning system that predicts sepsis risk from patient vitals data. The system ingests clinical features (temperature, blood pressure, heart rate, WBC count, lactate), runs a soft-voting ensemble ML model, and serves a real-time risk score via a FastAPI backend and a React dashboard frontend.

## 🌟 Key Features
- **Ensemble ML Model:** Soft-voting ensemble combining Random Forest, XGBoost, LightGBM, and Gradient Boosting for robust predictions on imbalanced clinical data.
- **Clinical Feature Engineering:** Dynamically computes Mean Arterial Pressure (MAP), Shock Index, Modified Shock Index, Pulse Pressure, and Temperature Deviation based on Sepsis-3 criteria and PhysioNet 2019 guidelines.
- **XAI Justification Engine:** Generates natural language clinical narratives explaining *why* a risk score was given, translating complex SHAP feature importance values into clinician-readable sentences.
- **SMOTE Balancing:** Handles extreme class imbalances common in medical datasets by synthetically oversampling positive sepsis cases.
- **Full-Stack Dashboard:** A React (TypeScript) frontend built with Tailwind CSS and Recharts to visualize patient risk, SHAP values, and historical predictions in real-time.

## 🛠️ Tech Stack
- **Backend:** Python 3.11, FastAPI, scikit-learn, XGBoost, LightGBM, SHAP, Uvicorn
- **Frontend:** React 18, TypeScript, Vite, Tailwind CSS, Recharts
- **Data Pipeline:** Pandas, NumPy, imbalanced-learn (SMOTE)

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Arjun7039/Early-Sepsis-detection-using-ML.git
cd Early-Sepsis-detection-using-ML
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

# Train the model (generates artifacts and metrics)
python pipeline/train.py

# Start the FastAPI server
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`.

### 3. Frontend Setup
In a new terminal window:
```bash
cd frontend
npm install

# Start the Vite development server
npm run dev
```
The dashboard will be available at `http://localhost:5173`.

## 🧠 Architecture Overview
The system architecture revolves around an end-to-end ML pipeline integrated directly into a web application:
1. **Data Preprocessing & Training:** Raw patient data is cleaned, labeled using medical heuristics (Sepsis-3), augmented using SMOTE, and passed through an ensemble classifier.
2. **FastAPI Backend:** Exposes a `/predict` endpoint that engineers clinical features on the fly, scales them, runs model inference, extracts SHAP values, and builds a natural language justification using the `XAI` module.
3. **React Dashboard:** Consumes the API and renders interactive charts (Risk Gauge, SHAP Bar Chart) and clinical alerts.

## 📊 Notebooks
Explore the detailed Exploratory Data Analysis (EDA) and Model Training process in the Jupyter notebook:
[`notebooks/eda_and_training.ipynb`](notebooks/eda_and_training.ipynb)

## 📄 License
This project is open-source and available under the MIT License.
