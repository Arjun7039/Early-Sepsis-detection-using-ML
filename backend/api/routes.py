"""
API Routes — All Endpoints for the Sepsis Detection System

Endpoints:
  GET  /           — Health check
  GET  /health     — Detailed health with artifact status
  POST /predict    — Main prediction (accepts PatientVitals → PredictionResponse)
  GET  /model/info — Model metadata
  GET  /model/sample — Sample patient payload for testing
"""

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from fastapi import APIRouter, HTTPException

from api.schemas import (
    PatientVitals,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from pipeline.preprocess import FEATURE_COLUMNS, NUMERICAL_COLUMNS
from pipeline.evaluate import compute_shap_values
from pipeline.label_augment import get_criteria_met
from models.ensemble import get_model_info
from xai.justification import generate_justification


router = APIRouter()

# ── Global State (loaded at startup via lifespan) ──────────
_model = None
_scaler = None
_explainer = None
_training_info = None

ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'artifacts'
)


def load_artifacts():
    """Load trained model artifacts into memory."""
    global _model, _scaler, _explainer, _training_info

    model_path = os.path.join(ARTIFACTS_DIR, 'sepsis_ensemble.pkl')
    scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
    shap_path = os.path.join(ARTIFACTS_DIR, 'shap_explainer.pkl')
    metrics_path = os.path.join(ARTIFACTS_DIR, 'training_metrics.pkl')

    if os.path.exists(model_path):
        _model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")
    else:
        print(f"⚠️  Model not found at {model_path}")

    if os.path.exists(scaler_path):
        _scaler = joblib.load(scaler_path)
        print(f"✅ Scaler loaded from {scaler_path}")
    else:
        print(f"⚠️  Scaler not found at {scaler_path}")

    if os.path.exists(shap_path):
        _explainer = joblib.load(shap_path)
        print(f"✅ SHAP explainer loaded from {shap_path}")
    else:
        print(f"⚠️  SHAP explainer not found at {shap_path}")

    if os.path.exists(metrics_path):
        _training_info = joblib.load(metrics_path)
        print(f"✅ Training info loaded from {metrics_path}")


# ── Risk Level Classification ─────────────────────────────

RISK_THRESHOLDS = [
    (0.75, "Critical", "Activate sepsis protocol — urgent intervention"),
    (0.55, "High", "Immediate clinical assessment required"),
    (0.30, "Moderate", "Increase monitoring frequency, alert care team"),
    (0.00, "Low", "Continue routine monitoring"),
]


def classify_risk(score: float) -> tuple[str, str]:
    """Map risk score to risk level and recommendation."""
    for threshold, level, recommendation in RISK_THRESHOLDS:
        if score >= threshold:
            return level, recommendation
    return "Low", "Continue routine monitoring"


# ── Endpoint: Health Check ─────────────────────────────────

@router.get("/", tags=["Health"])
async def root():
    """Basic health check."""
    return {"status": "ok", "model": "sepsis-ensemble-v1"}


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Detailed health check with artifact status."""
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_loaded=_model is not None,
        scaler_loaded=_scaler is not None,
        shap_loaded=_explainer is not None,
        model_name="sepsis-ensemble-v1",
        version="1.0.0"
    )


# ── Endpoint: Predict ─────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(vitals: PatientVitals):
    """
    Predict sepsis risk from patient vitals.

    Returns risk score, risk level, SHAP explainability values,
    clinical recommendation, and natural language justification.
    """
    if _model is None or _scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training first: python pipeline/train.py"
        )

    # ── Prepare feature vector ─────────────────────────
    # Map ward to one-hot encoding
    ward_map = {
        'ER': [1, 0, 0, 0],
        'ICU-A': [0, 1, 0, 0],
        'ICU-B': [0, 0, 1, 0],
        'Ward-X': [0, 0, 0, 1],
    }

    if vitals.ward not in ward_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ward: {vitals.ward}. Must be one of: {list(ward_map.keys())}"
        )

    # Dynamic feature engineering for the single prediction
    map_val = vitals.bp_diastolic + (vitals.bp_systolic - vitals.bp_diastolic) / 3
    shock_index = vitals.heart_rate / (vitals.bp_systolic if vitals.bp_systolic != 0 else 1)
    modified_shock_index = vitals.heart_rate / (map_val if map_val != 0 else 1)
    pulse_pressure = vitals.bp_systolic - vitals.bp_diastolic
    temp_deviation = abs(vitals.temperature_c - 37.0)

    # Build numerical features (same order as NUMERICAL_COLUMNS)
    numerical_values = np.array([[
        vitals.temperature_c,
        vitals.bp_systolic,
        vitals.bp_diastolic,
        vitals.heart_rate,
        vitals.wbc_count,
        vitals.lactate,
        map_val,
        shock_index,
        modified_shock_index,
        pulse_pressure,
        temp_deviation,
    ]])

    # Scale numerical features
    scaled_numerical = _scaler.transform(numerical_values)

    # Combine with ward one-hot encoding
    ward_encoded = np.array([ward_map[vitals.ward]])
    features = np.concatenate([scaled_numerical, ward_encoded], axis=1)

    # ── Get prediction ─────────────────────────────────
    risk_score = float(_model.predict_proba(features)[0][1])
    sepsis_predicted = risk_score >= 0.5
    risk_level, recommendation = classify_risk(risk_score)

    # ── Compute SHAP values ────────────────────────────
    shap_values = {}
    if _explainer is not None:
        try:
            shap_values = compute_shap_values(
                _explainer, features, FEATURE_COLUMNS
            )
        except Exception as e:
            print(f"⚠️  SHAP computation failed: {e}")
            shap_values = {col: 0.0 for col in FEATURE_COLUMNS}
    else:
        shap_values = {col: 0.0 for col in FEATURE_COLUMNS}

    # ── Evaluate Sepsis-3 criteria ─────────────────────
    vitals_dict = {
        'temperature_c': vitals.temperature_c,
        'bp_systolic': vitals.bp_systolic,
        'bp_diastolic': vitals.bp_diastolic,
        'heart_rate': vitals.heart_rate,
        'wbc_count': vitals.wbc_count,
        'lactate': vitals.lactate,
        'map': map_val,
        'shock_index': shock_index,
        'modified_shock_index': modified_shock_index,
        'pulse_pressure': pulse_pressure,
        'temp_deviation': temp_deviation,
    }
    criteria_met, criteria_count = get_criteria_met(vitals_dict)

    # ── Generate NLG justification ─────────────────────
    justification = generate_justification(
        risk_score=risk_score,
        risk_level=risk_level,
        shap_values=shap_values,
        patient_vitals=vitals_dict,
        criteria_count=criteria_count
    )

    return PredictionResponse(
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        sepsis_predicted=sepsis_predicted,
        shap_values=shap_values,
        recommendation=recommendation,
        justification=justification,
        criteria_met=criteria_met,
        criteria_count=criteria_count
    )


# ── Endpoint: Model Info ──────────────────────────────────

@router.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Return model metadata and training statistics."""
    info = get_model_info()

    training_auc = None
    training_samples = None

    if _training_info:
        metrics = _training_info.get('metrics', {})
        training_auc = metrics.get('auc_roc')
        training_samples = _training_info.get('training_samples')

    return ModelInfoResponse(
        model_name=info['model_name'],
        model_type=info['model_type'],
        base_models=info['base_models'],
        voting=info['voting'],
        feature_names=FEATURE_COLUMNS,
        training_auc=training_auc,
        training_samples=training_samples,
    )


# ── Endpoint: Sample Patient ─────────────────────────────

@router.get("/model/sample", tags=["Model"])
async def sample_patient():
    """Return a sample patient payload for testing."""
    return {
        "sample_high_risk": {
            "temperature_c": 39.5,
            "bp_systolic": 84,
            "bp_diastolic": 52,
            "heart_rate": 118,
            "wbc_count": 14.2,
            "lactate": 3.1,
            "ward": "ICU-A"
        },
        "sample_low_risk": {
            "temperature_c": 37.1,
            "bp_systolic": 122,
            "bp_diastolic": 78,
            "heart_rate": 72,
            "wbc_count": 7.5,
            "lactate": 1.2,
            "ward": "Ward-X"
        }
    }
