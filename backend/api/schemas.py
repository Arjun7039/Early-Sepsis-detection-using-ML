"""
Pydantic Request/Response Models

Defines the data schemas for API validation:
- PatientVitals: Input validation with clinical ranges
- PredictionResponse: Structured prediction output
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional


class PatientVitals(BaseModel):
    """Input schema for patient vital signs."""
    temperature_c: float = Field(
        ...,
        ge=33.0, le=43.0,
        description="Body temperature in Celsius (33.0 – 43.0)"
    )
    bp_systolic: int = Field(
        ...,
        ge=50, le=200,
        description="Systolic blood pressure in mmHg (50 – 200)"
    )
    bp_diastolic: int = Field(
        ...,
        ge=30, le=120,
        description="Diastolic blood pressure in mmHg (30 – 120)"
    )
    heart_rate: int = Field(
        ...,
        ge=20, le=200,
        description="Heart rate in bpm (20 – 200)"
    )
    wbc_count: float = Field(
        ...,
        ge=0.0, le=20.0,
        description="White blood cell count ×10³/µL (0.0 – 20.0)"
    )
    lactate: float = Field(
        ...,
        ge=0.0, le=10.0,
        description="Serum lactate in mmol/L (0.0 – 10.0)"
    )
    ward: str = Field(
        ...,
        description='Hospital ward: "ICU-A" | "ICU-B" | "Ward-X" | "ER"'
    )

    class Config:
        json_schema_extra = {
            "example": {
                "temperature_c": 39.5,
                "bp_systolic": 84,
                "bp_diastolic": 52,
                "heart_rate": 118,
                "wbc_count": 14.2,
                "lactate": 3.1,
                "ward": "ICU-A"
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for sepsis risk prediction."""
    risk_score: float = Field(
        ...,
        description="Sepsis probability score (0.0 – 1.0)"
    )
    risk_level: str = Field(
        ...,
        description='Risk classification: "Low" | "Moderate" | "High" | "Critical"'
    )
    sepsis_predicted: bool = Field(
        ...,
        description="Whether sepsis is predicted (risk_score >= 0.5)"
    )
    shap_values: dict[str, float] = Field(
        ...,
        description="SHAP feature importance values"
    )
    recommendation: str = Field(
        ...,
        description="Clinical action recommendation"
    )
    justification: str = Field(
        ...,
        description="Natural language clinical justification"
    )
    criteria_met: list[str] = Field(
        ...,
        description="Which Sepsis-3 criteria were triggered"
    )
    criteria_count: int = Field(
        ...,
        description="Number of Sepsis-3 criteria met (0–5)"
    )


class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    scaler_loaded: bool
    shap_loaded: bool
    model_name: str
    version: str


class ModelInfoResponse(BaseModel):
    """Response for /model/info endpoint."""
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_type: str
    base_models: list[dict]
    voting: str
    feature_names: list[str]
    training_auc: Optional[float] = None
    training_samples: Optional[int] = None
