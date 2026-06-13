"""
Clinical Thresholds for Sepsis-3 Feature Interpretation

Maps each clinical feature to human-readable status descriptions
based on medically validated ranges. Used by the justification
engine to convert raw values into clinical language.
"""

CLINICAL_THRESHOLDS = {
    "Lactate_mmol_L": {
        "name": "lactate level",
        "unit": " mmol/L",
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
        "unit": " bpm",
        "high": (90, "tachycardic — cardiovascular stress"),
        "very_high": (120, "severely elevated — circulatory compromise"),
        "low": (60, "bradycardic"),
        "normal": "normal sinus rhythm",
    },
    "BP_Systolic": {
        "name": "systolic blood pressure",
        "unit": " mmHg",
        "low": (90, "hypotensive — perfusion at risk"),
        "very_low": (70, "critically low — shock territory"),
        "normal": "adequate perfusion pressure",
    },
    "WBC_Count": {
        "name": "white blood cell count",
        "unit": " ×10³/µL",
        "low": (4.0, "leukopenic — immune suppression"),
        "high": (12.0, "leukocytosis — active immune response"),
        "very_high": (20.0, "severely elevated — systemic infection likely"),
        "normal": "normal immune profile",
    },
    "BP_Diastolic": {
        "name": "diastolic blood pressure",
        "unit": " mmHg",
        "low": (60, "low — monitor for septic shock"),
        "normal": "normal diastolic pressure",
    },
    "MAP": {
        "name": "mean arterial pressure",
        "unit": " mmHg",
        "low": (65.0, "hypotensive — risk of organ hypoperfusion"),
        "normal": "adequate tissue perfusion pressure",
    },
    "Shock_Index": {
        "name": "shock index",
        "unit": "",
        "high": (0.7, "elevated — early warning of hemodynamic instability"),
        "very_high": (1.0, "dangerously elevated — indicative of shock state"),
        "normal": "stable hemodynamic index",
    },
    "Modified_Shock_Index": {
        "name": "modified shock index",
        "unit": "",
        "high": (1.3, "elevated — cardiovascular compromise"),
        "normal": "reassuring modified shock index",
    },
    "Pulse_Pressure": {
        "name": "pulse pressure",
        "unit": " mmHg",
        "low": (30.0, "narrow — decreased stroke volume/shock risk"),
        "normal": "normal pulse pressure",
    },
    "Temp_Deviation": {
        "name": "temperature deviation",
        "unit": "°C",
        "high": (1.0, "abnormal body temperature variation"),
        "normal": "stable body temperature",
    },
}


# Feature name mapping: API field names → model feature names
API_TO_MODEL_FEATURE_MAP = {
    "temperature_c": "Temperature_C",
    "bp_systolic": "BP_Systolic",
    "bp_diastolic": "BP_Diastolic",
    "heart_rate": "Heart_Rate",
    "wbc_count": "WBC_Count",
    "lactate": "Lactate_mmol_L",
    "map": "MAP",
    "shock_index": "Shock_Index",
    "modified_shock_index": "Modified_Shock_Index",
    "pulse_pressure": "Pulse_Pressure",
    "temp_deviation": "Temp_Deviation",
}


def get_clinical_status(feature_name: str, value: float) -> str:
    """
    Get the clinical status description for a feature value.

    Args:
        feature_name: Model feature name (e.g., 'Temperature_C')
        value: The actual feature value

    Returns:
        Human-readable clinical status string
    """
    thresholds = CLINICAL_THRESHOLDS.get(feature_name)
    if thresholds is None:
        return "no clinical threshold defined"

    # Check very_high / very_low first (more specific)
    if "very_high" in thresholds and value >= thresholds["very_high"][0]:
        return thresholds["very_high"][1]
    if "very_low" in thresholds and value <= thresholds["very_low"][0]:
        return thresholds["very_low"][1]

    # Then check high / low
    if "high" in thresholds and value >= thresholds["high"][0]:
        return thresholds["high"][1]
    if "low" in thresholds and value <= thresholds["low"][0]:
        return thresholds["low"][1]

    return thresholds.get("normal", "within normal range")


def get_feature_display_name(feature_name: str) -> str:
    """Get the human-readable name for a feature."""
    thresholds = CLINICAL_THRESHOLDS.get(feature_name)
    if thresholds:
        return thresholds["name"]
    # Handle ward columns
    if feature_name.startswith("Ward_"):
        return f"ward ({feature_name.replace('Ward_', '')})"
    return feature_name


def get_feature_unit(feature_name: str) -> str:
    """Get the unit string for a feature."""
    thresholds = CLINICAL_THRESHOLDS.get(feature_name)
    if thresholds:
        return thresholds.get("unit", "")
    return ""
