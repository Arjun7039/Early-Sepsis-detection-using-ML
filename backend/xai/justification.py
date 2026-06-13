"""
XAI Justification Engine — Natural Language Explanations

Converts SHAP values and patient vitals into human-readable clinical
justification paragraphs. This is the standout feature that bridges
ML predictions and clinical decision-making.

Three-layer architecture:
1. SHAP Signal Extraction — rank features, classify as driver/protective/neutral
2. Clinical Threshold Mapping — map values to clinical descriptions
3. Sentence Assembly — build risk-level-specific paragraphs
"""

from xai.thresholds import (
    CLINICAL_THRESHOLDS,
    API_TO_MODEL_FEATURE_MAP,
    get_clinical_status,
    get_feature_display_name,
    get_feature_unit
)


# ── Sentence Templates by Risk Level ──────────────────────────

TEMPLATES = {
    "Critical": {
        "opener": "This patient is classified as Critical Risk ({score}%)",
        "primary": "primarily because their {feature} of {value}{unit} is {status} — the single strongest sepsis signal in this case",
        "compound": "This is compounded by a {feature2} of {value2}{unit2} ({status2}) and a {feature3} of {value3}{unit3} ({status3})",
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


def _get_vital_value(patient_vitals: dict, model_feature_name: str) -> float:
    """
    Get the patient's actual vital value for a model feature name.
    Handles both API-style and model-style key names.
    """
    # Direct model feature name match
    if model_feature_name in patient_vitals:
        return patient_vitals[model_feature_name]

    # Try reverse mapping (model name → API name)
    reverse_map = {v: k for k, v in API_TO_MODEL_FEATURE_MAP.items()}
    api_name = reverse_map.get(model_feature_name)
    if api_name and api_name in patient_vitals:
        return patient_vitals[api_name]

    return 0.0


def generate_justification(
    risk_score: float,
    risk_level: str,
    shap_values: dict,
    patient_vitals: dict,
    criteria_count: int
) -> str:
    """
    Generate a natural language clinical justification from SHAP values.

    Args:
        risk_score: Probability score (0.0 – 1.0)
        risk_level: "Critical" | "High" | "Moderate" | "Low"
        shap_values: Dict mapping feature names to SHAP values
        patient_vitals: Dict with patient's actual vital values
        criteria_count: Number of Sepsis-3 criteria met (0–5)

    Returns:
        A single natural language paragraph explaining the prediction.
    """
    score_pct = int(risk_score * 100)
    templates = TEMPLATES.get(risk_level, TEMPLATES["Moderate"])

    # ── Layer 1: SHAP Signal Extraction ────────────────────
    # Filter out ward features for the main narrative (low clinical relevance)
    clinical_shap = {
        k: v for k, v in shap_values.items()
        if not k.startswith("Ward_")
    }

    ranked = sorted(clinical_shap.items(), key=lambda x: abs(x[1]), reverse=True)
    drivers = [(f, v) for f, v in ranked if v > 0.05]
    protective = [(f, v) for f, v in ranked if v < -0.05]
    # neutral features: |SHAP| < 0.05 — not mentioned in narrative

    # ── Layer 2: Clinical Threshold Mapping ────────────────
    def _feature_context(feature_name: str) -> dict:
        """Build context dict for a feature."""
        value = _get_vital_value(patient_vitals, feature_name)
        return {
            "feature": get_feature_display_name(feature_name),
            "value": f"{value:.1f}" if isinstance(value, float) else str(value),
            "unit": get_feature_unit(feature_name),
            "status": get_clinical_status(feature_name, value),
        }

    # ── Layer 3: Sentence Assembly ─────────────────────────
    sentences = []

    # Opener
    sentences.append(templates["opener"].format(score=score_pct) + ".")

    # Primary driver
    if risk_level == "Low":
        # Low risk: state vitals are normal, then mention any mild flags
        sentences.append(templates["primary"] + ".")

        if drivers:
            ctx = _feature_context(drivers[0][0])
            compound = templates["compound"].format(**ctx)
            sentences.append(compound + ".")
        
        # Protective factors
        if protective and len(ranked) >= 3:
            pctx = _feature_context(protective[0][0])
            # Get 2 more features for reassurance
            safe_features = [f for f, v in ranked if v <= 0.05 and not f.startswith("Ward_")]
            if len(safe_features) >= 3:
                ctx2 = _feature_context(safe_features[0])
                ctx3 = _feature_context(safe_features[1])
                prot_sentence = templates["protective"].format(
                    pfeature=pctx["feature"],
                    pvalue=pctx["value"],
                    punit=pctx["unit"],
                    feature2=ctx2["feature"],
                    value2=ctx2["value"],
                    unit2=ctx2["unit"],
                    feature3=ctx3["feature"],
                    value3=ctx3["value"],
                    unit3=ctx3["unit"],
                )
                sentences.append(prot_sentence + ".")
    else:
        # Non-low risk: highlight drivers
        if drivers:
            ctx = _feature_context(drivers[0][0])
            primary = templates["primary"].format(**ctx)
            sentences.append(primary + ".")

        # Compound drivers
        if len(drivers) >= 3:
            ctx2 = _feature_context(drivers[1][0])
            ctx3 = _feature_context(drivers[2][0])
            compound = templates["compound"].format(
                feature2=ctx2["feature"],
                value2=ctx2["value"],
                unit2=ctx2["unit"],
                status2=ctx2["status"],
                feature3=ctx3["feature"],
                value3=ctx3["value"],
                unit3=ctx3["unit"],
                status3=ctx3["status"],
            )
            sentences.append(compound + ".")
        elif len(drivers) >= 2:
            ctx2 = _feature_context(drivers[1][0])
            sentences.append(
                f"This is reinforced by {ctx2['feature']} "
                f"of {ctx2['value']}{ctx2['unit']} ({ctx2['status']})."
            )

        # Protective factors
        if protective:
            pctx = _feature_context(protective[0][0])
            try:
                prot_sentence = templates["protective"].format(
                    pfeature=pctx["feature"],
                    pvalue=pctx["value"],
                    punit=pctx["unit"],
                    pstatus=pctx["status"],
                )
                sentences.append(prot_sentence + ".")
            except (KeyError, IndexError):
                pass

    # Synthesis
    synthesis = templates["synthesis"].format(criteria_count=criteria_count)
    sentences.append(synthesis + ".")

    # Join into paragraph
    paragraph = " ".join(sentences)

    # Clean up double periods and spacing
    paragraph = paragraph.replace("..", ".").replace("  ", " ")

    return paragraph
