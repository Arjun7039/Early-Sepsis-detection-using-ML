"""
Label Augmentation Module — Sepsis-3 Rule-Based Labeling

The original dataset has ALL labels as 'No' (zero positive cases).
This module applies clinically validated Sepsis-3 criteria to create
meaningful binary labels for the ML pipeline.

A patient is labeled sepsis-positive if they meet >= 2 of 5 criteria:
1. Temperature < 36.0°C OR > 38.3°C (fever/hypothermia)
2. Heart Rate > 90 bpm (tachycardia)
3. WBC Count < 4.0 OR > 12.0 ×10³/µL (abnormal WBC)
4. Lactate > 2.0 mmol/L (elevated lactate)
5. Systolic BP < 90 mmHg (hypotension)
"""

import pandas as pd


def augment_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Sepsis-3 criteria to create binary labels.
    A patient is sepsis-positive if they meet >= 2 criteria.

    Args:
        df: DataFrame with clinical features (Temperature_C, Heart_Rate,
            WBC_Count, Lactate_mmol_L, BP_Systolic)

    Returns:
        DataFrame with added 'sepsis_criteria_count' and 'Sepsis_Label' columns
    """
    criteria = pd.DataFrame(index=df.index)

    # Criterion 1: Fever or Hypothermia
    criteria['fever_hypothermia'] = (
        (df['Temperature_C'] < 36.0) | (df['Temperature_C'] > 38.3)
    )

    # Criterion 2: Tachycardia
    criteria['tachycardia'] = df['Heart_Rate'] > 90

    # Criterion 3: Abnormal WBC Count
    criteria['wbc_abnormal'] = (
        (df['WBC_Count'] < 4.0) | (df['WBC_Count'] > 12.0)
    )

    # Criterion 4: Elevated Lactate (tissue hypoperfusion marker)
    criteria['high_lactate'] = df['Lactate_mmol_L'] > 2.0

    # Criterion 5: Hypotension (inadequate perfusion pressure)
    criteria['hypotension'] = df['BP_Systolic'] < 90

    # Count how many criteria each patient meets
    df['sepsis_criteria_count'] = criteria.sum(axis=1)

    # Binary label: positive if >= 2 criteria met
    df['Sepsis_Label'] = (df['sepsis_criteria_count'] >= 2).astype(int)

    return df


def get_criteria_met(vitals: dict) -> tuple[list[str], int]:
    """
    Evaluate which Sepsis-3 criteria are met for a single patient.
    Used by the API for real-time prediction context.

    Args:
        vitals: Dictionary with keys matching PatientVitals schema

    Returns:
        Tuple of (list of criteria names met, count of criteria met)
    """
    criteria_met = []

    # Map API field names to check criteria
    temp = vitals.get('temperature_c', vitals.get('Temperature_C', 37.0))
    hr = vitals.get('heart_rate', vitals.get('Heart_Rate', 75))
    wbc = vitals.get('wbc_count', vitals.get('WBC_Count', 7.0))
    lactate = vitals.get('lactate', vitals.get('Lactate_mmol_L', 1.0))
    bp_sys = vitals.get('bp_systolic', vitals.get('BP_Systolic', 120))

    if temp < 36.0 or temp > 38.3:
        criteria_met.append('fever_hypothermia')
    if hr > 90:
        criteria_met.append('tachycardia')
    if wbc < 4.0 or wbc > 12.0:
        criteria_met.append('wbc_abnormal')
    if lactate > 2.0:
        criteria_met.append('high_lactate')
    if bp_sys < 90:
        criteria_met.append('hypotension')

    return criteria_met, len(criteria_met)
