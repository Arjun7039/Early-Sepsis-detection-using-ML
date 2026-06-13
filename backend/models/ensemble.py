"""
Ensemble Model Definition

Soft-voting ensemble of 4 base classifiers:
- Random Forest (300 trees) — stable baseline with class weighting
- XGBoost (200 trees) — gradient boosting with scale_pos_weight
- LightGBM (200 trees) — fast gradient boosting with balanced class weights
- Gradient Boosting (150 trees) — scikit-learn's native GBM

XGBoost and LightGBM are weighted higher (3x) as they typically
outperform on tabular clinical data with class imbalance.
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def build_ensemble() -> VotingClassifier:
    """
    Build and return the soft-voting ensemble classifier.

    Returns:
        VotingClassifier with 4 base models configured for
        sepsis detection on imbalanced clinical data.
    """
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


def get_model_info() -> dict:
    """
    Return model metadata for the /model/info endpoint.
    """
    return {
        'model_name': 'sepsis-ensemble-v1',
        'model_type': 'Soft Voting Ensemble',
        'base_models': [
            {'name': 'RandomForest', 'n_estimators': 300, 'weight': 2},
            {'name': 'XGBoost', 'n_estimators': 200, 'weight': 3},
            {'name': 'LightGBM', 'n_estimators': 200, 'weight': 3},
            {'name': 'GradientBoosting', 'n_estimators': 150, 'weight': 2},
        ],
        'voting': 'soft',
        'total_estimators': 850,
    }
