"""
Model Evaluation Module

Computes comprehensive evaluation metrics for the sepsis detection model:
- AUC-ROC Score
- F1 Score (weighted)
- Precision & Recall (for positive class)
- Confusion Matrix
- SHAP values for individual predictions
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities for positive class (optional)

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        'f1_positive': float(f1_score(y_true, y_pred, pos_label=1)),
        'precision': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None:
        metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob))
        metrics['average_precision'] = float(average_precision_score(y_true, y_prob))

        # ROC curve data
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }

        # Precision-Recall curve data
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
        metrics['pr_curve'] = {
            'precision': prec_curve.tolist(),
            'recall': rec_curve.tolist()
        }

    return metrics


def print_evaluation_report(metrics: dict, title: str = "Model Evaluation"):
    """Print a formatted evaluation report to console."""
    print(f"\n{'=' * 50}")
    print(f"📊 {title}")
    print(f"{'=' * 50}")

    if 'auc_roc' in metrics:
        auc = metrics['auc_roc']
        status = "✅" if auc >= 0.82 else "⚠️"
        print(f"  {status} AUC-ROC:           {auc:.4f}  (target: ≥ 0.82)")

    f1 = metrics['f1_weighted']
    status = "✅" if f1 >= 0.78 else "⚠️"
    print(f"  {status} F1 (weighted):     {f1:.4f}  (target: ≥ 0.78)")

    print(f"  📈 Precision (pos):  {metrics['precision']:.4f}")
    print(f"  📈 Recall (pos):     {metrics['recall']:.4f}")
    print(f"  📈 F1 (pos class):   {metrics['f1_positive']:.4f}")

    cm = metrics['confusion_matrix']
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                   Neg    Pos")
    print(f"  Actual Neg  [{cm[0][0]:>5}  {cm[0][1]:>5}]")
    print(f"  Actual Pos  [{cm[1][0]:>5}  {cm[1][1]:>5}]")
    print(f"{'=' * 50}")


def compute_shap_values(explainer, features: np.ndarray, feature_names: list) -> dict:
    """
    Compute SHAP values for a single prediction.

    Args:
        explainer: SHAP TreeExplainer
        features: 1D or 2D numpy array of features for one sample
        feature_names: List of feature column names

    Returns:
        Dictionary mapping feature names to SHAP values
    """
    if features.ndim == 1:
        features = features.reshape(1, -1)

    shap_vals = explainer.shap_values(features)

    # Handle different SHAP output formats
    if isinstance(shap_vals, list):
        # For multi-class: use positive class SHAP values
        shap_vals = shap_vals[1]  # index 1 = positive class

    shap_dict = {}
    for i, name in enumerate(feature_names):
        shap_dict[name] = float(shap_vals[0][i])

    return shap_dict
