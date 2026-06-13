"""
Model Training Script

Orchestrates the complete training workflow:
1. Run preprocessing pipeline (load → clean → label → scale → SMOTE)
2. Perform 5-fold stratified cross-validation
3. Train final model on full training set
4. Compute and display evaluation metrics
5. Save artifacts (model, scaler, SHAP explainer)

Run this script to train the model:
    cd backend
    python pipeline/train.py
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import StratifiedKFold, cross_val_score
from pipeline.preprocess import run_preprocessing_pipeline, FEATURE_COLUMNS
from pipeline.evaluate import compute_metrics, print_evaluation_report
from models.ensemble import build_ensemble


# Directory for saved artifacts
ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'artifacts'
)


def train_model():
    """
    Run the complete model training pipeline.

    Returns:
        Dictionary with trained model, metrics, and artifact paths
    """
    print("\n" + "🧬" * 30)
    print("  EARLY SEPSIS DETECTION — MODEL TRAINING")
    print("🧬" * 30)

    # ── Step 1: Preprocessing ──────────────────────────────
    print("\n📦 Step 1: Running preprocessing pipeline...")
    data = run_preprocessing_pipeline()

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    scaler = data['scaler']

    # ── Step 2: Cross-Validation ───────────────────────────
    print("\n🔄 Step 2: 5-Fold Stratified Cross-Validation...")
    ensemble = build_ensemble()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        ensemble, X_train, y_train,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )

    print(f"\n  Fold AUC-ROC Scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"    Fold {i}: {score:.4f}")
    print(f"    {'─' * 25}")
    print(f"    Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Step 3: Train Final Model ──────────────────────────
    print("\n🏋️ Step 3: Training final model on full training set...")
    ensemble = build_ensemble()  # Fresh instance
    ensemble.fit(X_train, y_train)
    print("   ✅ Model trained successfully!")

    # ── Step 4: Evaluate ───────────────────────────────────
    print("\n📊 Step 4: Evaluating on test set...")
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics['cv_scores'] = cv_scores.tolist()
    metrics['cv_mean_auc'] = float(cv_scores.mean())

    print_evaluation_report(metrics, "Test Set Evaluation")

    # ── Step 5: Create SHAP Explainer ──────────────────────
    print("\n🔍 Step 5: Creating SHAP explainer...")
    import shap

    # Use XGBoost base model for TreeExplainer (most compatible)
    xgb_model = ensemble.named_estimators_['xgb']
    explainer = shap.TreeExplainer(xgb_model)
    print("   ✅ SHAP TreeExplainer created (using XGBoost base model)")

    # ── Step 6: Save Artifacts ─────────────────────────────
    print("\n💾 Step 6: Saving artifacts...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    model_path = os.path.join(ARTIFACTS_DIR, 'sepsis_ensemble.pkl')
    scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
    shap_path = os.path.join(ARTIFACTS_DIR, 'shap_explainer.pkl')
    metrics_path = os.path.join(ARTIFACTS_DIR, 'training_metrics.pkl')

    joblib.dump(ensemble, model_path)
    print(f"   📁 Model saved:     {model_path}")

    joblib.dump(scaler, scaler_path)
    print(f"   📁 Scaler saved:    {scaler_path}")

    joblib.dump(explainer, shap_path)
    print(f"   📁 SHAP saved:      {shap_path}")

    # Save metrics for the /model/info endpoint
    training_info = {
        'metrics': metrics,
        'feature_names': FEATURE_COLUMNS,
        'label_distribution': data['label_distribution'],
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
    }
    joblib.dump(training_info, metrics_path)
    print(f"   📁 Metrics saved:   {metrics_path}")

    # ── Summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🎯 TRAINING COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"   AUC-ROC (test):    {metrics.get('auc_roc', 'N/A'):.4f}")
    print(f"   F1 (weighted):     {metrics['f1_weighted']:.4f}")
    print(f"   CV Mean AUC:       {cv_scores.mean():.4f}")
    print(f"   Artifacts saved to: {ARTIFACTS_DIR}")
    print("=" * 60)

    return {
        'model': ensemble,
        'scaler': scaler,
        'explainer': explainer,
        'metrics': metrics,
        'artifact_paths': {
            'model': model_path,
            'scaler': scaler_path,
            'shap': shap_path,
            'metrics': metrics_path,
        }
    }


if __name__ == '__main__':
    result = train_model()
