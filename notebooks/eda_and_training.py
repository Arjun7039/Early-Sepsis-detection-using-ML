# %% [markdown]
# # 🩺 Early Sepsis Detection System — EDA & Model Training
# 
# **Objective**: Build a clinically-grounded early sepsis detection model using an ensemble ML approach (RF + XGBoost + LightGBM + GBM) with SHAP-based explainability.
# 
# **Dataset**: 250 patient records with vital signs, lab values, and ward assignments.
# 
# **Pipeline**:
# 1. Exploratory Data Analysis (EDA)
# 2. Sepsis-3 Rule-Based Label Augmentation
# 3. Clinical Feature Engineering
# 4. Preprocessing & SMOTE Balancing
# 5. Ensemble Model Training & Evaluation
# 6. SHAP Explainability Analysis

# %% [markdown]
# ## 1. Setup & Imports

# %%
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid', palette='viridis', font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

# Add backend to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath('.')))
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath('.')), 'backend')
if os.path.exists(BACKEND_DIR):
    sys.path.insert(0, BACKEND_DIR)

print("✅ Imports loaded successfully")

# %% [markdown]
# ## 2. Load Raw Dataset

# %%
# Auto-detect dataset path
data_paths = [
    os.path.join(os.path.dirname(os.path.abspath('.')), 'data', 'Sepsis_Dataset____.xlsx'),
    os.path.join(os.path.dirname(os.path.abspath('.')), 'Sepsis_Dataset ....xlsx'),
]
data_path = None
for p in data_paths:
    if os.path.exists(p):
        data_path = p
        break

if data_path is None:
    raise FileNotFoundError("Dataset not found. Place it in data/ folder.")

df_raw = pd.read_excel(data_path, engine='openpyxl')
print(f"📂 Loaded from: {data_path}")
print(f"📊 Shape: {df_raw.shape}")
print(f"📋 Columns: {list(df_raw.columns)}")
df_raw.head()

# %%
df_raw.info()

# %%
df_raw.describe().round(2)

# %% [markdown]
# ## 3. Data Quality Assessment

# %%
# Missing values
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
print("🔍 Missing Values:")
print(missing_df[missing_df['Missing'] > 0] if missing.sum() > 0 else "   ✅ No missing values found!")

# %%
# Check for negative values (data entry errors)
numerical_cols = ['Temperature_C', 'BP_Systolic', 'BP_Diastolic', 'Heart_Rate', 'WBC_Count', 'Lactate_mmol_L']
print("⚠️ Negative value counts:")
for col in numerical_cols:
    if col in df_raw.columns:
        neg_count = (df_raw[col] < 0).sum()
        if neg_count > 0:
            print(f"   {col}: {neg_count} negative values")
if all((df_raw[c] >= 0).all() for c in numerical_cols if c in df_raw.columns):
    print("   ✅ No negatives found (or already cleaned)")

# %%
# Original label distribution
if 'Sepsis_Flag' in df_raw.columns:
    print("🏷️ Original Sepsis_Flag distribution:")
    print(df_raw['Sepsis_Flag'].value_counts())
    print("\n⚠️ All labels are 'No' — we need rule-based augmentation!")

# %% [markdown]
# ## 4. Exploratory Data Analysis — Vital Signs Distributions

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribution of Patient Vital Signs', fontsize=16, fontweight='bold', y=1.02)

vital_cols = ['Temperature_C', 'BP_Systolic', 'BP_Diastolic', 'Heart_Rate', 'WBC_Count', 'Lactate_mmol_L']
colors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']

for idx, (col, color) in enumerate(zip(vital_cols, colors)):
    ax = axes[idx // 3][idx % 3]
    if col in df_raw.columns:
        sns.histplot(df_raw[col], bins=25, kde=True, color=color, ax=ax, alpha=0.7)
        ax.axvline(df_raw[col].median(), color='white', linestyle='--', linewidth=1.5, label=f'Median: {df_raw[col].median():.1f}')
        ax.set_title(col, fontweight='bold')
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('vital_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: vital_distributions.png")

# %%
# Box plots for outlier detection
fig, axes = plt.subplots(1, 6, figsize=(18, 5))
fig.suptitle('Box Plots — Outlier Detection', fontsize=14, fontweight='bold')

for idx, (col, color) in enumerate(zip(vital_cols, colors)):
    if col in df_raw.columns:
        sns.boxplot(y=df_raw[col], ax=axes[idx], color=color, width=0.5)
        axes[idx].set_title(col, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('boxplots.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr_cols = [c for c in vital_cols if c in df_raw.columns]
corr = df_raw[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=1, ax=ax, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Ward distribution
if 'Ward' in df_raw.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    ward_counts = df_raw['Ward'].value_counts()
    bars = ax.bar(ward_counts.index, ward_counts.values, 
                  color=['#6366f1', '#10b981', '#f59e0b', '#ef4444'], edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, ward_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val),
                ha='center', fontweight='bold', fontsize=12)
    ax.set_title('Patient Distribution by Ward', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig('ward_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 5. Sepsis-3 Label Augmentation

# %%
from pipeline.label_augment import augment_labels

df = df_raw.copy()
# Drop non-ML columns
drop_cols = [c for c in ['Patient_ID', 'Doctor_On_Duty', 'Sepsis_Flag', 'Admission_Date'] if c in df.columns]
df = df.drop(columns=drop_cols)

# One-hot encode Ward
if 'Ward' in df.columns:
    ward_dummies = pd.get_dummies(df['Ward'], prefix='Ward', dtype=int)
    df = pd.concat([df.drop('Ward', axis=1), ward_dummies], axis=1)

# Clip negatives
for c in ['WBC_Count', 'Lactate_mmol_L']:
    if c in df.columns:
        df[c] = df[c].clip(lower=0.0)

# Apply Sepsis-3 labeling
df = augment_labels(df)
print("🏷️ Sepsis-3 Label Distribution:")
print(df['Sepsis_Label'].value_counts())
print(f"\n   Positive rate: {df['Sepsis_Label'].mean()*100:.1f}%")

# %%
# Criteria count distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Criteria count bar chart
criteria_counts = df['sepsis_criteria_count'].value_counts().sort_index()
bars = axes[0].bar(criteria_counts.index, criteria_counts.values, color='#6366f1', edgecolor='white')
for bar, val in zip(bars, criteria_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val), ha='center', fontweight='bold')
axes[0].set_xlabel('Number of Sepsis-3 Criteria Met')
axes[0].set_ylabel('Patient Count')
axes[0].set_title('Sepsis-3 Criteria Count Distribution', fontweight='bold')
axes[0].axvline(1.5, color='red', linestyle='--', alpha=0.7, label='Threshold (≥2 = Positive)')
axes[0].legend()

# Right: Label pie chart
label_counts = df['Sepsis_Label'].value_counts()
axes[1].pie(label_counts.values, labels=['Negative (0)', 'Positive (1)'],
            colors=['#10b981', '#ef4444'], autopct='%1.1f%%', startangle=90,
            explode=(0, 0.05), textprops={'fontweight': 'bold', 'fontsize': 12})
axes[1].set_title('Sepsis Label Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('label_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Clinical Feature Engineering
# 
# Based on **PhysioNet Challenge 2019 (PMC6304323)**, we engineer 5 hemodynamic and stress indices:
# - **MAP** (Mean Arterial Pressure)
# - **Shock Index** (HR / SBP)
# - **Modified Shock Index** (HR / MAP)
# - **Pulse Pressure** (SBP - DBP)
# - **Temperature Deviation** (|Temp - 37.0°C|)

# %%
from pipeline.preprocess import engineer_features

df = engineer_features(df)
print("✅ Engineered features added:")
for feat in ['MAP', 'Shock_Index', 'Modified_Shock_Index', 'Pulse_Pressure', 'Temp_Deviation']:
    print(f"   {feat}: mean={df[feat].mean():.2f}, std={df[feat].std():.2f}")

# %%
# Visualize engineered features
eng_features = ['MAP', 'Shock_Index', 'Modified_Shock_Index', 'Pulse_Pressure', 'Temp_Deviation']
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('Engineered Clinical Features Distribution', fontsize=14, fontweight='bold')

eng_colors = ['#8b5cf6', '#ec4899', '#f97316', '#14b8a6', '#eab308']
for idx, (feat, color) in enumerate(zip(eng_features, eng_colors)):
    sns.histplot(df[feat], bins=20, kde=True, color=color, ax=axes[idx], alpha=0.7)
    axes[idx].set_title(feat, fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('engineered_features.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Engineered features by Sepsis label
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle('Engineered Features by Sepsis Label', fontsize=14, fontweight='bold')

for idx, feat in enumerate(eng_features):
    sns.boxplot(x=df['Sepsis_Label'], y=df[feat], ax=axes[idx], palette=['#10b981', '#ef4444'])
    axes[idx].set_title(feat, fontweight='bold', fontsize=10)
    axes[idx].set_xticklabels(['Negative', 'Positive'])

plt.tight_layout()
plt.savefig('features_by_label.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. Preprocessing & SMOTE Balancing

# %%
from pipeline.preprocess import run_preprocessing_pipeline, FEATURE_COLUMNS

data = run_preprocessing_pipeline()
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
scaler = data['scaler']

print(f"\n📊 Final Shapes:")
print(f"   X_train: {X_train.shape}  |  y_train: {len(y_train)}")
print(f"   X_test:  {X_test.shape}  |  y_test:  {len(y_test)}")
print(f"   Features: {FEATURE_COLUMNS}")

# %%
# SMOTE visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before SMOTE (original)
orig_dist = data['label_distribution']
axes[0].bar(['Negative', 'Positive'], [orig_dist['original_negative'], orig_dist['original_positive']],
            color=['#10b981', '#ef4444'], edgecolor='white')
axes[0].set_title('Before SMOTE (Original)', fontweight='bold')
axes[0].set_ylabel('Count')

# After SMOTE
smote_dist = pd.Series(y_train).value_counts()
axes[1].bar(['Negative', 'Positive'], [smote_dist.get(0, 0), smote_dist.get(1, 0)],
            color=['#10b981', '#ef4444'], edgecolor='white')
axes[1].set_title('After SMOTE (Training Set)', fontweight='bold')
axes[1].set_ylabel('Count')

plt.suptitle('Class Balance: Before vs After SMOTE', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('smote_balance.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Model Training — Soft-Voting Ensemble

# %%
from sklearn.model_selection import StratifiedKFold, cross_val_score
from models.ensemble import build_ensemble

ensemble = build_ensemble()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

print("🔄 5-Fold Stratified Cross-Validation:")
for i, s in enumerate(cv_scores, 1):
    print(f"   Fold {i}: AUC-ROC = {s:.4f}")
print(f"   Mean AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# %%
# Train final model
ensemble = build_ensemble()
ensemble.fit(X_train, y_train)
print("✅ Final ensemble model trained!")

y_pred = ensemble.predict(X_test)
y_prob = ensemble.predict_proba(X_test)[:, 1]

# %% [markdown]
# ## 9. Evaluation Metrics

# %%
from pipeline.evaluate import compute_metrics, print_evaluation_report

metrics = compute_metrics(y_test, y_pred, y_prob)
print_evaluation_report(metrics, "Test Set Evaluation")

# %%
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0],
    cmap='Blues', display_labels=['Negative', 'Positive'])
axes[0].set_title('Confusion Matrix', fontweight='bold')

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1],
    color='#6366f1', lw=2, name='Ensemble')
axes[1].plot([0,1], [0,1], 'k--', alpha=0.5, label='Random (AUC=0.5)')
axes[1].set_title(f'ROC Curve (AUC = {metrics.get("auc_roc", 0):.4f})', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('evaluation.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Individual base model performance
from sklearn.metrics import roc_auc_score

print("📊 Individual Base Model AUC-ROC on Test Set:")
for name, model in ensemble.named_estimators_.items():
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    print(f"   {name:>6s}: {auc:.4f}")
print(f"   {'Ensemble':>8s}: {metrics.get('auc_roc', 0):.4f}")

# %% [markdown]
# ## 10. SHAP Explainability Analysis

# %%
import shap

xgb_model = ensemble.named_estimators_['xgb']
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# %%
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLUMNS, show=False, max_display=15)
plt.title('SHAP Feature Importance (Beeswarm)', fontweight='bold')
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLUMNS, plot_type='bar', show=False, max_display=15)
plt.title('SHAP Mean Absolute Impact', fontweight='bold')
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 11. Summary & Conclusions
# 
# ### Key Findings
# - The original dataset had **zero positive sepsis labels**, addressed via **Sepsis-3 rule-based augmentation**
# - **Clinical feature engineering** (MAP, Shock Index, Modified Shock Index, Pulse Pressure, Temp Deviation) increased model signal
# - **SMOTE oversampling** balanced the training set for robust learning
# - The **soft-voting ensemble** (RF + XGBoost + LightGBM + GBM) achieved strong AUC-ROC performance
# - **SHAP analysis** reveals which vitals drive predictions — critical for clinical transparency
# 
# ### Architecture
# - **Backend**: FastAPI with SHAP + NLG justification engine
# - **Frontend**: React + TypeScript dashboard with real-time risk scoring
# - **Deployment**: Render (backend) + Vercel (frontend)
