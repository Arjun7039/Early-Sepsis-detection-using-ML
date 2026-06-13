"""
Data Preprocessing Pipeline

Handles the complete data preprocessing workflow:
1. Load raw Excel data
2. Clean and transform features
3. Apply Sepsis-3 label augmentation
4. Scale numerical features
5. Balance classes with SMOTE
6. Return train/test splits ready for model training
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.label_augment import augment_labels


# Feature columns after preprocessing (exact order matters for inference)
FEATURE_COLUMNS = [
    'Temperature_C', 'BP_Systolic', 'BP_Diastolic', 'Heart_Rate',
    'WBC_Count', 'Lactate_mmol_L',
    'MAP', 'Shock_Index', 'Modified_Shock_Index', 'Pulse_Pressure', 'Temp_Deviation',
    'Ward_ER', 'Ward_ICU-A', 'Ward_ICU-B', 'Ward_Ward-X'
]

NUMERICAL_COLUMNS = [
    'Temperature_C', 'BP_Systolic', 'BP_Diastolic', 'Heart_Rate',
    'WBC_Count', 'Lactate_mmol_L',
    'MAP', 'Shock_Index', 'Modified_Shock_Index', 'Pulse_Pressure', 'Temp_Deviation'
]

WARD_COLUMNS = ['Ward_ER', 'Ward_ICU-A', 'Ward_ICU-B', 'Ward_Ward-X']

# Columns to drop (not useful for ML)
DROP_COLUMNS = ['Patient_ID', 'Doctor_On_Duty', 'Sepsis_Flag', 'Admission_Date']


def find_dataset_path() -> str:
    """
    Locate the dataset file — checks multiple possible paths.
    """
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_dir = os.path.dirname(backend_dir)

    possible_paths = [
        os.path.join(project_dir, 'data', 'Sepsis_Dataset____.xlsx'),
        os.path.join(project_dir, 'Sepsis_Dataset ....xlsx'),
        os.path.join(project_dir, 'Sepsis_Dataset____.xlsx'),
        os.path.join(project_dir, 'data', 'Sepsis_Dataset ....xlsx'),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Dataset not found. Searched:\n" +
        "\n".join(f"  - {p}" for p in possible_paths)
    )


def load_raw_data(filepath: str = None) -> pd.DataFrame:
    """Load raw Excel dataset."""
    if filepath is None:
        filepath = find_dataset_path()

    print(f"📂 Loading dataset from: {filepath}")
    df = pd.read_excel(filepath, engine='openpyxl')
    print(f"   Raw shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
    - Drop non-ML columns
    - Fix negative values (data entry errors)
    - One-hot encode Ward
    """
    df = df.copy()

    # Step 1: Drop columns not needed for ML
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"   Dropped columns: {cols_to_drop}")

    # Step 2: One-hot encode Ward (before dropping)
    if 'Ward' in df.columns:
        ward_dummies = pd.get_dummies(df['Ward'], prefix='Ward', dtype=int)
        df = pd.concat([df.drop('Ward', axis=1), ward_dummies], axis=1)
        print(f"   One-hot encoded Ward → {list(ward_dummies.columns)}")

    # Step 3: Clip physiologically invalid values
    if 'WBC_Count' in df.columns:
        neg_wbc = (df['WBC_Count'] < 0).sum()
        df['WBC_Count'] = df['WBC_Count'].clip(lower=0.0)
        if neg_wbc > 0:
            print(f"   ⚠ Clipped {neg_wbc} negative WBC_Count values to 0.0")

    if 'Lactate_mmol_L' in df.columns:
        neg_lac = (df['Lactate_mmol_L'] < 0).sum()
        df['Lactate_mmol_L'] = df['Lactate_mmol_L'].clip(lower=0.0)
        if neg_lac > 0:
            print(f"   ⚠ Clipped {neg_lac} negative Lactate_mmol_L values to 0.0")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers clinical features based on PhysioNet Challenge 2019 and clinical standards:
    - MAP (Mean Arterial Pressure): BP_Diastolic + (BP_Systolic - BP_Diastolic) / 3
    - Shock Index: Heart_Rate / BP_Systolic
    - Modified Shock Index: Heart_Rate / MAP
    - Pulse Pressure: BP_Systolic - BP_Diastolic
    - Temp Deviation: abs(Temperature_C - 37.0)
    """
    df = df.copy()

    # 1. MAP
    df['MAP'] = df['BP_Diastolic'] + (df['BP_Systolic'] - df['BP_Diastolic']) / 3

    # 2. Shock Index (handle division by zero just in case)
    df['Shock_Index'] = df['Heart_Rate'] / df['BP_Systolic'].replace(0, 1)

    # 3. Modified Shock Index
    df['Modified_Shock_Index'] = df['Heart_Rate'] / df['MAP'].replace(0, 1)

    # 4. Pulse Pressure
    df['Pulse_Pressure'] = df['BP_Systolic'] - df['BP_Diastolic']

    # 5. Temp Deviation
    df['Temp_Deviation'] = (df['Temperature_C'] - 37.0).abs()

    return df


def run_preprocessing_pipeline(
    filepath: str = None,
    test_size: float = 0.2,
    smote_ratio: float = 0.4,
    random_state: int = 42
) -> dict:
    """
    Run the complete preprocessing pipeline.

    Args:
        filepath: Path to the Excel dataset (auto-detected if None)
        test_size: Fraction for test split
        smote_ratio: Target ratio of positive class after SMOTE
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
        - X_train, X_test, y_train, y_test: Train/test splits
        - scaler: Fitted StandardScaler
        - feature_names: List of feature column names
        - label_distribution: Dict with class counts
    """
    print("\n" + "=" * 60)
    print("🔬 SEPSIS DETECTION — DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load raw data
    print("\n📥 Step 1: Loading raw data...")
    df = load_raw_data(filepath)

    # Step 2: Clean data
    print("\n🧹 Step 2: Cleaning data...")
    df = clean_data(df)

    # Step 2.5: Feature Engineering
    print("\n🔬 Step 2.5: Engineering clinical features...")
    df = engineer_features(df)

    # Step 3: Apply Sepsis-3 label augmentation
    print("\n🏷️  Step 3: Applying Sepsis-3 label augmentation...")
    df = augment_labels(df)

    label_counts = df['Sepsis_Label'].value_counts()
    total = len(df)
    pos_count = label_counts.get(1, 0)
    neg_count = label_counts.get(0, 0)
    print(f"   Label distribution after Sepsis-3 criteria:")
    print(f"     Negative (0): {neg_count} ({neg_count/total*100:.1f}%)")
    print(f"     Positive (1): {pos_count} ({pos_count/total*100:.1f}%)")

    # Step 4: Extract features and target
    print("\n📊 Step 4: Extracting features and target...")

    # Drop the criteria count column (used only for labeling)
    if 'sepsis_criteria_count' in df.columns:
        df = df.drop('sepsis_criteria_count', axis=1)

    # Ensure all expected feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURE_COLUMNS].copy()
    y = df['Sepsis_Label'].copy()
    print(f"   Features shape: {X.shape}")
    print(f"   Feature columns: {list(X.columns)}")

    # Step 5: Scale numerical features
    print("\n⚖️  Step 5: Scaling numerical features...")
    scaler = StandardScaler()
    X[NUMERICAL_COLUMNS] = scaler.fit_transform(X[NUMERICAL_COLUMNS])
    print(f"   Scaled {len(NUMERICAL_COLUMNS)} numerical columns")

    # Step 6: Train-test split (stratified)
    print("\n✂️  Step 6: Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")

    # Step 7: Apply SMOTE to balance training set
    print("\n🔄 Step 7: Applying SMOTE...")
    print(f"   Before SMOTE — Train class dist: {dict(y_train.value_counts())}")

    # Calculate sampling strategy for target ratio
    n_negative = (y_train == 0).sum()
    n_positive_target = int(n_negative * smote_ratio / (1 - smote_ratio))

    smote = SMOTE(
        sampling_strategy={1: max(n_positive_target, (y_train == 1).sum())},
        random_state=random_state,
        k_neighbors=min(5, (y_train == 1).sum() - 1) if (y_train == 1).sum() > 1 else 1
    )
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"   After SMOTE  — Train class dist: {dict(pd.Series(y_train_resampled).value_counts())}")
    print(f"   Resampled train size: {X_train_resampled.shape[0]}")

    # Convert back to DataFrame for consistency
    X_train_resampled = pd.DataFrame(X_train_resampled, columns=FEATURE_COLUMNS)

    print("\n✅ Preprocessing pipeline complete!")
    print("=" * 60)

    return {
        'X_train': X_train_resampled,
        'X_test': X_test,
        'y_train': y_train_resampled,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': FEATURE_COLUMNS,
        'label_distribution': {
            'original_positive': int(pos_count),
            'original_negative': int(neg_count),
            'train_after_smote': dict(pd.Series(y_train_resampled).value_counts()),
            'test': dict(y_test.value_counts())
        }
    }


if __name__ == '__main__':
    # Run standalone for testing
    result = run_preprocessing_pipeline()
    print(f"\n📋 Summary:")
    print(f"   X_train shape: {result['X_train'].shape}")
    print(f"   X_test shape:  {result['X_test'].shape}")
    print(f"   Features: {result['feature_names']}")
    print(f"   Label dist: {result['label_distribution']}")
