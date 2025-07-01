# ไฟล์: config.py
# Path: src/config.py
# วัตถุประสงค์: การตั้งค่าทั้งหมดของโปรเจค (แก้แล้ว - เพิ่ม Smart Sequential Feature Selection)

"""
config.py - การตั้งค่าทั้งหมดของโปรเจค (ปรับปรุงแล้ว - เพิ่ม Smart Sequential)
"""

import os
from pathlib import Path
from typing import Dict, List

# ==================== PATH SETTINGS ====================
# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
RESULT_DIR = BASE_DIR / "result"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATA_PATH = BASE_DIR / "student_realistic_data.csv"

# Result paths
PREPROCESSING_RESULT_DIR = RESULT_DIR / "preprocessing"
FEATURE_SELECTION_RESULT_DIR = RESULT_DIR / "feature_selection"
MODELS_RESULT_DIR = RESULT_DIR / "models"
EVALUATION_RESULT_DIR = RESULT_DIR / "evaluation"
COMPARISON_RESULT_DIR = RESULT_DIR / "comparison"

# สร้างโฟลเดอร์ถ้ายังไม่มี
for directory in [DATA_DIR, RESULT_DIR, MODELS_DIR, LOGS_DIR, 
                  RAW_DATA_DIR, PROCESSED_DATA_DIR,
                  PREPROCESSING_RESULT_DIR, FEATURE_SELECTION_RESULT_DIR,
                  MODELS_RESULT_DIR, EVALUATION_RESULT_DIR, COMPARISON_RESULT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== DATA SETTINGS ====================
TARGET_COLUMN = "แผนก"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature groups (แก้แล้ว - ลบ lifestyle features)
CORE_FEATURES = [
    'คะแนนคณิตศาสตร์', 'คะแนนคอมพิวเตอร์', 'คะแนนภาษาไทย', 
    'คะแนนวิทยาศาสตร์', 'คะแนนศิลปะ', 'ทักษะการคิดเชิงตรรกะ',
    'ทักษะความคิดสร้างสรรค์', 'ทักษะการแก้ปัญหา',
    'ความสนใจด้านตัวเลข', 'ความสนใจด้านเทคโนโลยี', 'ความสนใจด้านการทำอาหาร'
]

DEMOGRAPHIC_FEATURES = [
    'อายุ', 'เพศ'  # ลดลงเหลือแค่ 2 ตัวที่จำเป็น
]

# ลบ LIFESTYLE_FEATURES และ VALIDATION_FEATURES ออกทั้งหมด
EXCLUDE_FEATURES = []  # ไม่มีการ exclude แล้ว

# ==================== FEATURE SELECTION SETTINGS (แก้แล้ว - เพิ่ม Smart Sequential) ====================
FEATURE_SELECTION_METHODS = {
    'forward': 'SequentialFeatureSelector with forward direction',
    'forward_smart': 'Smart Sequential with auto-stopping when accuracy stops improving',
    'backward': 'SequentialFeatureSelector with backward direction', 
    'rfe': 'Recursive Feature Elimination',
    'rfe_cv': 'RFE with Cross Validation (auto-select features)',
    'univariate': 'Univariate Statistical Tests',
    'lasso': 'LASSO Regularization',
    'rf_importance': 'Random Forest Feature Importance'
}

# เลือกวิธี feature selection (แก้เป็น smart version)
FEATURE_SELECTION_METHOD = 'forward_smart'  # หยุดเมื่อ accuracy ไม่เพิ่มแล้ว

# จำนวน features ที่ต้องการ (ใช้เป็น max limit สำหรับ smart method)
N_FEATURES_TO_SELECT = 10  # ใช้เป็น upper bound
MIN_FEATURES = 6
MAX_FEATURES = 12

# Scoring metric สำหรับ feature selection
FEATURE_SELECTION_SCORING = 'accuracy'

# ==================== PREPROCESSING SETTINGS ====================
NORMALIZATION_METHODS = {
    'standard': 'StandardScaler',
    'minmax': 'MinMaxScaler',
    'robust': 'RobustScaler',
    'quantile': 'QuantileTransformer'
}

NORMALIZATION_METHOD = 'standard'
MISSING_VALUE_STRATEGY = 'mean'
CATEGORICAL_ENCODING = 'label'

# ==================== MODEL SETTINGS ====================
AVAILABLE_MODELS = {
    'random_forest': {
        'name': 'Random Forest',
        'class': 'RandomForestClassifier',
        'params': {
            'n_estimators': 100,
            'max_depth': 8,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'class': 'GradientBoostingClassifier', 
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': RANDOM_STATE
        }
    },
    'logistic_regression': {
        'name': 'Logistic Regression',
        'class': 'LogisticRegression',
        'params': {
            'max_iter': 1000,
            'C': 1.0,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'solver': 'lbfgs'
        }
    }
}

# เลือกโมเดลที่ต้องการทดสอบ
SELECTED_MODELS = ['random_forest', 'gradient_boosting', 'logistic_regression']

# ==================== CROSS VALIDATION SETTINGS (แก้แล้ว) ====================
CV_FOLDS = 10  # เปลี่ยนจาก 15 เป็น 10
CV_SCORING_METRICS = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
CV_N_JOBS = -1

# ==================== EVALUATION SETTINGS (ลดความซับซ้อน) ====================
EVALUATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1_score', 
    'confusion_matrix', 'classification_report'
]

# ลบ statistical tests ออก
STATISTICAL_TESTS = {
    'paired_ttest': False,  # ปิดการใช้งาน
    'wilcoxon': False,
    'friedman': False
}

SIGNIFICANCE_LEVEL = 0.05

# ==================== VISUALIZATION SETTINGS ====================
PLOT_SETTINGS = {
    'figsize': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'palette': 'Set2',
    'font_size': 12,
    'title_size': 14,
    'save_format': 'png'
}

GENERATE_PLOTS = True
PLOT_TYPES = [
    'confusion_matrix',
    'feature_importance', 
    'performance_comparison',
    'cross_validation_scores'
]

# ==================== OUTPUT SETTINGS ====================
SAVE_MODELS = True
SAVE_PREDICTIONS = True
SAVE_PROBABILITIES = True

OUTPUT_FORMATS = {
    'data': 'csv',
    'models': 'pickle',
    'reports': 'json',
    'plots': 'png'
}

# ==================== LOGGING SETTINGS ====================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'pipeline.log',
    'console': True
}

VERBOSE = True

# ==================== PERFORMANCE SETTINGS ====================
N_JOBS = -1
MEMORY_LIMIT = '4GB'
CACHE_SIZE = 200

# ==================== EXPERIMENT TRACKING ====================
EXPERIMENT_NAME = "smart_feature_selection_department_recommendation"
EXPERIMENT_VERSION = "v2.2"  # อัพเดท version
TRACK_EXPERIMENTS = False

# ==================== VALIDATION CHECKS ====================
def validate_config():
    """ตรวจสอบความถูกต้องของ configuration"""
    errors = []
    
    # ตรวจสอบไฟล์ข้อมูล
    if not DATA_PATH.exists():
        errors.append(f"Data file not found: {DATA_PATH}")
    
    # ตรวจสอบ model selection
    for model in SELECTED_MODELS:
        if model not in AVAILABLE_MODELS:
            errors.append(f"Unknown model: {model}")
    
    # ตรวจสอบ feature selection
    if FEATURE_SELECTION_METHOD not in FEATURE_SELECTION_METHODS:
        errors.append(f"Unknown feature selection method: {FEATURE_SELECTION_METHOD}")
    
    # ตรวจสอบจำนวน features (สำหรับ smart method ใช้เป็น upper bound)
    if N_FEATURES_TO_SELECT < MIN_FEATURES or N_FEATURES_TO_SELECT > MAX_FEATURES:
        errors.append(f"N_FEATURES_TO_SELECT must be between {MIN_FEATURES} and {MAX_FEATURES}")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))
    
    return True

# ==================== HELPER FUNCTIONS ====================
def get_model_config(model_name):
    """ดึง configuration ของโมเดล"""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return AVAILABLE_MODELS[model_name]

def get_output_path(category, filename):
    """สร้าง path สำหรับไฟล์ output"""
    if category == 'preprocessing':
        return PREPROCESSING_RESULT_DIR / filename
    elif category == 'feature_selection':
        return FEATURE_SELECTION_RESULT_DIR / filename
    elif category == 'models':
        return MODELS_RESULT_DIR / filename
    elif category == 'evaluation':
        return EVALUATION_RESULT_DIR / filename
    elif category == 'comparison':
        return COMPARISON_RESULT_DIR / filename
    else:
        return RESULT_DIR / filename

def get_model_save_path(model_name):
    """สร้าง path สำหรับบันทึกโมเดล"""
    return MODELS_DIR / f"{model_name}.{OUTPUT_FORMATS['models']}"

# ==================== EXPORT SETTINGS ====================
__all__ = [
    'DATA_PATH', 'TARGET_COLUMN', 'RANDOM_STATE', 'TEST_SIZE', 'VALIDATION_SIZE',
    'CORE_FEATURES', 'DEMOGRAPHIC_FEATURES', 'EXCLUDE_FEATURES',
    'NORMALIZATION_METHOD', 'MISSING_VALUE_STRATEGY', 'CATEGORICAL_ENCODING',
    'FEATURE_SELECTION_METHOD', 'N_FEATURES_TO_SELECT', 'MIN_FEATURES', 'MAX_FEATURES',
    'FEATURE_SELECTION_SCORING',
    'SELECTED_MODELS', 'AVAILABLE_MODELS',
    'CV_FOLDS', 'CV_SCORING_METRICS', 'CV_N_JOBS',
    'EVALUATION_METRICS', 'STATISTICAL_TESTS', 'SIGNIFICANCE_LEVEL',
    'PLOT_SETTINGS', 'GENERATE_PLOTS', 'PLOT_TYPES',
    'SAVE_MODELS', 'SAVE_PREDICTIONS', 'SAVE_PROBABILITIES',
    'OUTPUT_FORMATS', 'LOGGING_CONFIG', 'VERBOSE',
    'N_JOBS', 'MEMORY_LIMIT', 'CACHE_SIZE',
    'EXPERIMENT_NAME', 'EXPERIMENT_VERSION', 'TRACK_EXPERIMENTS',
    'get_model_config', 'get_output_path', 'get_model_save_path',
    'validate_config', 'COMPARISON_RESULT_DIR'
]

# เรียกใช้การตรวจสอบเมื่อ import module
if __name__ != "__main__":
    try:
        validate_config()
    except ValueError as e:
        print(f"⚠️  Configuration validation failed: {e}")
        print("Please fix the configuration before running the pipeline.")