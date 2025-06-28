# ไฟล์: config.py
# Path: src/config.py
# วัตถุประสงค์: การตั้งค่าทั้งหมดของโปรเจค (แก้แล้ว - แก้ DATA_PATH)

"""
config.py - การตั้งค่าทั้งหมดของโปรเจค
"""

import os
from pathlib import Path

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
# แก้ path ให้ชี้ไปที่ root directory
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

# Feature groups (สำหรับการวิเคราะห์)
CORE_FEATURES = [
    'คะแนนคณิตศาสตร์', 'คะแนนคอมพิวเตอร์', 'คะแนนภาษาไทย', 
    'คะแนนวิทยาศาสตร์', 'คะแนนศิลปะ', 'ทักษะการคิดเชิงตรรกะ',
    'ทักษะความคิดสร้างสรรค์', 'ทักษะการแก้ปัญหา',
    'ความสนใจด้านตัวเลข', 'ความสนใจด้านเทคโนโลยี', 'ความสนใจด้านการทำอาหาร'
]

DEMOGRAPHIC_FEATURES = [
    'อายุ', 'เพศ', 'รายได้ครอบครัว', 'การศึกษาของผู้ปกครอง', 'จำนวนพี่น้อง'
]

LIFESTYLE_FEATURES = [
    'ชั่วโมงการนอน', 'ความถี่การออกกำลังกาย', 'ชั่วโมงใช้โซเชียลมีเดีย',
    'ชอบอ่านหนังสือ', 'ประเภทเพลงที่ชอบ'
]

VALIDATION_FEATURES = [
    'ชั้นปี', 'GPA', 'ความพึงพอใจในแผนก', 'หากเลือกใหม่จะเลือกแผนกเดิมไหม'
]

# Features ที่ต้อง exclude จากการทำ ML (เป็น validation data)
EXCLUDE_FEATURES = ['ชั้นปี', 'GPA', 'ความพึงพอใจในแผนก', 'หากเลือกใหม่จะเลือกแผนกเดิมไหม']

# ==================== PREPROCESSING SETTINGS ====================
NORMALIZATION_METHODS = {
    'standard': 'StandardScaler',      # แนะนำสำหรับข้อมูลปกติ
    'minmax': 'MinMaxScaler',         # แนะนำสำหรับข้อมูลที่มี outliers น้อย  
    'robust': 'RobustScaler',         # แนะนำสำหรับข้อมูลที่มี outliers เยอะ
    'quantile': 'QuantileTransformer' # แนะนำสำหรับข้อมูลที่ไม่ปกติ
}

# เลือกวิธี normalization (เปลี่ยนได้ตามต้องการ)
NORMALIZATION_METHOD = 'standard'

# การจัดการ missing values
MISSING_VALUE_STRATEGY = 'mean'  # 'mean', 'median', 'mode', 'drop'

# การจัดการ categorical variables
CATEGORICAL_ENCODING = 'label'   # 'label', 'onehot', 'target'

# ==================== FEATURE SELECTION SETTINGS ====================
FEATURE_SELECTION_METHODS = {
    'forward': 'SequentialFeatureSelector with forward direction',
    'backward': 'SequentialFeatureSelector with backward direction', 
    'rfe': 'Recursive Feature Elimination',
    'rfe_cv': 'RFE with Cross Validation',
    'univariate': 'Univariate Statistical Tests',
    'lasso': 'LASSO Regularization',
    'rf_importance': 'Random Forest Feature Importance'
}

# เลือกวิธี feature selection
FEATURE_SELECTION_METHOD = 'forward'  # หรือ 'backward', 'rfe', etc.

# จำนวน features ที่ต้องการ
N_FEATURES_TO_SELECT = 15
MIN_FEATURES = 10
MAX_FEATURES = 20

# Scoring metric สำหรับ feature selection
FEATURE_SELECTION_SCORING = 'accuracy'

# ==================== MODEL SETTINGS ====================
AVAILABLE_MODELS = {
    'random_forest': {
        'name': 'Random Forest',
        'class': 'RandomForestClassifier',
        'params': {
            'n_estimators': 100,
            'max_depth': 10,
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
            'max_depth': 6,
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
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'solver': 'lbfgs'
        }
    },
    'svm': {
        'name': 'Support Vector Machine',
        'class': 'SVC',
        'params': {
            'kernel': 'rbf',
            'C': 1.0,
            'random_state': RANDOM_STATE,
            'probability': True
        }
    },
    'xgboost': {
        'name': 'XGBoost',
        'class': 'XGBClassifier',
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
    }
}

# เลือกโมเดลที่ต้องการทดสอบ

SELECTED_MODELS = ['random_forest', 'gradient_boosting', 'logistic_regression']

# ==================== CROSS VALIDATION SETTINGS ====================
CV_FOLDS = 10
CV_SCORING_METRICS = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
CV_N_JOBS = -1

# ==================== EVALUATION SETTINGS ====================
EVALUATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1_score', 
    'confusion_matrix', 'classification_report'
]

# การตั้งค่า statistical testing
STATISTICAL_TESTS = {
    'paired_ttest': True,      # Paired t-test
    'wilcoxon': True,          # Wilcoxon signed-rank test
    'friedman': True,          # Friedman test (สำหรับหลายโมเดล)
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
    'cross_validation_scores',
    'learning_curves'
]

# ==================== OUTPUT SETTINGS ====================
SAVE_MODELS = True
SAVE_PREDICTIONS = True
SAVE_PROBABILITIES = True

# รูปแบบการบันทึกไฟล์
OUTPUT_FORMATS = {
    'data': 'csv',           # csv, parquet, pickle
    'models': 'pickle',      # pickle, joblib
    'reports': 'json',       # json, yaml
    'plots': 'png'          # png, pdf, svg
}

# ==================== LOGGING SETTINGS ====================
LOGGING_CONFIG = {
    'level': 'INFO',         # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'pipeline.log',
    'console': True
}

VERBOSE = True

# ==================== PERFORMANCE SETTINGS ====================
N_JOBS = -1                  # จำนวน CPU cores ที่ใช้ (-1 = ใช้ทั้งหมด)
MEMORY_LIMIT = '4GB'         # ขีดจำกัดการใช้ memory
CACHE_SIZE = 200             # Cache size สำหรับ SVM

# ==================== ADVANCED SETTINGS ====================
# Grid Search settings (สำหรับ hyperparameter tuning)
GRID_SEARCH_PARAMS = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'logistic_regression': {
        'C': [0.1, 1.0, 10.0],
        'solver': ['lbfgs', 'liblinear']
    }
}

ENABLE_GRID_SEARCH = False   # เปิด/ปิด grid search
GRID_SEARCH_CV = 5
GRID_SEARCH_SCORING = 'accuracy'

# Early stopping settings
EARLY_STOPPING = {
    'enabled': False,
    'patience': 10,
    'min_delta': 0.001
}

# ==================== EXPERIMENT TRACKING ====================
EXPERIMENT_NAME = "feature_selection_department_recommendation"
EXPERIMENT_VERSION = "v1.0"
TRACK_EXPERIMENTS = False    # เปิด/ปิด experiment tracking (MLflow, Weights & Biases)

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
    
    # ตรวจสอบจำนวน features
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
# สำหรับการ import ใน modules อื่น
__all__ = [
    'DATA_PATH', 'TARGET_COLUMN', 'RANDOM_STATE',
    'CORE_FEATURES', 'DEMOGRAPHIC_FEATURES', 'LIFESTYLE_FEATURES', 'VALIDATION_FEATURES',
    'NORMALIZATION_METHOD', 'FEATURE_SELECTION_METHOD', 'N_FEATURES_TO_SELECT',
    'SELECTED_MODELS', 'CV_FOLDS', 'EVALUATION_METRICS',
    'SAVE_MODELS', 'GENERATE_PLOTS', 'VERBOSE',
    'get_model_config', 'get_output_path', 'get_model_save_path',
    'validate_config', 'EXCLUDE_FEATURES', 'DATA_PATH'
]

# เรียกใช้การตรวจสอบเมื่อ import module
if __name__ != "__main__":
    validate_config()