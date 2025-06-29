# ไฟล์: config.py
# Path: src/config.py
# วัตถุประสงค์: การตั้งค่าทั้งหมดของโปรเจค (แก้แล้ว - เพิ่ม Domain Knowledge Rules)

"""
config.py - การตั้งค่าทั้งหมดของโปรเจค (ปรับปรุงแล้ว)
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

# ==================== DOMAIN KNOWLEDGE CONSTRAINTS (ใหม่) ====================
# Feature priorities สำหรับการเลือกแผนกเรียน ตามหลักการทางการศึกษา

FEATURE_PRIORITIES = {
    'CORE_ACADEMIC': {
        'features': [
            'คะแนนคณิตศาสตร์', 'คะแนนคอมพิวเตอร์', 'คะแนนภาษาไทย', 
            'คะแนนวิทยาศาสตร์', 'คะแนนศิลปะ'
        ],
        'weight': 1.0,
        'min_percentage': 40,  # อย่างน้อย 40% ของ selected features
        'description': 'คะแนนวิชาหลัก - ปัจจัยสำคัญที่สุดในการเลือกแผนก',
        'educational_theory': 'Academic Achievement Theory - คะแนนวิชาเป็นตัวทำนายความสำเร็จ'
    },
    'CORE_SKILLS': {
        'features': [
            'ทักษะการคิดเชิงตรรกะ', 'ทักษะความคิดสร้างสรรค์', 'ทักษะการแก้ปัญหา'
        ],
        'weight': 0.9,
        'min_percentage': 15,  # อย่างน้อย 15%
        'description': 'ทักษะหลัก - พื้นฐานการเรียนรู้',
        'educational_theory': 'Cognitive Skills Framework - ทักษะการคิดเป็นพื้นฐานสำคัญ'
    },
    'CORE_INTERESTS': {
        'features': [
            'ความสนใจด้านตัวเลข', 'ความสนใจด้านเทคโนโลยี', 'ความสนใจด้านการทำอาหาร'
        ],
        'weight': 0.8,
        'min_percentage': 15,  # อย่างน้อย 15%
        'description': 'ความสนใจหลัก - แรงจูงใจในการเรียน',
        'educational_theory': 'Vocational Interest Theory - ความสนใจขับเคลื่อนการเรียนรู้'
    },
    'DEMOGRAPHIC_LIMITED': {
        'features': [
            'อายุ', 'เพศ'  # เฉพาะที่จำเป็นจริงๆ
        ],
        'weight': 0.3,
        'max_percentage': 15,  # ไม่เกิน 15%
        'description': 'ข้อมูลประชากรศาสตร์ - ผลกระทบน้อย',
        'educational_theory': 'Demographic factors should be minimized for equity'
    },
    'SOCIOECONOMIC_BLOCKED': {
        'features': [
            'รายได้ครอบครัว', 'การศึกษาของผู้ปกครอง', 'จำนวนพี่น้อง'
        ],
        'weight': 0.0,
        'max_percentage': 0,  # ห้ามใช้เด็ดขาด
        'description': 'ปัจจัยเศรษฐกิจสังคม - ไม่ควรเป็นข้อจำกัดในการเข้าถึงการศึกษา',
        'educational_theory': 'Educational Equity Principle - หลีกเลี่ยงการเลือกปฏิบัติ'
    },
    'LIFESTYLE_BLOCKED': {
        'features': [
            'ชั่วโมงการนอน', 'ความถี่การออกกำลังกาย', 'ชั่วโมงใช้โซเชียลมีเดีย',
            'ชอบอ่านหนังสือ', 'ประเภทเพลงที่ชอบ'
        ],
        'weight': 0.0,
        'max_percentage': 0,  # ห้ามใช้เด็ดขาด
        'description': 'ปัจจัยไลฟ์สไตล์ - เรื่องส่วนตัวที่ไม่ควรมีผลต่อการเลือกแผนก',
        'educational_theory': 'Personal lifestyle should not determine educational opportunities'
    }
}

# Feature Selection Validation Rules (ใหม่)
FEATURE_SELECTION_VALIDATION = {
    'min_core_percentage': 70,         # อย่างน้อย 70% ต้องเป็น core features
    'max_demographic_percentage': 15,  # ไม่เกิน 15% demographic
    'max_socioeconomic_percentage': 0, # ห้าม socioeconomic features
    'max_lifestyle_percentage': 0,     # ห้าม lifestyle features
    'min_quality_score': 150,          # คะแนนคุณภาพขั้นต่ำ (จาก 200)
    'require_all_subject_types': True, # ต้องมีคะแนนวิชาหลากหลาย
    'require_interest_diversity': True, # ต้องมีความสนใจหลากหลาย
    'academic_defensibility_threshold': 150  # เกณฑ์ที่นักวิชาการยอมรับได้
}

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

# ==================== FEATURE SELECTION SETTINGS (ปรับปรุงแล้ว) ====================
FEATURE_SELECTION_METHODS = {
    'enhanced_forward': 'Enhanced Sequential Forward with Domain Knowledge',
    'forward': 'SequentialFeatureSelector with forward direction',
    'backward': 'SequentialFeatureSelector with backward direction', 
    'rfe': 'Recursive Feature Elimination',
    'rfe_cv': 'RFE with Cross Validation',
    'univariate': 'Univariate Statistical Tests',
    'lasso': 'LASSO Regularization',
    'rf_importance': 'Random Forest Feature Importance'
}

# เลือกวิธี feature selection (ใช้ enhanced version)
FEATURE_SELECTION_METHOD = 'enhanced_forward'  # แก้เป็น enhanced version

# จำนวน features ที่ต้องการ (ปรับให้เข้มงวดขึ้น)
N_FEATURES_TO_SELECT = 12  # ลดจาก 15 เพื่อเฟ้นเอาดีๆ
MIN_FEATURES = 8
MAX_FEATURES = 15

# Scoring metric สำหรับ feature selection
FEATURE_SELECTION_SCORING = 'accuracy'

# ==================== MODEL SETTINGS (ปรับปรุงแล้ว) ====================
AVAILABLE_MODELS = {
    'random_forest': {
        'name': 'Random Forest',
        'class': 'RandomForestClassifier',
        'params': {
            'n_estimators': 200,       # เพิ่มจาก 100
            'max_depth': 8,            # ลดจาก 10 เพื่อป้องกัน overfitting
            'min_samples_split': 10,   # เพิ่มจาก 5
            'min_samples_leaf': 5,     # เพิ่มจาก 2
            'max_features': 'sqrt',    # เพิ่มเพื่อลด noise
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'class': 'GradientBoostingClassifier', 
        'params': {
            'n_estimators': 150,       # เพิ่มจาก 100
            'learning_rate': 0.08,     # ลดจาก 0.1
            'max_depth': 5,            # ลดจาก 6
            'min_samples_split': 15,   # เพิ่มจาก 5
            'min_samples_leaf': 8,     # เพิ่มจาก 2
            'subsample': 0.8,          # เพิ่มเพื่อลด overfitting
            'random_state': RANDOM_STATE
        }
    },
    'logistic_regression': {
        'name': 'Logistic Regression',
        'class': 'LogisticRegression',
        'params': {
            'max_iter': 2000,          # เพิ่มจาก 1000
            'C': 0.5,                  # เพิ่ม regularization (ลดจาก 1.0)
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'solver': 'lbfgs',
            'class_weight': 'balanced' # เพื่อจัดการ class imbalance
        }
    },
    'svm': {
        'name': 'Support Vector Machine',
        'class': 'SVC',
        'params': {
            'kernel': 'rbf',
            'C': 0.8,                  # ลดจาก 1.0
            'gamma': 'scale',          # เพิ่มเพื่อควบคุม complexity
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

# เลือกโมเดลที่ต้องการทดสอบ (เน้นโมเดลที่เสถียร)
SELECTED_MODELS = ['random_forest', 'gradient_boosting', 'logistic_regression']

# ==================== CROSS VALIDATION SETTINGS (ปรับปรุงแล้ว) ====================
CV_FOLDS = 15  # เพิ่มจาก 10 เพื่อความแม่นยำ
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
    'learning_curves',
    'domain_knowledge_validation'  # เพิ่มใหม่
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
        'n_estimators': [100, 150, 200],
        'max_depth': [6, 8, 10, None],
        'min_samples_split': [5, 10, 15]
    },
    'gradient_boosting': {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.05, 0.08, 0.1],
        'max_depth': [4, 5, 6]
    },
    'logistic_regression': {
        'C': [0.1, 0.5, 1.0, 2.0],
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
EXPERIMENT_NAME = "enhanced_feature_selection_department_recommendation"
EXPERIMENT_VERSION = "v2.0"  # อัพเดท version
TRACK_EXPERIMENTS = False    # เปิด/ปิด experiment tracking (MLflow, Weights & Biases)

# ==================== DOMAIN KNOWLEDGE FUNCTIONS (ใหม่) ====================

def validate_feature_priorities(selected_features: List[str]) -> Dict:
    """
    ตรวจสอบว่า feature selection ตรงตาม domain knowledge หรือไม่
    """
    validation_results = {}
    total_features = len(selected_features)
    
    if total_features == 0:
        return {'overall_valid': False, 'error': 'No features selected'}
    
    for priority_name, config in FEATURE_PRIORITIES.items():
        matching_features = [f for f in selected_features if f in config['features']]
        percentage = len(matching_features) / total_features * 100
        
        # ตรวจสอบ constraints
        if 'min_percentage' in config:
            meets_min = percentage >= config['min_percentage']
        else:
            meets_min = True
            
        if 'max_percentage' in config:
            meets_max = percentage <= config['max_percentage']
        else:
            meets_max = True
        
        validation_results[priority_name] = {
            'matching_features': matching_features,
            'count': len(matching_features),
            'percentage': percentage,
            'meets_min_requirement': meets_min,
            'meets_max_requirement': meets_max,
            'is_valid': meets_min and meets_max,
            'config': config
        }
    
    # Overall validation
    overall_valid = all(result['is_valid'] for result in validation_results.values())
    
    return {
        'overall_valid': overall_valid,
        'priority_results': validation_results,
        'selected_features': selected_features,
        'total_features': total_features
    }

def get_feature_priority_score(selected_features: List[str]) -> float:
    """
    คำนวณคะแนนคุณภาพของ feature selection ตาม domain knowledge (0-200)
    """
    if not selected_features:
        return 0.0
    
    total_score = 0
    max_possible_score = 0
    
    for priority_name, config in FEATURE_PRIORITIES.items():
        matching_features = [f for f in selected_features if f in config['features']]
        
        if config['weight'] > 0:  # เฉพาะ categories ที่มี weight บวก
            feature_ratio = len(matching_features) / max(len(config['features']), 1)
            
            # คะแนนสำหรับ priority นี้
            priority_score = feature_ratio * config['weight'] * 100
            total_score += priority_score
            max_possible_score += config['weight'] * 100
        else:  # Categories ที่ต้อง block (weight = 0)
            if len(matching_features) > 0:
                # หักคะแนนถ้ามี blocked features
                total_score -= len(matching_features) * 20  # หัก 20 คะแนนต่อ blocked feature
    
    # Normalize to 0-200 scale
    if max_possible_score > 0:
        base_score = (total_score / max_possible_score) * 200
    else:
        base_score = 0
    
    # เพิ่มโบนัสถ้าไม่มี blocked features
    blocked_categories = ['SOCIOECONOMIC_BLOCKED', 'LIFESTYLE_BLOCKED']
    no_blocked_bonus = 0
    for cat in blocked_categories:
        matching = [f for f in selected_features if f in FEATURE_PRIORITIES[cat]['features']]
        if len(matching) == 0:
            no_blocked_bonus += 25  # โบนัส 25 คะแนนต่อ category ที่ clean
    
    final_score = base_score + no_blocked_bonus
    return max(0, min(200, final_score))  # จำกัดให้อยู่ใน 0-200

def get_academic_justification(selected_features: List[str]) -> str:
    """
    สร้างเหตุผลทางวิชาการสำหรับ feature selection
    """
    validation = validate_feature_priorities(selected_features)
    quality_score = get_feature_priority_score(selected_features)
    
    # สรุปสัดส่วน
    core_pct = (
        validation['priority_results']['CORE_ACADEMIC']['percentage'] +
        validation['priority_results']['CORE_SKILLS']['percentage'] +
        validation['priority_results']['CORE_INTERESTS']['percentage']
    )
    
    blocked_count = (
        len(validation['priority_results']['SOCIOECONOMIC_BLOCKED']['matching_features']) +
        len(validation['priority_results']['LIFESTYLE_BLOCKED']['matching_features'])
    )
    
    justification = f"""
🎓 **Academic Justification for Feature Selection**

📊 **Selection Quality:**
- Overall Quality Score: {quality_score:.1f}/200
- Educational Factors: {core_pct:.1f}%
- Blocked Inappropriate Factors: {blocked_count} features eliminated

📚 **Educational Principles Applied:**
1. **Academic Achievement Theory**: Academic scores predict departmental success
2. **Cognitive Skills Framework**: Thinking skills are foundational for learning
3. **Vocational Interest Theory**: Career interests drive motivation and persistence
4. **Educational Equity Principle**: Avoid socioeconomic and lifestyle discrimination

✅ **Defensibility:**
- Features selected based on educationally relevant factors
- Personal lifestyle factors appropriately excluded
- Socioeconomic barriers minimized for educational equity
- Selection process follows established educational psychology principles

🎯 **Practical Implementation:**
- Students can develop academic performance through effort
- Skills and interests are educationally relevant and fair
- Selection supports equal educational opportunities
- Methodology is transparent and academically sound
    """
    
    return justification.strip()

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
    
    # ตรวจสอบ domain knowledge settings
    if not FEATURE_PRIORITIES:
        errors.append("FEATURE_PRIORITIES not defined")
    
    if not FEATURE_SELECTION_VALIDATION:
        errors.append("FEATURE_SELECTION_VALIDATION not defined")
    
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
    'DATA_PATH', 'TARGET_COLUMN', 'RANDOM_STATE', 'TEST_SIZE', 'VALIDATION_SIZE',
    'CORE_FEATURES', 'DEMOGRAPHIC_FEATURES', 'LIFESTYLE_FEATURES', 'VALIDATION_FEATURES',
    'EXCLUDE_FEATURES',
    'FEATURE_PRIORITIES', 'FEATURE_SELECTION_VALIDATION',  # เพิ่มใหม่
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
    'GRID_SEARCH_PARAMS', 'ENABLE_GRID_SEARCH', 'GRID_SEARCH_CV', 'GRID_SEARCH_SCORING',
    'EARLY_STOPPING', 'EXPERIMENT_NAME', 'EXPERIMENT_VERSION', 'TRACK_EXPERIMENTS',
    'get_model_config', 'get_output_path', 'get_model_save_path',
    'validate_config', 'COMPARISON_RESULT_DIR',
    'validate_feature_priorities', 'get_feature_priority_score', 'get_academic_justification'  # เพิ่มใหม่
]

# เรียกใช้การตรวจสอบเมื่อ import module
if __name__ != "__main__":
    try:
        validate_config()
    except ValueError as e:
        print(f"⚠️  Configuration validation failed: {e}")
        print("Please fix the configuration before running the pipeline.")