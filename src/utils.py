# ไฟล์: utils.py
# Path: src/utils.py
# วัตถุประสงค์: Helper functions สำหรับทั้งโปรเจค (แก้แล้ว - แก้ circular import)

"""
utils.py - Helper functions สำหรับทั้งโปรเจค
"""

import os
import json
import pickle
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# ==================== LOGGING SETUP ====================
def setup_logging():
    """ตั้งค่า logging สำหรับโปรเจค"""
    # Import config here to avoid circular import
    from .config import LOGS_DIR, LOGGING_CONFIG
    
    # สร้างโฟลเดอร์ logs ถ้ายังไม่มี
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ตั้งค่า logging
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(LOGGING_CONFIG['file'], encoding='utf-8'),
            logging.StreamHandler() if LOGGING_CONFIG['console'] else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ==================== FILE I/O FUNCTIONS ====================
def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """โหลดข้อมูลจากไฟล์"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading data from: {file_path}")
    
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    elif file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    elif file_path.suffix == '.pickle' or file_path.suffix == '.pkl':
        df = pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

def save_data(df: pd.DataFrame, file_path: Union[str, Path], format_type: str = None):
    """บันทึกข้อมูลลงไฟล์"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type is None:
        # Import here to avoid circular import
        from .config import OUTPUT_FORMATS
        format_type = OUTPUT_FORMATS['data']
    
    logger.info(f"Saving data to: {file_path}")
    
    if format_type == 'csv':
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
    elif format_type == 'parquet':
        df.to_parquet(file_path, index=False)
    elif format_type == 'pickle':
        df.to_pickle(file_path)
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    logger.info(f"Data saved successfully")

def save_json(data: Dict, file_path: Union[str, Path]):
    """บันทึกข้อมูล JSON"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving JSON to: {file_path}")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info("JSON saved successfully")

def load_json(file_path: Union[str, Path]) -> Dict:
    """โหลดข้อมูล JSON"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading JSON from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info("JSON loaded successfully")
    return data

def save_model(model, file_path: Union[str, Path], format_type: str = None):
    """บันทึกโมเดล"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type is None:
        # Import here to avoid circular import
        from .config import OUTPUT_FORMATS
        format_type = OUTPUT_FORMATS['models']
    
    logger.info(f"Saving model to: {file_path}")
    
    if format_type == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
    elif format_type == 'joblib':
        joblib.dump(model, file_path)
    else:
        raise ValueError(f"Unsupported model format: {format_type}")
    
    logger.info("Model saved successfully")

def load_model(file_path: Union[str, Path], format_type: str = None):
    """โหลดโมเดล"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    if format_type is None:
        # Import here to avoid circular import
        from .config import OUTPUT_FORMATS
        format_type = OUTPUT_FORMATS['models']
    
    logger.info(f"Loading model from: {file_path}")
    
    if format_type == 'pickle':
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
    elif format_type == 'joblib':
        model = joblib.load(file_path)
    else:
        raise ValueError(f"Unsupported model format: {format_type}")
    
    logger.info("Model loaded successfully")
    return model

# ==================== DATA ANALYSIS FUNCTIONS ====================
def analyze_data(df: pd.DataFrame) -> Dict:
    """วิเคราะห์ข้อมูลเบื้องต้น"""
    logger.info("Analyzing data...")
    
    # Import here to avoid circular import
    from .config import TARGET_COLUMN
    
    analysis = {
        'basic_info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        },
        'missing_values': {
            'count': df.isnull().sum().to_dict(),
            'percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        },
        'summary_statistics': {},
        'categorical_info': {},
        'target_distribution': {}
    }
    
    # สถิติเชิงพรรณนา
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        analysis['summary_statistics'] = df[numeric_cols].describe().to_dict()
    
    # ข้อมูล categorical
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        analysis['categorical_info'][col] = {
            'unique_count': df[col].nunique(),
            'unique_values': df[col].unique().tolist(),
            'value_counts': df[col].value_counts().to_dict()
        }
    
    # การกระจายของ target variable
    if TARGET_COLUMN in df.columns:
        analysis['target_distribution'] = {
            'value_counts': df[TARGET_COLUMN].value_counts().to_dict(),
            'percentage': (df[TARGET_COLUMN].value_counts() / len(df) * 100).to_dict()
        }
    
    logger.info("Data analysis completed")
    return analysis

def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> Dict:
    """ตรวจหา outliers"""
    logger.info(f"Detecting outliers using {method} method...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_indices': df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist(),
                'outlier_count': len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
            }
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            threshold = 3
            outlier_indices = df[z_scores > threshold].index.tolist()
            
            outliers[col] = {
                'threshold': threshold,
                'outlier_indices': outlier_indices,
                'outlier_count': len(outlier_indices)
            }
    
    logger.info(f"Outlier detection completed. Found outliers in {len(outliers)} columns")
    return outliers

# ==================== EVALUATION FUNCTIONS ====================
def calculate_metrics(y_true: np.array, y_pred: np.array, y_prob: np.array = None) -> Dict:
    """คำนวณ metrics ต่างๆ"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    
    # AUC สำหรับ multi-class
    if y_prob is not None and len(np.unique(y_true)) > 2:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except ValueError:
            logger.warning("Could not calculate AUC score")
            metrics['auc'] = None
    
    return metrics

def statistical_significance_test(scores1: List[float], scores2: List[float], 
                                test_type: str = 'paired_ttest') -> Dict:
    """ทดสอบนัยสำคัญทางสถิติ"""
    # Import here to avoid circular import
    from .config import SIGNIFICANCE_LEVEL
    
    from scipy import stats
    
    if test_type == 'paired_ttest':
        statistic, p_value = stats.ttest_rel(scores1, scores2)
    elif test_type == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(scores1, scores2)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    result = {
        'test_type': test_type,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < SIGNIFICANCE_LEVEL,
        'significance_level': SIGNIFICANCE_LEVEL
    }
    
    return result

# ==================== VISUALIZATION FUNCTIONS ====================
def setup_plot_style():
    """ตั้งค่า style สำหรับ plot"""
    # Import here to avoid circular import
    from .config import PLOT_SETTINGS
    
    plt.style.use('default')  # เปลี่ยนจาก seaborn-v0_8 เป็น default
    sns.set_palette(PLOT_SETTINGS['palette'])
    plt.rcParams['figure.figsize'] = PLOT_SETTINGS['figsize']
    plt.rcParams['figure.dpi'] = PLOT_SETTINGS['dpi']
    plt.rcParams['font.size'] = PLOT_SETTINGS['font_size']

def save_plot(fig, filename: str, category: str = 'evaluation'):
    """บันทึก plot"""
    # Import here to avoid circular import
    from .config import GENERATE_PLOTS, PLOT_SETTINGS, get_output_path
    
    if not GENERATE_PLOTS:
        return
    
    file_path = get_output_path(category, filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving plot to: {file_path}")
    
    fig.savefig(file_path, dpi=PLOT_SETTINGS['dpi'], 
                bbox_inches='tight', format=PLOT_SETTINGS['save_format'])
    
    logger.info(f"Plot saved: {file_path}")

def plot_confusion_matrix(cm: np.array, labels: List[str], title: str = "Confusion Matrix"):
    """สร้าง confusion matrix plot"""
    # Import here to avoid circular import
    from .config import PLOT_SETTINGS
    
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_title(title, fontsize=PLOT_SETTINGS['title_size'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    return fig

def plot_feature_importance(importances: Dict, title: str = "Feature Importance", top_n: int = 15):
    """สร้าง feature importance plot"""
    # Import here to avoid circular import
    from .config import PLOT_SETTINGS
    
    setup_plot_style()
    
    # เรียงลำดับความสำคัญ
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    features = [item[0] for item in sorted_features[:top_n]]
    values = [item[1] for item in sorted_features[:top_n]]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.barh(features, values)
    ax.set_title(title, fontsize=PLOT_SETTINGS['title_size'])
    ax.set_xlabel('Importance Score')
    
    # เพิ่มค่าที่แท่งกราฟ
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    return fig

def plot_model_comparison(results: Dict, metric: str = 'accuracy'):
    """สร้างกราฟเปรียบเทียบโมเดล"""
    # Import here to avoid circular import
    from .config import PLOT_SETTINGS
    
    setup_plot_style()
    
    models = list(results.keys())
    values = [results[model][metric]['mean'] for model in models]
    errors = [results[model][metric]['std'] for model in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, values, yerr=errors, capsize=5, 
                  alpha=0.8, edgecolor='black')
    
    ax.set_title(f'Model Comparison - {metric.title()}', fontsize=PLOT_SETTINGS['title_size'])
    ax.set_ylabel(f'{metric.title()} Score')
    ax.set_ylim(0, 1.0)
    
    # เพิ่มค่าที่แท่งกราฟ
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# ==================== PROGRESS TRACKING ====================
class ProgressTracker:
    """คลาสสำหรับติดตามความคืบหน้า"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
        logger.info(f"Starting {description} - {total_steps} steps")
    
    def update(self, step_name: str = ""):
        """อัพเดทความคืบหน้า"""
        self.current_step += 1
        percentage = (self.current_step / self.total_steps) * 100
        
        elapsed_time = datetime.now() - self.start_time
        
        # Import here to avoid circular import
        try:
            from .config import VERBOSE
            verbose = VERBOSE
        except ImportError:
            verbose = True
        
        if verbose:
            message = f"Step {self.current_step}/{self.total_steps} ({percentage:.1f}%)"
            if step_name:
                message += f" - {step_name}"
            message += f" - Elapsed: {elapsed_time}"
            
            logger.info(message)
    
    def finish(self):
        """จบการทำงาน"""
        total_time = datetime.now() - self.start_time
        logger.info(f"{self.description} completed in {total_time}")

# ==================== VALIDATION FUNCTIONS ====================
def validate_data(df: pd.DataFrame) -> Dict:
    """ตรวจสอบความถูกต้องของข้อมูล"""
    logger.info("Validating data...")
    
    # Import here to avoid circular import
    from .config import TARGET_COLUMN, CORE_FEATURES, DEMOGRAPHIC_FEATURES, LIFESTYLE_FEATURES
    
    issues = {
        'missing_target': False,
        'missing_features': [],
        'invalid_dtypes': [],
        'duplicate_rows': 0,
        'constant_features': [],
        'high_cardinality_features': []
    }
    
    # ตรวจสอบ target column
    if TARGET_COLUMN not in df.columns:
        issues['missing_target'] = True
        logger.error(f"Target column '{TARGET_COLUMN}' not found")
    
    # ตรวจสอบ features ที่จำเป็น
    required_features = CORE_FEATURES + DEMOGRAPHIC_FEATURES + LIFESTYLE_FEATURES
    for feature in required_features:
        if feature not in df.columns:
            issues['missing_features'].append(feature)
    
    # ตรวจสอบ duplicate rows
    issues['duplicate_rows'] = df.duplicated().sum()
    
    # ตรวจสอบ constant features
    for col in df.columns:
        if df[col].nunique() <= 1:
            issues['constant_features'].append(col)
    
    # ตรวจสอบ high cardinality features
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > len(df) * 0.8:  # มากกว่า 80% ของจำนวนแถว
            issues['high_cardinality_features'].append(col)
    
    # สรุปผลการตรวจสอบ
    total_issues = sum([
        issues['missing_target'],
        len(issues['missing_features']),
        len(issues['invalid_dtypes']),
        1 if issues['duplicate_rows'] > 0 else 0,
        len(issues['constant_features']),
        len(issues['high_cardinality_features'])
    ])
    
    if total_issues == 0:
        logger.info("Data validation passed - no issues found")
    else:
        logger.warning(f"Data validation found {total_issues} issues")
    
    return issues

# ==================== UTILITY FUNCTIONS ====================
def create_experiment_id() -> str:
    """สร้าง ID สำหรับ experiment"""
    # Import here to avoid circular import
    from .config import EXPERIMENT_NAME
    return f"{EXPERIMENT_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def get_memory_usage() -> Dict:
    """ตรวจสอบการใช้ memory"""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }

def print_summary(title: str, data: Dict):
    """พิมพ์สรุปข้อมูล"""
    # Import here to avoid circular import
    try:
        from .config import VERBOSE
        verbose = VERBOSE
    except ImportError:
        verbose = True
    
    if not verbose:
        return
    
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    print(f"{'='*50}\n")

# ==================== ERROR HANDLING ====================
class PipelineError(Exception):
    """Custom exception สำหรับ pipeline errors"""
    pass

def handle_pipeline_error(func):
    """Decorator สำหรับจัดการ errors ใน pipeline"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise PipelineError(f"Pipeline failed at {func.__name__}: {str(e)}")
    return wrapper

def get_display_labels(thai_labels=None):
    """แปลง labels ภาษาไทยเป็นภาษาอังกฤษสำหรับแสดงผล"""
    thai_to_english = {
        'บัญชี': 'ACC',
        'สารสนเทศ': 'IT', 
        'อาหาร': 'FOOD'
    }
    
    if thai_labels is None:
        return list(thai_to_english.values())
    
    return [thai_to_english.get(label, label) for label in thai_labels]



def get_english_feature_names(thai_features=None):
    """แปลงชื่อ features ภาษาไทยเป็นภาษาอังกฤษ"""
    
    feature_mapping = {
        # คะแนนวิชา
        'คะแนนคณิตศาสตร์': 'Math Score',
        'คะแนนคอมพิวเตอร์': 'Computer Score', 
        'คะแนนภาษาไทย': 'Thai Score',
        'คะแนนวิทยาศาสตร์': 'Science Score',
        'คะแนนศิลปะ': 'Art Score',
        
        # ทักษะ
        'ทักษะการคิดเชิงตรรกะ': 'Logical Thinking',
        'ทักษะความคิดสร้างสรรค์': 'Creative Thinking',
        'ทักษะการแก้ปัญหา': 'Problem Solving',
        
        # ความสนใจ
        'ความสนใจด้านตัวเลข': 'Interest in Numbers',
        'ความสนใจด้านเทคโนโลยี': 'Interest in Technology', 
        'ความสนใจด้านการทำอาหาร': 'Interest in Cooking',
        
        # ข้อมูลส่วนตัว
        'อายุ': 'Age',
        'เพศ': 'Gender',
        'รายได้ครอบครัว': 'Family Income',
        'การศึกษาของผู้ปกครอง': 'Parent Education',
        'จำนวนพี่น้อง': 'Number of Siblings',
        
        # ไลฟ์สไตล์
        'ชั่วโมงการนอน': 'Sleep Hours',
        'ความถี่การออกกำลังกาย': 'Exercise Frequency',
        'ชั่วโมงใช้โซเชียลมีเดีย': 'Social Media Hours',
        'ชอบอ่านหนังสือ': 'Like Reading',
        'ประเภทเพลงที่ชอบ': 'Music Preference'
    }
    
    if thai_features is None:
        return feature_mapping
    
    if isinstance(thai_features, list):
        return [feature_mapping.get(feature, feature) for feature in thai_features]
    else:
        return feature_mapping.get(thai_features, thai_features)




# ==================== EXPORTS ====================
__all__ = [
    'setup_logging', 'logger',
    'load_data', 'save_data', 'save_json', 'load_json',
    'save_model', 'load_model',
    'analyze_data', 'detect_outliers',
    'calculate_metrics', 'statistical_significance_test',
    'setup_plot_style', 'save_plot', 'plot_confusion_matrix', 
    'plot_feature_importance', 'plot_model_comparison',
    'ProgressTracker', 'validate_data',
    'create_experiment_id', 'get_memory_usage', 'print_summary',
    'PipelineError', 'handle_pipeline_error' ,'get_display_labels','get_english_feature_names'
]