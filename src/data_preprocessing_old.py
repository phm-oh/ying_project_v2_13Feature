# ไฟล์: data_preprocessing.py
# Path: src/data_preprocessing.py
# วัตถุประสงค์: Step 1 - Normalization (แก้แล้ว - แก้ imports และเอา decorator ออก)

"""
data_preprocessing.py - ขั้นตอนการทำ Data Normalization และการเตรียมข้อมูล
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    LabelEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Any, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# แก้ imports เพื่อหลีกเลี่ยง circular import
from .config import (
    TARGET_COLUMN, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE,
    CORE_FEATURES, DEMOGRAPHIC_FEATURES, LIFESTYLE_FEATURES, VALIDATION_FEATURES,
    EXCLUDE_FEATURES, NORMALIZATION_METHOD, MISSING_VALUE_STRATEGY, CATEGORICAL_ENCODING,
    VERBOSE, get_output_path, DATA_PATH, PREPROCESSING_RESULT_DIR
)
from .utils import (
    setup_logging, logger, load_data, save_data, save_json, load_json,
    analyze_data, detect_outliers, ProgressTracker, print_summary,
    handle_pipeline_error, validate_data
)

class DataPreprocessor:
    """คลาสสำหรับการเตรียมข้อมูล"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_stats = {}
        
    def load_raw_data(self) -> pd.DataFrame:
        """โหลดข้อมูลดิบ"""
        self.logger.info("Loading raw data...")
        
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
        
        df = load_data(DATA_PATH)
        self.logger.info(f"Raw data loaded: {df.shape}")
        
        return df
    
    def analyze_raw_data(self, df: pd.DataFrame) -> Dict:
        """วิเคราะห์ข้อมูลดิบ"""
        self.logger.info("Analyzing raw data...")
        
        analysis = analyze_data(df)
        
        # เพิ่มการวิเคราะห์เฉพาะสำหรับ features groups
        analysis['feature_groups'] = {
            'core_features': {
                'count': len([f for f in CORE_FEATURES if f in df.columns]),
                'missing': [f for f in CORE_FEATURES if f not in df.columns]
            },
            'demographic_features': {
                'count': len([f for f in DEMOGRAPHIC_FEATURES if f in df.columns]),
                'missing': [f for f in DEMOGRAPHIC_FEATURES if f not in df.columns]
            },
            'lifestyle_features': {
                'count': len([f for f in LIFESTYLE_FEATURES if f in df.columns]),
                'missing': [f for f in LIFESTYLE_FEATURES if f not in df.columns]
            },
            'validation_features': {
                'count': len([f for f in VALIDATION_FEATURES if f in df.columns]),
                'missing': [f for f in VALIDATION_FEATURES if f not in df.columns]
            }
        }
        
        # ตรวจหา outliers
        outliers = detect_outliers(df, method='iqr')
        analysis['outliers'] = outliers
        
        # บันทึกผลการวิเคราะห์
        save_json(analysis, get_output_path('preprocessing', 'raw_data_analysis.json'))
        
        if VERBOSE:
            print_summary("Raw Data Analysis", {
                'Shape': df.shape,
                'Missing Values': sum(analysis['missing_values']['count'].values()),
                'Numeric Columns': len(df.select_dtypes(include=[np.number]).columns),
                'Categorical Columns': len(df.select_dtypes(include=['object']).columns),
                'Target Classes': len(df[TARGET_COLUMN].unique()) if TARGET_COLUMN in df.columns else 'N/A'
            })
        
        return analysis
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ทำความสะอาดข้อมูล"""
        self.logger.info("Cleaning data...")
        
        df_clean = df.copy()
        
        # ลบ duplicate rows
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        
        if removed_duplicates > 0:
            self.logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # ตรวจสอบและทำความสะอาด column names
        df_clean.columns = df_clean.columns.str.strip()
        
        # ลบ features ที่ไม่ต้องการ (validation features สำหรับ ML)
        features_to_exclude = [f for f in EXCLUDE_FEATURES if f in df_clean.columns]
        if features_to_exclude:
            self.logger.info(f"Excluding features for ML: {features_to_exclude}")
        
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """จัดการ missing values"""
        self.logger.info("Handling missing values...")
        
        df_imputed = df.copy()
        
        # แยก numeric และ categorical columns
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_imputed.select_dtypes(include=['object', 'category']).columns
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            if MISSING_VALUE_STRATEGY in ['mean', 'median']:
                imputer = SimpleImputer(strategy=MISSING_VALUE_STRATEGY)
                df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
                self.imputers['numeric'] = imputer
            elif MISSING_VALUE_STRATEGY == 'drop':
                df_imputed = df_imputed.dropna(subset=numeric_cols)
        
        # Impute categorical columns
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df_imputed[categorical_cols] = imputer.fit_transform(df_imputed[categorical_cols])
            self.imputers['categorical'] = imputer
        
        missing_after = df_imputed.isnull().sum().sum()
        self.logger.info(f"Missing values after imputation: {missing_after}")
        
        return df_imputed
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        self.logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        # ลบ target column ออกจาก categorical encoding
        categorical_cols = [col for col in categorical_cols if col != TARGET_COLUMN]
        
        for col in categorical_cols:
            if CATEGORICAL_ENCODING == 'label':
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col])
                self.encoders[col] = encoder
                
            elif CATEGORICAL_ENCODING == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                self.encoders[col] = list(dummies.columns)
        
        # Encode target variable
        if TARGET_COLUMN in df_encoded.columns:
            target_encoder = LabelEncoder()
            df_encoded[TARGET_COLUMN] = target_encoder.fit_transform(df_encoded[TARGET_COLUMN])
            self.encoders[TARGET_COLUMN] = target_encoder
            
            # บันทึก mapping ของ target classes
            target_mapping = dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))
            save_json(target_mapping, get_output_path('preprocessing', 'target_mapping.json'))
        
        self.logger.info(f"Encoded {len(categorical_cols)} categorical features")
        return df_encoded
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ทำ normalization ของ features"""
        self.logger.info(f"Normalizing features using {NORMALIZATION_METHOD}...")
        
        df_normalized = df.copy()
        
        # แยก features และ target
        if TARGET_COLUMN in df_normalized.columns:
            X = df_normalized.drop(TARGET_COLUMN, axis=1)
            y = df_normalized[TARGET_COLUMN]
        else:
            X = df_normalized
            y = None
        
        # เลือก scaler
        if NORMALIZATION_METHOD == 'standard':
            scaler = StandardScaler()
        elif NORMALIZATION_METHOD == 'minmax':
            scaler = MinMaxScaler()
        elif NORMALIZATION_METHOD == 'robust':
            scaler = RobustScaler()
        elif NORMALIZATION_METHOD == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal')
        else:
            raise ValueError(f"Unknown normalization method: {NORMALIZATION_METHOD}")
        
        # แยก numeric columns สำหรับ normalization
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # บันทึกสถิติก่อน normalization
            self.feature_stats['before_normalization'] = {
                'mean': X[numeric_cols].mean().to_dict(),
                'std': X[numeric_cols].std().to_dict(),
                'min': X[numeric_cols].min().to_dict(),
                'max': X[numeric_cols].max().to_dict()
            }
            
            # ทำ normalization
            X_normalized = X.copy()
            X_normalized[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            
            # บันทึกสถิติหลัง normalization
            self.feature_stats['after_normalization'] = {
                'mean': X_normalized[numeric_cols].mean().to_dict(),
                'std': X_normalized[numeric_cols].std().to_dict(),
                'min': X_normalized[numeric_cols].min().to_dict(),
                'max': X_normalized[numeric_cols].max().to_dict()
            }
            
            # บันทึก scaler
            self.scalers['features'] = scaler
            
            # รวม target กลับเข้าไป
            if y is not None:
                df_normalized = pd.concat([X_normalized, y], axis=1)
            else:
                df_normalized = X_normalized
        
        self.logger.info(f"Normalized {len(numeric_cols)} numeric features")
        return df_normalized
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """แบ่งข้อมูลเป็น train/test"""
        self.logger.info("Splitting data into train/test sets...")
        
        # แยก features และ target
        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN]
        
        # ลบ features ที่ไม่ต้องการออกจาก X
        features_to_exclude = [f for f in EXCLUDE_FEATURES if f in X.columns]
        if features_to_exclude:
            X = X.drop(features_to_exclude, axis=1)
            self.logger.info(f"Excluded {len(features_to_exclude)} features from ML")
        
        # แบ่งข้อมูล
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
        
        self.logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # บันทึกข้อมูลการแบ่ง
        split_info = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_percentage': len(X_train) / len(X) * 100,
            'test_percentage': len(X_test) / len(X) * 100,
            'features_used': list(X.columns),
            'features_excluded': features_to_exclude,
            'target_distribution_train': y_train.value_counts().to_dict(),
            'target_distribution_test': y_test.value_counts().to_dict()
        }
        
        save_json(split_info, get_output_path('preprocessing', 'data_split_info.json'))
        
        return X_train, X_test, y_train, y_test
    
    def create_preprocessing_report(self) -> Dict:
        """สร้างรายงานการเตรียมข้อมูล"""
        self.logger.info("Creating preprocessing report...")
        
        report = {
            'preprocessing_method': NORMALIZATION_METHOD,
            'missing_value_strategy': MISSING_VALUE_STRATEGY,
            'categorical_encoding': CATEGORICAL_ENCODING,
            'scalers_used': list(self.scalers.keys()),
            'encoders_used': list(self.encoders.keys()),
            'feature_statistics': self.feature_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def run_preprocessing(self) -> Tuple[pd.DataFrame, Dict]:
        """รันขั้นตอนการเตรียมข้อมูลทั้งหมด (เอา decorator ออก)"""
        self.logger.info("Starting data preprocessing pipeline...")
        
        try:
            tracker = ProgressTracker(6, "Data Preprocessing")
            
            # 1. โหลดข้อมูล
            df_raw = self.load_raw_data()
            tracker.update("Loading raw data")
            
            # 2. วิเคราะห์ข้อมูลดิบ
            raw_analysis = self.analyze_raw_data(df_raw)
            tracker.update("Analyzing raw data")
            
            # 3. ทำความสะอาดข้อมูล
            df_clean = self.clean_data(df_raw)
            tracker.update("Cleaning data")
            
            # 4. จัดการ missing values
            df_imputed = self.handle_missing_values(df_clean)
            tracker.update("Handling missing values")
            
            # 5. Encode categorical features
            df_encoded = self.encode_categorical_features(df_imputed)
            tracker.update("Encoding categorical features")
            
            # 6. Normalize features
            df_normalized = self.normalize_features(df_encoded)
            tracker.update("Normalizing features")
            
            tracker.finish()
            
            # บันทึกข้อมูลที่ประมวลผลแล้ว
            save_data(df_normalized, get_output_path('preprocessing', 'data_normalization.csv'))
            
            # สร้างและบันทึกรายงาน
            report = self.create_preprocessing_report()
            save_json(report, get_output_path('preprocessing', 'normalization_report.json'))
            
            # แบ่งข้อมูล train/test
            X_train, X_test, y_train, y_test = self.split_data(df_normalized)
            
            # บันทึกข้อมูลที่แบ่งแล้ว
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
            
            save_data(train_data, get_output_path('preprocessing', 'train_data.csv'))
            save_data(test_data, get_output_path('preprocessing', 'test_data.csv'))
            
            if VERBOSE:
                print_summary("Preprocessing Results", {
                    'Original Shape': df_raw.shape,
                    'Final Shape': df_normalized.shape,
                    'Features Used': len(X_train.columns),
                    'Normalization Method': NORMALIZATION_METHOD,
                    'Train/Test Split': f"{len(X_train)}/{len(X_test)}"
                })
            
            self.logger.info("Data preprocessing completed successfully")
            
            return df_normalized, report
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise

def main():
    """ฟังก์ชันหลักสำหรับการทดสอบ"""
    try:
        preprocessor = DataPreprocessor()
        df_processed, report = preprocessor.run_preprocessing()
        
        print("✅ Data preprocessing completed successfully!")
        print(f"📊 Processed data shape: {df_processed.shape}")
        print(f"📁 Results saved to: {PREPROCESSING_RESULT_DIR}")
        
        return df_processed, report
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()