# ไฟล์: feature_selection.py
# Path: src/feature_selection.py
# วัตถุประสงค์: Step 2 - Feature Selection ด้วยวิธี Forward/Backward Selection (แก้แล้ว)

"""
feature_selection.py - ขั้นตอนการคัดเลือก Features ที่สำคัญ
"""

import pandas as pd
import numpy as np
import logging 
from sklearn.feature_selection import (
    SequentialFeatureSelector, RFE, RFECV,
    SelectKBest, chi2, mutual_info_classif, f_classif,
    SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from typing import Dict, Tuple, Any, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# แก้ imports เป็น relative imports
from .config import *
from .utils import *

class FeatureSelector:
    """คลาสสำหรับการคัดเลือก Features"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.selector = None
        self.selected_features = []
        self.feature_scores = {}
        self.selection_results = {}
        
    def load_normalized_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """โหลดข้อมูลที่ผ่าน normalization แล้ว"""
        self.logger.info("Loading normalized data...")
        
        # โหลดข้อมูล train/test
        train_data = load_data(get_output_path('preprocessing', 'train_data.csv'))
        test_data = load_data(get_output_path('preprocessing', 'test_data.csv'))
        
        # แยก features และ target
        X_train = train_data.drop(TARGET_COLUMN, axis=1)
        y_train = train_data[TARGET_COLUMN]
        X_test = test_data.drop(TARGET_COLUMN, axis=1)
        y_test = test_data[TARGET_COLUMN]
        
        # ลบ features ที่ไม่ต้องการ
        features_to_exclude = [f for f in EXCLUDE_FEATURES if f in X_train.columns]
        if features_to_exclude:
            X_train = X_train.drop(features_to_exclude, axis=1)
            X_test = X_test.drop(features_to_exclude, axis=1)
            self.logger.info(f"Excluded {len(features_to_exclude)} features: {features_to_exclude}")
        
        self.logger.info(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def sequential_feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """ทำ Sequential Feature Selection (Forward/Backward)"""
        self.logger.info(f"Starting Sequential Feature Selection ({FEATURE_SELECTION_METHOD})...")
        
        # เลือก estimator สำหรับ feature selection
        estimator = RandomForestClassifier(
            n_estimators=50,  # ใช้น้อยกว่าปกติเพื่อความเร็ว
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        # ตั้งค่า direction
        direction = 'forward' if FEATURE_SELECTION_METHOD == 'forward' else 'backward'
        
        # สร้าง selector
        self.selector = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=N_FEATURES_TO_SELECT,
            direction=direction,
            scoring=FEATURE_SELECTION_SCORING,
            cv=CV_FOLDS,
            n_jobs=CV_N_JOBS
        )
        
        # ทำ feature selection
        self.logger.info("Fitting Sequential Feature Selector...")
        X_selected = self.selector.fit_transform(X_train, y_train)
        
        # เก็บผลลัพธ์
        selected_mask = self.selector.get_support()
        self.selected_features = X_train.columns[selected_mask].tolist()
        
        # คำนวณ scores
        scores = cross_val_score(estimator, X_selected, y_train, 
                               cv=CV_FOLDS, scoring=FEATURE_SELECTION_SCORING)
        
        results = {
            'method': f'Sequential_{direction}',
            'n_features_selected': len(self.selected_features),
            'selected_features': self.selected_features,
            'cv_scores': scores.tolist(),
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'feature_mask': selected_mask.tolist()
        }
        
        self.logger.info(f"Sequential FS completed - Selected {len(self.selected_features)} features")
        self.logger.info(f"Cross-validation score: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        
        return results
    
    def rfe_feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """ทำ Recursive Feature Elimination"""
        self.logger.info("Starting RFE Feature Selection...")
        
        estimator = RandomForestClassifier(
            n_estimators=50,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        # RFE with Cross-Validation
        rfe_cv = RFECV(
            estimator=estimator,
            min_features_to_select=MIN_FEATURES,
            cv=CV_FOLDS,
            scoring=FEATURE_SELECTION_SCORING,
            n_jobs=CV_N_JOBS
        )
        
        X_selected = rfe_cv.fit_transform(X_train, y_train)
        
        # เก็บผลลัพธ์
        selected_mask = rfe_cv.get_support()
        selected_features = X_train.columns[selected_mask].tolist()
        
        results = {
            'method': 'RFE_CV',
            'n_features_selected': len(selected_features),
            'selected_features': selected_features,
            'optimal_n_features': rfe_cv.n_features_,
            'cv_scores': rfe_cv.cv_results_['mean_test_score'].tolist(),
            'feature_ranking': rfe_cv.ranking_.tolist(),
            'feature_mask': selected_mask.tolist()
        }
        
        self.logger.info(f"RFE-CV completed - Selected {len(selected_features)} features")
        
        return results
    
    def filter_feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """ทำ Filter-based Feature Selection"""
        self.logger.info("Starting Filter Feature Selection...")
        
        results = {}
        
        # 1. Mutual Information
        mi_selector = SelectKBest(mutual_info_classif, k=N_FEATURES_TO_SELECT)
        X_mi = mi_selector.fit_transform(X_train, y_train)
        mi_features = X_train.columns[mi_selector.get_support()].tolist()
        mi_scores = mi_selector.scores_
        
        results['mutual_info'] = {
            'method': 'Mutual_Information',
            'selected_features': mi_features,
            'feature_scores': dict(zip(X_train.columns, mi_scores))
        }
        
        # 2. ANOVA F-test
        f_selector = SelectKBest(f_classif, k=N_FEATURES_TO_SELECT)
        X_f = f_selector.fit_transform(X_train, y_train)
        f_features = X_train.columns[f_selector.get_support()].tolist()
        f_scores = f_selector.scores_
        
        results['anova_f'] = {
            'method': 'ANOVA_F_test',
            'selected_features': f_features,
            'feature_scores': dict(zip(X_train.columns, f_scores))
        }
        
        self.logger.info("Filter Feature Selection completed")
        
        return results
    
    def embedded_feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """ทำ Embedded Feature Selection"""
        self.logger.info("Starting Embedded Feature Selection...")
        
        results = {}
        
        # 1. LASSO Regularization
        lasso = LassoCV(cv=CV_FOLDS, random_state=RANDOM_STATE, max_iter=1000)
        lasso_selector = SelectFromModel(lasso)
        X_lasso = lasso_selector.fit_transform(X_train, y_train)
        lasso_features = X_train.columns[lasso_selector.get_support()].tolist()
        
        results['lasso'] = {
            'method': 'LASSO_Regularization',
            'selected_features': lasso_features,
            'n_features_selected': len(lasso_features)
        }
        
        # 2. Random Forest Feature Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # เลือก features ตาม importance threshold
        rf_selector = SelectFromModel(rf, threshold='median')
        X_rf = rf_selector.fit_transform(X_train, y_train)
        rf_features = X_train.columns[rf_selector.get_support()].tolist()
        
        # Feature importance scores
        feature_importance = dict(zip(X_train.columns, rf.feature_importances_))
        
        results['random_forest'] = {
            'method': 'RF_Feature_Importance',
            'selected_features': rf_features,
            'feature_importance': feature_importance,
            'n_features_selected': len(rf_features)
        }
        
        self.logger.info("Embedded Feature Selection completed")
        
        return results
    
    def compare_feature_selection_methods(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """เปรียบเทียบวิธี Feature Selection ต่างๆ"""
        self.logger.info("Comparing feature selection methods...")
        
        comparison_results = {}
        estimator = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
        
        # 1. Sequential (หลัก)
        if FEATURE_SELECTION_METHOD in ['forward', 'backward']:
            seq_results = self.sequential_feature_selection(X_train, y_train)
            comparison_results['sequential'] = seq_results
        
        # 2. RFE-CV
        rfe_results = self.rfe_feature_selection(X_train, y_train)
        comparison_results['rfe_cv'] = rfe_results
        
        # 3. Filter methods
        filter_results = self.filter_feature_selection(X_train, y_train)
        comparison_results.update(filter_results)
        
        # 4. Embedded methods
        embedded_results = self.embedded_feature_selection(X_train, y_train)
        comparison_results.update(embedded_results)
        
        # 5. All features (baseline)
        baseline_scores = cross_val_score(estimator, X_train, y_train, 
                                        cv=CV_FOLDS, scoring=FEATURE_SELECTION_SCORING)
        comparison_results['all_features'] = {
            'method': 'All_Features_Baseline',
            'n_features_selected': X_train.shape[1],
            'selected_features': X_train.columns.tolist(),
            'cv_scores': baseline_scores.tolist(),
            'mean_score': baseline_scores.mean(),
            'std_score': baseline_scores.std()
        }
        
        self.logger.info("Feature selection comparison completed")
        
        return comparison_results
    
    def create_selected_dataset(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                              y_train: pd.Series, y_test: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """สร้าง dataset ที่ใช้ features ที่เลือกแล้ว"""
        self.logger.info("Creating dataset with selected features...")
        
        if not self.selected_features:
            raise ValueError("No features selected. Please run feature selection first.")
        
        # สร้าง dataset ใหม่
        X_train_selected = X_train[self.selected_features].copy()
        X_test_selected = X_test[self.selected_features].copy()
        
        # รวมกับ target
        train_selected = pd.concat([X_train_selected, y_train], axis=1)
        test_selected = pd.concat([X_test_selected, y_test], axis=1)
        
        # บันทึกไฟล์
        save_data(train_selected, get_output_path('feature_selection', 'train_data_selected.csv'))
        save_data(test_selected, get_output_path('feature_selection', 'test_data_selected.csv'))
        
        # บันทึกข้อมูลรวม
        full_selected = pd.concat([train_selected, test_selected], axis=0, ignore_index=True)
        save_data(full_selected, get_output_path('feature_selection', 'data_selection.csv'))
        
        self.logger.info(f"Selected dataset saved - Train: {train_selected.shape}, Test: {test_selected.shape}")
        
        return train_selected, test_selected
    
    def create_feature_selection_report(self, comparison_results: Dict) -> Dict:
        """สร้างรายงานการคัดเลือก Features"""
        self.logger.info("Creating feature selection report...")
        
        # สรุปผลการเปรียบเทียบ
        method_summary = {}
        for method_name, results in comparison_results.items():
            if 'mean_score' in results:
                method_summary[method_name] = {
                    'accuracy': results['mean_score'],
                    'std': results['std_score'],
                    'n_features': results['n_features_selected'],
                    'features': results['selected_features'][:5]  # แสดงแค่ 5 ตัวแรก
                }
        
        # หาวิธีที่ดีที่สุด
        best_method = max(method_summary.keys(), 
                         key=lambda x: method_summary[x]['accuracy'])
        
        # สร้างรายงาน
        report = {
            'feature_selection_config': {
                'method': FEATURE_SELECTION_METHOD,
                'n_features_to_select': N_FEATURES_TO_SELECT,
                'scoring_metric': FEATURE_SELECTION_SCORING,
                'cv_folds': CV_FOLDS
            },
            'comparison_results': comparison_results,
            'method_summary': method_summary,
            'best_method': {
                'name': best_method,
                'accuracy': method_summary[best_method]['accuracy'],
                'std': method_summary[best_method]['std'],
                'n_features': method_summary[best_method]['n_features']
            },
            'final_selection': {
                'method_used': FEATURE_SELECTION_METHOD,
                'selected_features': self.selected_features,
                'n_features_selected': len(self.selected_features)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def run_feature_selection(self) -> Tuple[pd.DataFrame, Dict]:
        """รันขั้นตอนการคัดเลือก Features ทั้งหมด (เอา decorator ออก)"""
        self.logger.info("Starting feature selection pipeline...")
        
        try:
            tracker = ProgressTracker(6, "Feature Selection")
            
            # 1. โหลดข้อมูล
            X_train, X_test, y_train, y_test = self.load_normalized_data()
            tracker.update("Loading normalized data")
            
            # 2. เปรียบเทียบวิธีต่างๆ
            comparison_results = self.compare_feature_selection_methods(X_train, y_train)
            tracker.update("Comparing feature selection methods")
            
            # 3. เลือกใช้วิธีหลัก (Sequential)
            if FEATURE_SELECTION_METHOD in ['forward', 'backward']:
                main_results = comparison_results['sequential']
                self.selected_features = main_results['selected_features']
            else:
                # ใช้ RFE-CV เป็นวิธีหลัก
                main_results = comparison_results['rfe_cv']
                self.selected_features = main_results['selected_features']
            
            tracker.update("Selecting primary method")
            
            # 4. สร้าง dataset ที่ใช้ features ที่เลือก
            train_selected, test_selected = self.create_selected_dataset(X_train, X_test, y_train, y_test)
            tracker.update("Creating selected dataset")
            
            # 5. สร้างรายงาน
            report = self.create_feature_selection_report(comparison_results)
            tracker.update("Creating selection report")
            
            # 6. บันทึกผลลัพธ์
            save_json(report, get_output_path('feature_selection', 'feature_selection_report.json'))
            save_json({'selected_features': self.selected_features}, 
                     get_output_path('feature_selection', 'selected_features.json'))
            
            # บันทึก feature scores
            if 'feature_importance' in comparison_results.get('random_forest', {}):
                feature_scores_df = pd.DataFrame([
                    {'feature': feat, 'importance': score}
                    for feat, score in comparison_results['random_forest']['feature_importance'].items()
                ]).sort_values('importance', ascending=False)
                
                save_data(feature_scores_df, get_output_path('feature_selection', 'selection_scores.csv'))
            
            tracker.update("Saving results")
            tracker.finish()
            
            if VERBOSE:
                print_summary("Feature Selection Results", {
                    'Method Used': FEATURE_SELECTION_METHOD,
                    'Features Selected': len(self.selected_features),
                    'Original Features': X_train.shape[1],
                    'Reduction': f"{(1 - len(self.selected_features)/X_train.shape[1])*100:.1f}%",
                    'Primary Method Score': f"{main_results.get('mean_score', 0):.4f}",
                    'Selected Features': self.selected_features[:10]  # แสดง 10 ตัวแรก
                })
            
            self.logger.info("Feature selection completed successfully")
            
            return train_selected, report
            
        except Exception as e:
            self.logger.error(f"Feature selection failed: {str(e)}")
            raise

def main():
    """ฟังก์ชันหลักสำหรับการทดสอบ"""
    try:
        selector = FeatureSelector()
        data_selected, report = selector.run_feature_selection()
        
        print("✅ Feature selection completed successfully!")
        print(f"📊 Selected features: {len(selector.selected_features)}")
        print(f"📁 Results saved to: {FEATURE_SELECTION_RESULT_DIR}")
        
        return data_selected, report
        
    except Exception as e:
        logger.error(f"Feature selection failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()