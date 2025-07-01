# ไฟล์: feature_selection.py
# Path: src/feature_selection.py
# วัตถุประสงค์: Step 2 - Feature Selection ด้วย RFECV vs Baseline (แก้ไขแล้ว)

"""
feature_selection.py - ขั้นตอนการคัดเลือก Features ที่สำคัญ (แก้ไขแล้ว - ใช้ RFECV เป็นหลัก)
"""

import pandas as pd
import numpy as np
import logging 
from sklearn.feature_selection import (
    RFECV, RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from typing import Dict, Tuple, Any, Optional, Union, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# แก้ imports เป็น relative imports
from .config import *
from .utils import *

class FeatureSelector:
    """คลาสสำหรับการคัดเลือก Features (แก้ไขแล้ว - ใช้ RFECV เป็นหลัก)"""
    
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
        
        # ลบ features ที่ไม่ต้องการ (ถ้ามี)
        features_to_exclude = [f for f in EXCLUDE_FEATURES if f in X_train.columns]
        if features_to_exclude:
            X_train = X_train.drop(features_to_exclude, axis=1)
            X_test = X_test.drop(features_to_exclude, axis=1)
            self.logger.info(f"Excluded {len(features_to_exclude)} features: {features_to_exclude}")
        
        self.logger.info(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def rfecv_feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """ทำ RFECV Feature Selection (หลัก)"""
        self.logger.info("Starting RFECV Feature Selection...")
        
        # สร้าง estimator สำหรับ RFECV
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        # สร้าง RFECV
        self.selector = RFECV(
            estimator=estimator,
            min_features_to_select=MIN_FEATURES,  # ทดสอบจนถึง 1 feature
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring=FEATURE_SELECTION_SCORING,
            n_jobs=CV_N_JOBS
        )
        
        self.logger.info(f"RFECV will test from {len(X_train.columns)} down to {MIN_FEATURES} features...")
        
        # ทำ feature selection
        X_selected = self.selector.fit_transform(X_train, y_train)
        
        # เก็บผลลัพธ์
        selected_mask = self.selector.get_support()
        selected_features = X_train.columns[selected_mask].tolist()
        
        # ข้อมูลละเอียด
        n_features_tested = len(self.selector.cv_results_['mean_test_score'])
        optimal_n_features = self.selector.n_features_
        cv_scores = self.selector.cv_results_['mean_test_score']
        cv_std = self.selector.cv_results_['std_test_score']
        
        # สร้าง detailed history
        detailed_history = []
        feature_range = range(len(X_train.columns), MIN_FEATURES - 1, -1)
        
        for i, n_features in enumerate(feature_range):
            if i < len(cv_scores):
                round_info = {
                    'n_features': n_features,
                    'cv_mean': cv_scores[i],
                    'cv_std': cv_std[i],
                    'is_optimal': n_features == optimal_n_features
                }
                detailed_history.append(round_info)
        
        # หาคะแนนสูงสุด
        best_score_idx = np.argmax(cv_scores)
        best_score = cv_scores[best_score_idx]
        
        results = {
            'method': 'RFECV_Auto_Selection',
            'n_features_selected': len(selected_features),
            'selected_features': selected_features,
            'optimal_n_features': optimal_n_features,
            'cv_scores': cv_scores.tolist(),
            'cv_std': cv_std.tolist(),
            'mean_score': best_score,
            'std_score': cv_std[best_score_idx],
            'feature_ranking': self.selector.ranking_.tolist(),
            'feature_mask': selected_mask.tolist(),
            'detailed_history': detailed_history,  # เพิ่มข้อมูลละเอียด
            'n_features_tested': n_features_tested,
            'improvement': 0  # จะคำนวณเมื่อเปรียบเทียบกับ baseline
        }
        
        # เก็บ feature importance (จาก estimator ที่ fit แล้ว)
        if hasattr(self.selector.estimator_, 'feature_importances_'):
            feature_names = X_train.columns.tolist()
            importance_dict = dict(zip(feature_names, self.selector.estimator_.feature_importances_))
            results['feature_importance'] = importance_dict
        
        self.logger.info(f"RFECV completed - Selected {len(selected_features)} features")
        self.logger.info(f"Optimal features: {optimal_n_features}, Best CV score: {best_score:.4f}")
        self.logger.info(f"Selected features: {selected_features}")
        
        return results
    
    def compare_feature_selection_methods(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """เปรียบเทียบ RFECV กับ Baseline (แก้ไขแล้ว)"""
        self.logger.info("Comparing RFECV vs Baseline...")
        
        comparison_results = {}
        estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        
        # 1. RFECV Feature Selection (หลัก)
        self.logger.info("Running RFECV...")
        rfecv_results = self.rfecv_feature_selection(X_train, y_train)
        comparison_results['rfecv'] = rfecv_results
        
        # 2. All features (baseline)
        self.logger.info("Running Baseline (All Features)...")
        baseline_scores = cross_val_score(estimator, X_train, y_train, 
                                        cv=CV_FOLDS, scoring=FEATURE_SELECTION_SCORING)
        comparison_results['all_features'] = {
            'method': 'All_Features_Baseline',
            'n_features_selected': len(X_train.columns),
            'selected_features': X_train.columns.tolist(),
            'cv_scores': baseline_scores.tolist(),
            'mean_score': baseline_scores.mean(),
            'std_score': baseline_scores.std()
        }
        
        # คำนวณ improvement
        baseline_score = baseline_scores.mean()
        rfecv_score = rfecv_results['mean_score']
        improvement = rfecv_score - baseline_score
        
        # อัพเดท improvement ใน RFECV results
        comparison_results['rfecv']['improvement'] = improvement
        
        # หาวิธีที่ดีที่สุด (แก้บัค)
        if rfecv_score > baseline_score:
            best_method = 'rfecv'
            best_score = rfecv_score
        else:
            best_method = 'all_features'
            best_score = baseline_score
        
        comparison_results['best_method'] = {
            'name': best_method,
            'score': best_score
        }
        
        # ใช้ features จากวิธีที่ดีที่สุด
        if best_method == 'rfecv':
            self.selected_features = comparison_results['rfecv']['selected_features']
        else:
            self.selected_features = comparison_results['all_features']['selected_features']
        
        self.logger.info(f"Comparison completed - Best: {best_method} ({best_score:.4f})")
        self.logger.info(f"RFECV vs Baseline: {rfecv_score:.4f} vs {baseline_score:.4f} (Δ{improvement:+.4f})")
        
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
        """สร้างรายงานการคัดเลือก Features (แก้ไขแล้ว)"""
        self.logger.info("Creating feature selection report...")
        
        # หาวิธีที่ดีที่สุด
        best_method_info = comparison_results.get('best_method', {})
        best_method_name = best_method_info.get('name', 'rfecv')
        
        # ข้อมูลของ best method
        best_method_data = comparison_results.get(best_method_name, {})
        
        # สร้างรายงาน
        report = {
            'feature_selection_config': {
                'method': 'rfecv_vs_baseline',
                'rfecv_enabled': True,
                'baseline_comparison': True,
                'min_features_tested': MIN_FEATURES,
                'max_features_available': MAX_FEATURES,
                'scoring_metric': FEATURE_SELECTION_SCORING,
                'cv_folds': CV_FOLDS,
                'estimator': 'RandomForestClassifier'
            },
            'comparison_results': comparison_results,
            'best_method': {
                'name': best_method_name,
                'accuracy': best_method_data.get('mean_score', 0),
                'std': best_method_data.get('std_score', 0),
                'n_features': best_method_data.get('n_features_selected', 0),
                'improvement': best_method_data.get('improvement', 0)
            },
            'final_selection': {
                'method_used': best_method_name,
                'selected_features': self.selected_features,
                'n_features_selected': len(self.selected_features)
            },
            'feature_analysis': {
                'core_features_selected': len([f for f in self.selected_features if f in CORE_FEATURES]),
                'demographic_features_selected': len([f for f in self.selected_features if f in DEMOGRAPHIC_FEATURES]),
                'total_features_available': len(CORE_FEATURES) + len(DEMOGRAPHIC_FEATURES)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # เพิ่มข้อมูลเฉพาะสำหรับ RFECV
        if 'rfecv' in comparison_results:
            rfecv_data = comparison_results['rfecv']
            report['rfecv_details'] = {
                'optimal_n_features': rfecv_data.get('optimal_n_features', 0),
                'n_features_tested': rfecv_data.get('n_features_tested', 0),
                'cv_scores': rfecv_data.get('cv_scores', []),
                'cv_std': rfecv_data.get('cv_std', []),
                'detailed_history': rfecv_data.get('detailed_history', []),
                'feature_ranking': rfecv_data.get('feature_ranking', []),
                'improvement_over_baseline': rfecv_data.get('improvement', 0)
            }
        
        return report
    
    def run_feature_selection(self) -> Tuple[pd.DataFrame, Dict]:
        """รันขั้นตอนการคัดเลือก Features ทั้งหมด (แก้ไขแล้ว)"""
        self.logger.info("Starting RFECV feature selection pipeline...")
        
        try:
            tracker = ProgressTracker(5, "Feature Selection (RFECV)")
            
            # 1. โหลดข้อมูล
            X_train, X_test, y_train, y_test = self.load_normalized_data()
            tracker.update("Loading normalized data")
            
            # 2. เปรียบเทียบ RFECV กับ Baseline
            comparison_results = self.compare_feature_selection_methods(X_train, y_train)
            tracker.update("Comparing RFECV vs Baseline")
            
            # 3. สร้าง dataset ที่ใช้ features ที่เลือก
            train_selected, test_selected = self.create_selected_dataset(X_train, X_test, y_train, y_test)
            tracker.update("Creating selected dataset")
            
            # 4. สร้างรายงาน
            report = self.create_feature_selection_report(comparison_results)
            tracker.update("Creating feature selection report")
            
            # 5. บันทึกผลลัพธ์
            save_json(report, get_output_path('feature_selection', 'feature_selection_report.json'))
            save_json({'selected_features': self.selected_features}, 
                     get_output_path('feature_selection', 'selected_features.json'))
            
            # บันทึก feature importance (สำหรับ visualization)
            if 'rfecv' in comparison_results:
                rfecv_data = comparison_results['rfecv']
                feature_importance = rfecv_data.get('feature_importance', {})
                
                if feature_importance:
                    feature_scores_df = pd.DataFrame([
                        {'feature': feat, 'importance': score}
                        for feat, score in feature_importance.items()
                    ]).sort_values('importance', ascending=False)
                    
                    save_data(feature_scores_df, get_output_path('feature_selection', 'selection_scores.csv'))
            
            tracker.update("Saving results")
            tracker.finish()
            
            # แสดงผลสรุป
            if VERBOSE:
                best_method = report['best_method']
                feature_analysis = report['feature_analysis']
                rfecv_details = report.get('rfecv_details', {})
                
                comparison_summary = {
                    'Method': 'RFECV vs Baseline',
                    'Winner': best_method['name'],
                    'Accuracy': f"{best_method['accuracy']:.4f}",
                    'Improvement': f"{best_method['improvement']:.4f}",
                    'Features Selected': f"{best_method['n_features']}/{feature_analysis['total_features_available']}",
                    'RFECV Optimal': rfecv_details.get('optimal_n_features', 0),
                    'Features Tested': f"{MIN_FEATURES}-{feature_analysis['total_features_available']}",
                    'Core Features': f"{feature_analysis['core_features_selected']}/{len(CORE_FEATURES)}",
                    'Demographic Features': f"{feature_analysis['demographic_features_selected']}/{len(DEMOGRAPHIC_FEATURES)}",
                    'Selected Features': self.selected_features
                }
                
                print_summary("Feature Selection Results (RFECV)", comparison_summary)
            
            self.logger.info("Feature selection completed successfully (RFECV)")
            
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
        print(f"🎯 Best method: {report['best_method']['name']}")
        print(f"📁 Results saved to: {FEATURE_SELECTION_RESULT_DIR}")
        
        return data_selected, report
        
    except Exception as e:
        logger.error(f"Feature selection failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()