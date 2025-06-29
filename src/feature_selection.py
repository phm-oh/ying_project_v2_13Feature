# ไฟล์: feature_selection.py
# Path: src/feature_selection.py
# วัตถุประสงค์: Step 2 - Feature Selection ด้วย Forward Selection (แก้แล้ว - เพิ่ม Smart Sequential)

"""
feature_selection.py - ขั้นตอนการคัดเลือก Features ที่สำคัญ (เพิ่ม Smart Sequential)
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
from typing import Dict, Tuple, Any, Optional, Union, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# แก้ imports เป็น relative imports
from .config import *
from .utils import *

class FeatureSelector:
    """คลาสสำหรับการคัดเลือก Features (เพิ่ม Smart Sequential)"""
    
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
    
    def smart_sequential_selection(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """ทำ Sequential Selection แบบ auto-stopping เมื่อ accuracy ไม่เพิ่ม"""
        self.logger.info("Starting Smart Sequential Feature Selection...")
        
        estimator = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
        )
        
        selected_features = []
        remaining_features = list(X_train.columns)
        best_score = 0
        patience = 3  # หยุดถ้า 3 รอบติดต่อกันไม่มีการปรับปรุง
        no_improvement_count = 0
        scores_history = []
        
        self.logger.info(f"Starting with {len(remaining_features)} features")
        
        while remaining_features and no_improvement_count < patience:
            best_feature_this_round = None
            best_score_this_round = best_score
            
            self.logger.info(f"Round {len(selected_features) + 1}: Testing {len(remaining_features)} features...")
            
            # ลองเพิ่มแต่ละ feature ที่เหลือ
            for feature in remaining_features:
                test_features = selected_features + [feature]
                X_temp = X_train[test_features]
                
                # ทำ CV
                cv_scores = cross_val_score(estimator, X_temp, y_train, 
                                          cv=CV_FOLDS, scoring=FEATURE_SELECTION_SCORING)
                score = cv_scores.mean()
                
                if score > best_score_this_round:
                    best_score_this_round = score
                    best_feature_this_round = feature
            
            # เช็คว่ามีการปรับปรุงหรือไม่
            improvement = best_score_this_round - best_score
            if best_feature_this_round and improvement > 0.001:  # threshold 0.1%
                selected_features.append(best_feature_this_round)
                remaining_features.remove(best_feature_this_round)
                best_score = best_score_this_round
                no_improvement_count = 0
                scores_history.append(best_score)
                self.logger.info(f"✅ Added '{best_feature_this_round}', Score: {best_score:.4f} (+{improvement:.4f})")
            else:
                no_improvement_count += 1
                self.logger.info(f"❌ No significant improvement, Patience: {patience - no_improvement_count}/3")
        
        # เหตุผลที่หยุด
        if no_improvement_count >= patience:
            stop_reason = "patience_reached"
            self.logger.info(f"🛑 Stopped: No improvement for {patience} consecutive rounds")
        else:
            stop_reason = "no_more_features"
            self.logger.info(f"🛑 Stopped: No more features to test")
        
        self.selected_features = selected_features
        
        return {
            'method': 'Smart_Sequential_Forward',
            'n_features_selected': len(selected_features),
            'selected_features': selected_features,
            'final_score': best_score,
            'scores_history': scores_history,
            'stop_reason': stop_reason,
            'total_rounds': len(scores_history),
            'cv_scores': scores_history,  # สำหรับ compatibility
            'mean_score': best_score,
            'std_score': 0.0  # จะคำนวณในรอบสุดท้าย
        }
    
    def sequential_feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """ทำ Sequential Feature Selection แบบธรรมดา"""
        self.logger.info(f"Starting Sequential Feature Selection ({FEATURE_SELECTION_METHOD})...")
        
        # เลือก estimator สำหรับ feature selection
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        # ตั้งค่า direction
        direction = 'forward' if FEATURE_SELECTION_METHOD == 'forward' else 'backward'
        
        # สร้าง selector
        # หมายเหตุ: SequentialFeatureSelector ยังไม่มี auto-stopping
        # แต่เราสามารถใช้ RFE-CV แทนได้ (มี auto feature selection)
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
        
        # คำนวณ baseline (ใช้ features ทั้งหมด)
        baseline_scores = cross_val_score(estimator, X_train, y_train,
                                        cv=CV_FOLDS, scoring=FEATURE_SELECTION_SCORING)
        
        results = {
            'method': f'Sequential_{direction}',
            'n_features_selected': len(self.selected_features),
            'selected_features': self.selected_features,
            'cv_scores': scores.tolist(),
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'baseline_score': baseline_scores.mean(),
            'improvement': scores.mean() - baseline_scores.mean(),
            'feature_mask': selected_mask.tolist()
        }
        
        self.logger.info(f"Sequential FS completed - Selected {len(self.selected_features)} features")
        self.logger.info(f"Cross-validation score: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        self.logger.info(f"Improvement over baseline: {results['improvement']:.4f}")
        
        return results
    
    def rfe_feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """ทำ Recursive Feature Elimination"""
        self.logger.info("Starting RFE Feature Selection...")
        
        estimator = RandomForestClassifier(
            n_estimators=100,
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
        
        # Random Forest Feature Importance
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
        estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        
        # เลือกวิธีตาม config
        if FEATURE_SELECTION_METHOD == 'forward_smart':
            # ใช้ smart sequential ที่หยุดเมื่อ accuracy ไม่เพิ่ม
            main_results = self.smart_sequential_selection(X_train, y_train)
        else:
            # ใช้ sequential ปกติ
            main_results = self.sequential_feature_selection(X_train, y_train)
        
        comparison_results['sequential'] = main_results
        
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
            'n_features_selected': len(X_train.columns),
            'selected_features': X_train.columns.tolist(),
            'cv_scores': baseline_scores.tolist(),
            'mean_score': baseline_scores.mean(),
            'std_score': baseline_scores.std()
        }
        
        # หาวิธีที่ดีที่สุด
        best_method = None
        best_score = 0
        
        for method, results in comparison_results.items():
            if 'mean_score' in results:
                if results['mean_score'] > best_score:
                    best_score = results['mean_score']
                    best_method = method
        
        comparison_results['best_method'] = {
            'name': best_method,
            'score': best_score
        }
        
        # ใช้ features จากวิธีที่ดีที่สุด
        if best_method and best_method in comparison_results:
            self.selected_features = comparison_results[best_method]['selected_features']
        
        self.logger.info(f"Feature selection comparison completed - Best: {best_method} ({best_score:.4f})")
        
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
        
        # หาวิธีที่ดีที่สุด
        best_method_info = comparison_results.get('best_method', {})
        best_method_name = best_method_info.get('name', 'sequential')
        
        # ข้อมูลของ best method
        best_method_data = comparison_results.get(best_method_name, {})
        
        # สร้างรายงาน
        report = {
            'feature_selection_config': {
                'method': FEATURE_SELECTION_METHOD,
                'n_features_to_select': N_FEATURES_TO_SELECT,
                'scoring_metric': FEATURE_SELECTION_SCORING,
                'cv_folds': CV_FOLDS,
                'smart_selection_enabled': FEATURE_SELECTION_METHOD == 'forward_smart'
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
        
        # เพิ่มข้อมูลเฉพาะสำหรับ smart selection
        if FEATURE_SELECTION_METHOD == 'forward_smart' and 'sequential' in comparison_results:
            smart_data = comparison_results['sequential']
            report['smart_selection_details'] = {
                'stop_reason': smart_data.get('stop_reason', 'unknown'),
                'total_rounds': smart_data.get('total_rounds', 0),
                'scores_history': smart_data.get('scores_history', []),
                'early_stopping_worked': smart_data.get('stop_reason') == 'patience_reached'
            }
        
        return report
    
    def run_feature_selection(self) -> Tuple[pd.DataFrame, Dict]:
        """รันขั้นตอนการคัดเลือก Features ทั้งหมด"""
        self.logger.info("Starting feature selection pipeline...")
        
        try:
            tracker = ProgressTracker(6, "Feature Selection")
            
            # 1. โหลดข้อมูล
            X_train, X_test, y_train, y_test = self.load_normalized_data()
            tracker.update("Loading normalized data")
            
            # 2. เปรียบเทียบวิธี Feature Selection
            comparison_results = self.compare_feature_selection_methods(X_train, y_train)
            tracker.update("Comparing feature selection methods")
            
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
            
            # บันทึก feature scores (จาก RF importance)
            if 'random_forest' in comparison_results:
                feature_scores_df = pd.DataFrame([
                    {'feature': feat, 'importance': score}
                    for feat, score in comparison_results['random_forest']['feature_importance'].items()
                ]).sort_values('importance', ascending=False)
                
                save_data(feature_scores_df, get_output_path('feature_selection', 'selection_scores.csv'))
            
            tracker.update("Saving results")
            tracker.finish()
            
            # แสดงผลสรุป
            if VERBOSE:
                best_method = report['best_method']
                feature_analysis = report['feature_analysis']
                method_details = ""
                
                # เพิ่มรายละเอียดสำหรับ smart selection
                if FEATURE_SELECTION_METHOD == 'forward_smart' and 'smart_selection_details' in report:
                    smart_details = report['smart_selection_details']
                    method_details = f" (Rounds: {smart_details['total_rounds']}, Stop: {smart_details['stop_reason']})"
                
                print_summary("Feature Selection Results", {
                    'Method': f"{FEATURE_SELECTION_METHOD}{method_details}",
                    'Best Method': best_method['name'],
                    'Accuracy': f"{best_method['accuracy']:.4f}",
                    'Improvement': f"{best_method['improvement']:.4f}",
                    'Features Selected': f"{best_method['n_features']}/{feature_analysis['total_features_available']}",
                    'Core Features': f"{feature_analysis['core_features_selected']}/{len(CORE_FEATURES)}",
                    'Demographic Features': f"{feature_analysis['demographic_features_selected']}/{len(DEMOGRAPHIC_FEATURES)}",
                    'Selected Features': self.selected_features
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
        print(f"🎯 Best method: {report['best_method']['name']}")
        print(f"📁 Results saved to: {FEATURE_SELECTION_RESULT_DIR}")
        
        return data_selected, report
        
    except Exception as e:
        logger.error(f"Feature selection failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()