# ไฟล์: feature_selection.py
# Path: src/feature_selection.py
# วัตถุประสงค์: Step 2 - Feature Selection ด้วยวิธี Forward/Backward Selection (แก้ไขแล้ว - เพิ่ม Domain Knowledge Filtering)

"""
feature_selection.py - ขั้นตอนการคัดเลือก Features ที่สำคัญ (ปรับปรุงแล้ว)
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
    """คลาสสำหรับการคัดเลือก Features (ปรับปรุงแล้ว)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.selector = None
        self.selected_features = []
        self.feature_scores = {}
        self.selection_results = {}
        self.domain_filter_results = {}
        
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
    
    def domain_knowledge_filtering(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        กรอง features ตาม domain knowledge ก่อนทำ feature selection
        เพื่อป้องกัน lifestyle factors เข้ามา
        """
        self.logger.info("Applying domain knowledge filtering...")
        
        # กำหนด priority levels ตามหลักการทางการศึกษา
        CORE_PRIORITY = [  # ความสำคัญสูงสุด - ต้องมี
            'คะแนนคณิตศาสตร์', 'คะแนนคอมพิวเตอร์', 'คะแนนภาษาไทย', 
            'คะแนนวิทยาศาสตร์', 'คะแนนศิลปะ',
            'ทักษะการคิดเชิงตรรกะ', 'ทักษะความคิดสร้างสรรค์', 'ทักษะการแก้ปัญหา',
            'ความสนใจด้านตัวเลข', 'ความสนใจด้านเทคโนโลยี', 'ความสนใจด้านการทำอาหาร'
        ]
        
        DEMOGRAPHIC_ALLOWED = [  # อนุญาตบ้าง แต่จำกัด
            'อายุ', 'เพศ'  # เฉพาะ demographic ที่เกี่ยวข้องกับการเรียนจริงๆ
        ]
        
        LIFESTYLE_BLOCKED = [  # ห้ามใช้เด็ดขาด
            'ชั่วโมงการนอน', 'ความถี่การออกกำลังกาย', 'ชั่วโมงใช้โซเชียลมีเดีย',
            'ชอบอ่านหนังสือ', 'ประเภทเพลงที่ชอบ',
            'รายได้ครอบครัว', 'การศึกษาของผู้ปกครอง', 'จำนวนพี่น้อง'  
        ]
        
        # สร้าง feature pool สำหรับ selection
        available_features = []
        
        # 1. เอา CORE ทั้งหมดที่มี
        core_available = [f for f in CORE_PRIORITY if f in X_train.columns]
        available_features.extend(core_available)
        
        # 2. เพิ่ม demographic ที่อนุญาต (สูงสุด 2 ตัว)
        demo_available = [f for f in DEMOGRAPHIC_ALLOWED if f in X_train.columns]
        available_features.extend(demo_available[:2])  # จำกัดแค่ 2 ตัว
        
        # กรอง lifestyle ออก
        blocked_found = [f for f in LIFESTYLE_BLOCKED if f in X_train.columns]
        
        domain_filter = {
            'filtered_features': available_features,
            'core_features': core_available,
            'demographic_features': demo_available[:2],
            'blocked_features': blocked_found,
            'original_count': len(X_train.columns),
            'filtered_count': len(available_features),
            'blocked_count': len(blocked_found),
            'justification': {
                'core_rationale': "คะแนน + ทักษะ + ความสนใจ เป็นปัจจัยหลักในการเลือกแผนก",
                'demographic_rationale': "อายุ เพศ อาจมีผลต่อการเรียนในระดับต่ำ",
                'lifestyle_blocked_rationale': "การนอน การออกกำลังกาย รายได้ครอบครัว เป็นเรื่องส่วนตัวที่ไม่ควรเป็นปัจจัยการเลือกแผนก"
            }
        }
        
        self.logger.info(f"Domain filtering: {len(X_train.columns)} -> {len(available_features)} features")
        self.logger.info(f"Core features: {len(core_available)}")
        self.logger.info(f"Blocked lifestyle features: {blocked_found}")
        
        return domain_filter
    
    def enhanced_sequential_selection(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Sequential Feature Selection ที่ปรับปรุงแล้วด้วย Domain Knowledge Filtering
        """
        self.logger.info("Starting enhanced sequential feature selection...")
        
        # 1. Domain knowledge filtering ก่อน
        domain_filter = self.domain_knowledge_filtering(X_train, y_train)
        filtered_X = X_train[domain_filter['filtered_features']]
        
        # 2. ทำ Sequential Selection บน filtered features
        estimator = RandomForestClassifier(
            n_estimators=100,  # เพิ่มขึ้นเพื่อความแม่นยำ
            max_depth=8,       # ลดลงเพื่อป้องกัน overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        # ปรับ n_features ให้เหมาะสม (ไม่เกิน 80% ของ filtered features)
        max_features = min(N_FEATURES_TO_SELECT, int(len(filtered_X.columns) * 0.8))
        max_features = max(max_features, 8)  # อย่างน้อย 8 features
        
        self.selector = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=max_features,
            direction='forward',  # ใช้ forward เพื่อเลือกสำคัญก่อน
            scoring=FEATURE_SELECTION_SCORING,
            cv=CV_FOLDS,
            n_jobs=CV_N_JOBS
        )
        
        # Fit และเลือก features
        self.logger.info(f"Selecting {max_features} features from {len(filtered_X.columns)} candidates...")
        X_selected = self.selector.fit_transform(filtered_X, y_train)
        selected_mask = self.selector.get_support()
        selected_features = filtered_X.columns[selected_mask].tolist()
        
        # 3. Validate การเลือก
        validation_results = self.validate_feature_selection(selected_features, domain_filter)
        
        # 4. คำนวณ CV scores
        cv_scores = cross_val_score(estimator, X_selected, y_train, 
                                  cv=CV_FOLDS, scoring=FEATURE_SELECTION_SCORING)
        
        results = {
            'method': 'Enhanced_Sequential_Forward',
            'domain_filtering': domain_filter,
            'n_features_selected': len(selected_features),
            'selected_features': selected_features,
            'validation': validation_results,
            'cv_scores': cv_scores.tolist(),
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'feature_mask': selected_mask.tolist()
        }
        
        self.logger.info(f"Enhanced sequential selection completed")
        self.logger.info(f"Selected {len(selected_features)} features with validation score: {validation_results['quality_score']:.1f}/200")
        
        return results
    
    def validate_feature_selection(self, selected_features: List[str], domain_filter: Dict) -> Dict:
        """
        ตรวจสอบว่า feature selection สมเหตุสมผลตาม domain knowledge
        """
        core_selected = [f for f in selected_features if f in domain_filter['core_features']]
        demo_selected = [f for f in selected_features if f in domain_filter['demographic_features']]
        blocked_found = [f for f in selected_features if f in domain_filter['blocked_features']]
        
        # คำนวณ percentages
        total_features = len(selected_features)
        core_percentage = len(core_selected) / total_features * 100 if total_features > 0 else 0
        demo_percentage = len(demo_selected) / total_features * 100 if total_features > 0 else 0
        blocked_percentage = len(blocked_found) / total_features * 100 if total_features > 0 else 0
        
        # Validation criteria ตามหลักการทางการศึกษา
        is_valid = (
            core_percentage >= 70 and  # อย่างน้อย 70% ต้องเป็น core features
            demo_percentage <= 20 and  # ไม่เกิน 20% demographic
            blocked_percentage == 0    # ห้าม blocked features เด็ดขาด
        )
        
        # คะแนนคุณภาพ (0-200)
        quality_score = (
            (core_percentage / 70) * 100 +           # สูงสุด 100 คะแนนจาก core
            max(0, (20 - demo_percentage) / 20) * 50 + # สูงสุด 50 คะแนนจาก demo constraint
            (50 if blocked_percentage == 0 else 0)     # 50 คะแนนถ้าไม่มี blocked
        )
        
        validation = {
            'is_valid': is_valid,
            'core_percentage': core_percentage,
            'demographic_percentage': demo_percentage,
            'blocked_percentage': blocked_percentage,
            'core_features_selected': core_selected,
            'demographic_features_selected': demo_selected,
            'blocked_features_found': blocked_found,
            'quality_score': min(quality_score, 200),  # Cap ที่ 200
            'recommendation': 'PASS' if is_valid else 'REVIEW_NEEDED',
            'academic_defensible': quality_score >= 150  # ผ่านเกณฑ์นักวิชาการ
        }
        
        return validation
    
    def sequential_feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """ทำ Sequential Feature Selection (เวอร์ชันเดิม - เก็บไว้เป็น backup)"""
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
        """เปรียบเทียบวิธี Feature Selection ต่างๆ (ใช้ enhanced method เป็นหลัก)"""
        self.logger.info("Comparing feature selection methods...")
        
        comparison_results = {}
        estimator = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
        
        # 1. Enhanced Sequential (หลักใหม่)
        enhanced_results = self.enhanced_sequential_selection(X_train, y_train)
        comparison_results['enhanced_sequential'] = enhanced_results
        
        # 2. Sequential เดิม (เก็บไว้เปรียบเทียบ)
        if FEATURE_SELECTION_METHOD in ['forward', 'backward']:
            seq_results = self.sequential_feature_selection(X_train, y_train)
            comparison_results['sequential_original'] = seq_results
        
        # 3. RFE-CV
        rfe_results = self.rfe_feature_selection(X_train, y_train)
        comparison_results['rfe_cv'] = rfe_results
        
        # 4. Filter methods (บน enhanced features เท่านั้น)
        domain_filter = enhanced_results['domain_filtering']
        filtered_X = X_train[domain_filter['filtered_features']]
        filter_results = self.filter_feature_selection(filtered_X, y_train)
        comparison_results.update(filter_results)
        
        # 5. Embedded methods (บน enhanced features เท่านั้น)
        embedded_results = self.embedded_feature_selection(filtered_X, y_train)
        comparison_results.update(embedded_results)
        
        # 6. All features (baseline) - กรองแล้ว
        baseline_scores = cross_val_score(estimator, filtered_X, y_train, 
                                        cv=CV_FOLDS, scoring=FEATURE_SELECTION_SCORING)
        comparison_results['filtered_baseline'] = {
            'method': 'Filtered_Baseline',
            'n_features_selected': len(filtered_X.columns),
            'selected_features': filtered_X.columns.tolist(),
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
    
    def create_enhanced_report(self, main_results: Dict) -> Dict:
        """สร้างรายงานการคัดเลือก Features ที่ปรับปรุงแล้ว"""
        self.logger.info("Creating enhanced feature selection report...")
        
        # สรุปผลการเปรียบเทียบ (ถ้ามี)
        method_summary = {}
        
        validation = main_results['validation']
        domain_filter = main_results['domain_filtering']
        
        # สร้างรายงาน
        report = {
            'feature_selection_config': {
                'method': 'enhanced_sequential_forward',
                'domain_knowledge_applied': True,
                'n_features_to_select': main_results['n_features_selected'],
                'scoring_metric': FEATURE_SELECTION_SCORING,
                'cv_folds': CV_FOLDS
            },
            'domain_knowledge_filtering': domain_filter,
            'validation_results': validation,
            'main_results': main_results,
            'method_summary': {
                'enhanced_sequential': {
                    'accuracy': main_results['mean_score'],
                    'std': main_results['std_score'],
                    'n_features': main_results['n_features_selected'],
                    'quality_score': validation['quality_score'],
                    'academic_defensible': validation['academic_defensible']
                }
            },
            'best_method': {
                'name': 'enhanced_sequential',
                'accuracy': main_results['mean_score'],
                'std': main_results['std_score'],
                'n_features': main_results['n_features_selected'],
                'quality_score': validation['quality_score']
            },
            'final_selection': {
                'method_used': 'enhanced_sequential_forward',
                'selected_features': self.selected_features,
                'n_features_selected': len(self.selected_features),
                'validation_status': validation['recommendation']
            },
            'academic_justification': {
                'core_features_percentage': validation['core_percentage'],
                'blocked_features_eliminated': len(domain_filter['blocked_features']),
                'educational_principles_applied': [
                    "Academic Achievement Theory",
                    "Cognitive Skills Framework", 
                    "Vocational Interest Theory",
                    "Educational Equity Principle"
                ],
                'defensibility_score': validation['quality_score']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def run_feature_selection(self) -> Tuple[pd.DataFrame, Dict]:
        """รันขั้นตอนการคัดเลือก Features ทั้งหมด (ปรับปรุงแล้ว)"""
        self.logger.info("Starting ENHANCED feature selection pipeline...")
        
        try:
            tracker = ProgressTracker(7, "Enhanced Feature Selection")
            
            # 1. โหลดข้อมูล
            X_train, X_test, y_train, y_test = self.load_normalized_data()
            tracker.update("Loading normalized data")
            
            # 2. รัน Enhanced Sequential Selection (หลัก)
            main_results = self.enhanced_sequential_selection(X_train, y_train)
            self.selected_features = main_results['selected_features']
            tracker.update("Enhanced sequential selection")
            
            # 3. ตรวจสอบ validation
            validation = main_results['validation']
            if not validation['is_valid']:
                self.logger.warning("🚨 Feature selection validation FAILED!")
                self.logger.warning(f"Core%: {validation['core_percentage']:.1f}%")
                self.logger.warning(f"Demo%: {validation['demographic_percentage']:.1f}%")
                self.logger.warning(f"Blocked found: {validation['blocked_features_found']}")
                self.logger.warning(f"Quality score: {validation['quality_score']:.1f}/200")
            else:
                self.logger.info("✅ Feature selection validation PASSED!")
                self.logger.info(f"Quality score: {validation['quality_score']:.1f}/200")
            tracker.update("Validating selection")
            
            # 4. เปรียบเทียบกับวิธีอื่นๆ (optional)
            comparison_results = self.compare_feature_selection_methods(X_train, y_train)
            tracker.update("Comparing methods")
            
            # 5. สร้าง dataset ที่ใช้ features ที่เลือก
            train_selected, test_selected = self.create_selected_dataset(X_train, X_test, y_train, y_test)
            tracker.update("Creating selected dataset")
            
            # 6. สร้างรายงาน
            report = self.create_enhanced_report(main_results)
            report['comparison_results'] = comparison_results  # เพิ่มผลการเปรียบเทียบ
            tracker.update("Creating enhanced report")
            
            # 7. บันทึกผลลัพธ์
            save_json(report, get_output_path('feature_selection', 'enhanced_feature_selection_report.json'))
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
                validation = main_results['validation']
                domain_filter = main_results['domain_filtering']
                print_summary("Enhanced Feature Selection Results", {
                    'Validation Status': f"{validation['recommendation']} ({'✅' if validation['is_valid'] else '❌'})",
                    'Quality Score': f"{validation['quality_score']:.1f}/200",
                    'Academic Defensible': f"{'✅' if validation['academic_defensible'] else '❌'}",
                    'Core Features': f"{len(validation['core_features_selected'])}/{main_results['n_features_selected']} ({validation['core_percentage']:.1f}%)",
                    'Blocked Features Eliminated': f"{len(domain_filter['blocked_features'])}",
                    'Accuracy': f"{main_results['mean_score']:.4f}",
                    'Selected Features': self.selected_features
                })
            
            self.logger.info("Enhanced feature selection completed successfully")
            
            return train_selected, report
            
        except Exception as e:
            self.logger.error(f"Enhanced feature selection failed: {str(e)}")
            raise

def main():
    """ฟังก์ชันหลักสำหรับการทดสอบ"""
    try:
        selector = FeatureSelector()
        data_selected, report = selector.run_feature_selection()
        
        print("✅ Enhanced feature selection completed successfully!")
        print(f"📊 Selected features: {len(selector.selected_features)}")
        print(f"🎯 Quality score: {report['validation_results']['quality_score']:.1f}/200")
        print(f"📁 Results saved to: {FEATURE_SELECTION_RESULT_DIR}")
        
        return data_selected, report
        
    except Exception as e:
        logger.error(f"Feature selection failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()