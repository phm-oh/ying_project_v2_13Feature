# ไฟล์: model_training.py
# Path: src/model_training.py
# วัตถุประสงค์: Step 3 - Model Training ด้วย 3 algorithms + 10-fold CV + Confusion Matrix (แก้แล้ว)

"""
model_training.py - ขั้นตอนการฝึกสอนโมเดลและประเมินผล
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Tuple, Any, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional XGBoost import เพื่อหลีกเลี่ยง error บน macOS
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available, skipping...")

# แก้ imports เป็น relative imports
from .config import *
from .utils import *

class ModelTrainer:
    """คลาสสำหรับการฝึกสอนและประเมินโมเดล"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.trained_models = {}
        self.evaluation_results = {}
        self.cv_results = {}
        self.predictions = {}
        
    def load_selected_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """โหลดข้อมูลที่ผ่าน feature selection แล้ว"""
        self.logger.info("Loading selected features data...")
        
        # โหลดข้อมูล train/test ที่ feature selection แล้ว
        try:
            train_data = load_data(get_output_path('feature_selection', 'train_data_selected.csv'))
            test_data = load_data(get_output_path('feature_selection', 'test_data_selected.csv'))
        except FileNotFoundError:
            self.logger.warning("Selected data not found, using normalized data...")
            train_data = load_data(get_output_path('preprocessing', 'train_data.csv'))
            test_data = load_data(get_output_path('preprocessing', 'test_data.csv'))
            
            # ลบ features ที่ไม่ต้องการ
            features_to_exclude = [f for f in EXCLUDE_FEATURES if f in train_data.columns]
            if features_to_exclude:
                train_data = train_data.drop(features_to_exclude, axis=1)
                test_data = test_data.drop(features_to_exclude, axis=1)
        
        # แยก features และ target
        X_train = train_data.drop(TARGET_COLUMN, axis=1)
        y_train = train_data[TARGET_COLUMN]
        X_test = test_data.drop(TARGET_COLUMN, axis=1)
        y_test = test_data[TARGET_COLUMN]
        
        self.logger.info(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
        self.logger.info(f"Features: {list(X_train.columns)}")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self) -> Dict:
        """สร้างและตั้งค่าโมเดลต่างๆ"""
        self.logger.info("Initializing models...")
        
        models = {}
        
        # Random Forest
        if 'random_forest' in SELECTED_MODELS:
            rf_params = get_model_config('random_forest')['params']
            models['Random_Forest'] = RandomForestClassifier(**rf_params)
        
        # Gradient Boosting
        if 'gradient_boosting' in SELECTED_MODELS:
            gb_params = get_model_config('gradient_boosting')['params']
            models['Gradient_Boosting'] = GradientBoostingClassifier(**gb_params)
        
        # Logistic Regression
        if 'logistic_regression' in SELECTED_MODELS:
            lr_params = get_model_config('logistic_regression')['params']
            models['Logistic_Regression'] = LogisticRegression(**lr_params)
        
        # SVM (ถ้ามีในการตั้งค่า)
        if 'svm' in SELECTED_MODELS:
            svm_params = get_model_config('svm')['params']
            models['SVM'] = SVC(**svm_params)
        
        # XGBoost (ถ้าติดตั้งแล้ว)
        if 'xgboost' in SELECTED_MODELS and XGBOOST_AVAILABLE:
            xgb_params = get_model_config('xgboost')['params']
            models['XGBoost'] = XGBClassifier(**xgb_params)
        elif 'xgboost' in SELECTED_MODELS and not XGBOOST_AVAILABLE:
            self.logger.warning("XGBoost requested but not available, skipping...")
        
        self.models = models
        self.logger.info(f"Initialized {len(models)} models: {list(models.keys())}")
        
        return models
    
    def cross_validation_evaluation(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """ทำ Cross-Validation ประเมินผล"""
        self.logger.info(f"Starting {CV_FOLDS}-fold cross-validation...")
        
        cv_results = {}
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        for model_name, model in self.models.items():
            self.logger.info(f"Cross-validating {model_name}...")
            
            # ทำ cross-validation
            scoring_metrics = CV_SCORING_METRICS
            cv_scores = cross_validate(
                model, X_train, y_train,
                cv=cv,
                scoring=scoring_metrics,
                n_jobs=CV_N_JOBS,
                return_train_score=True
            )
            
            # เก็บผลลัพธ์
            results = {}
            for metric in scoring_metrics:
                test_scores = cv_scores[f'test_{metric}']
                train_scores = cv_scores[f'train_{metric}']
                
                results[metric] = {
                    'test_scores': test_scores.tolist(),
                    'train_scores': train_scores.tolist(),
                    'test_mean': test_scores.mean(),
                    'test_std': test_scores.std(),
                    'train_mean': train_scores.mean(),
                    'train_std': train_scores.std()
                }
            
            # เพิ่มข้อมูลเพิ่มเติม
            results['fit_time'] = cv_scores['fit_time'].tolist()
            results['score_time'] = cv_scores['score_time'].tolist()
            results['mean_fit_time'] = cv_scores['fit_time'].mean()
            results['mean_score_time'] = cv_scores['score_time'].mean()
            
            cv_results[model_name] = results
            
            self.logger.info(f"{model_name} CV completed - Accuracy: {results['accuracy']['test_mean']:.4f} (+/- {results['accuracy']['test_std']*2:.4f})")
        
        self.cv_results = cv_results
        return cv_results
    
    def train_final_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """ฝึกสอนโมเดลสุดท้ายด้วยข้อมูล train ทั้งหมด"""
        self.logger.info("Training final models...")
        
        trained_models = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"Training {model_name}...")
            
            # ฝึกสอนโมเดล
            model.fit(X_train, y_train)
            trained_models[model_name] = model
            
            # บันทึกโมเดล
            model_path = get_model_save_path(model_name.lower())
            save_model(model, model_path)
            
            self.logger.info(f"{model_name} training completed")
        
        self.trained_models = trained_models
        return trained_models
    
    def evaluate_test_performance(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """ประเมินผลบน test set"""
        self.logger.info("Evaluating models on test set...")
        
        test_results = {}
        
        for model_name, model in self.trained_models.items():
            self.logger.info(f"Evaluating {model_name} on test set...")
            
            # ทำนายผลลัพธ์
            y_pred = model.predict(X_test)
            y_prob = None
            
            # ทำนายความน่าจะเป็น (ถ้าโมเดลรองรับ)
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
            elif hasattr(model, 'decision_function'):
                y_prob = model.decision_function(X_test)
            
            # คำนวณ metrics
            metrics = calculate_metrics(y_test, y_pred, y_prob)
            
            # เพิ่ม feature importance (ถ้ามี)
            if hasattr(model, 'feature_importances_'):
                feature_names = X_test.columns.tolist()
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                metrics['feature_importance'] = importance_dict
            
            # เก็บการทำนาย
            self.predictions[model_name] = {
                'y_pred': y_pred.tolist(),
                'y_prob': y_prob.tolist() if y_prob is not None else None,
                'y_true': y_test.tolist()
            }
            
            test_results[model_name] = metrics
            
            self.logger.info(f"{model_name} test evaluation completed - Accuracy: {metrics['accuracy']:.4f}")
        
        self.evaluation_results = test_results
        return test_results
    
    def create_confusion_matrices(self, y_test: pd.Series) -> Dict:
        """สร้าง Confusion Matrix สำหรับทุกโมเดล"""
        self.logger.info("Creating confusion matrices...")
        
        # โหลด target mapping
        try:
            target_mapping_path = get_output_path('preprocessing', 'target_mapping.json')
            target_mapping = load_json(target_mapping_path)
            # สร้าง reverse mapping
            labels = [k for k, v in sorted(target_mapping.items(), key=lambda x: x[1])]
        except:
            # ใช้ labels จาก data
            labels = sorted(y_test.unique())
        
        setup_plot_style()
        
        # สร้างกราฟสำหรับทุกโมเดล
        n_models = len(self.trained_models)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        confusion_matrices = {}
        
        for idx, (model_name, prediction_data) in enumerate(self.predictions.items()):
            y_pred = np.array(prediction_data['y_pred'])
            y_true = np.array(prediction_data['y_true'])
            
            # คำนวณ confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            confusion_matrices[model_name] = cm.tolist()
            
            # สร้างกราฟ
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels,
                       ax=axes[idx])
            axes[idx].set_title(f'{model_name}')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_ylabel('True Label')
        
        plt.tight_layout()
        
        # บันทึกกราฟ
        save_plot(fig, 'confusion_matrices.png', 'evaluation')
        
        self.logger.info("Confusion matrices created and saved")
        
        return confusion_matrices
    
    def statistical_significance_testing(self) -> Dict:
        """ทดสอบนัยสำคัญทางสถิติระหว่างโมเดล"""
        self.logger.info("Performing statistical significance testing...")
        
        if len(self.cv_results) < 2:
            self.logger.warning("Need at least 2 models for statistical testing")
            return {}
        
        statistical_tests = {}
        model_names = list(self.cv_results.keys())
        
        # เปรียบเทียบทุกคู่โมเดล
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                
                # ดึง accuracy scores จาก CV
                scores1 = self.cv_results[model1]['accuracy']['test_scores']
                scores2 = self.cv_results[model2]['accuracy']['test_scores']
                
                # ทดสอบทางสถิติ
                comparison_key = f"{model1}_vs_{model2}"
                statistical_tests[comparison_key] = {}
                
                # Paired t-test
                if STATISTICAL_TESTS['paired_ttest']:
                    t_stat, p_value = stats.ttest_rel(scores1, scores2)
                    statistical_tests[comparison_key]['paired_ttest'] = {
                        'statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < SIGNIFICANCE_LEVEL
                    }
                
                # Wilcoxon signed-rank test
                if STATISTICAL_TESTS['wilcoxon']:
                    try:
                        w_stat, p_value = stats.wilcoxon(scores1, scores2)
                        statistical_tests[comparison_key]['wilcoxon'] = {
                            'statistic': w_stat,
                            'p_value': p_value,
                            'significant': p_value < SIGNIFICANCE_LEVEL
                        }
                    except ValueError:
                        # ถ้าค่าเท่ากันหมด
                        statistical_tests[comparison_key]['wilcoxon'] = {
                            'error': 'All values are identical'
                        }
        
        # Friedman test (สำหรับหลายโมเดล)
        if len(model_names) >= 3 and STATISTICAL_TESTS['friedman']:
            all_scores = [self.cv_results[model]['accuracy']['test_scores'] 
                         for model in model_names]
            f_stat, p_value = stats.friedmanchisquare(*all_scores)
            
            statistical_tests['friedman_test'] = {
                'statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < SIGNIFICANCE_LEVEL,
                'models_tested': model_names
            }
        
        self.logger.info("Statistical significance testing completed")
        
        return statistical_tests
    
    def create_training_report(self, confusion_matrices: Dict, statistical_tests: Dict) -> Dict:
        """สร้างรายงานการฝึกสอนโมเดล"""
        self.logger.info("Creating training report...")
        
        # สรุปผลการประเมิน
        model_summary = {}
        for model_name in self.models.keys():
            cv_result = self.cv_results.get(model_name, {})
            test_result = self.evaluation_results.get(model_name, {})
            
            model_summary[model_name] = {
                'cross_validation': {
                    'accuracy_mean': cv_result.get('accuracy', {}).get('test_mean', 0),
                    'accuracy_std': cv_result.get('accuracy', {}).get('test_std', 0),
                    'precision_mean': cv_result.get('precision_macro', {}).get('test_mean', 0),
                    'recall_mean': cv_result.get('recall_macro', {}).get('test_mean', 0),
                    'f1_mean': cv_result.get('f1_macro', {}).get('test_mean', 0)
                },
                'test_performance': {
                    'accuracy': test_result.get('accuracy', 0),
                    'precision': test_result.get('precision', 0),
                    'recall': test_result.get('recall', 0),
                    'f1_score': test_result.get('f1_score', 0)
                },
                'training_time': cv_result.get('mean_fit_time', 0),
                'prediction_time': cv_result.get('mean_score_time', 0)
            }
        
        # หาโมเดลที่ดีที่สุด
        best_model_cv = max(model_summary.keys(), 
                           key=lambda x: model_summary[x]['cross_validation']['accuracy_mean'])
        best_model_test = max(model_summary.keys(), 
                             key=lambda x: model_summary[x]['test_performance']['accuracy'])
        
        # สร้างรายงาน
        report = {
            'training_config': {
                'models_used': list(self.models.keys()),
                'cv_folds': CV_FOLDS,
                'scoring_metrics': CV_SCORING_METRICS,
                'random_state': RANDOM_STATE
            },
            'cross_validation_results': self.cv_results,
            'test_evaluation_results': self.evaluation_results,
            'model_summary': model_summary,
            'confusion_matrices': confusion_matrices,
            'statistical_significance': statistical_tests,
            'best_models': {
                'best_cv_accuracy': best_model_cv,
                'best_test_accuracy': best_model_test,
                'cv_accuracy': model_summary[best_model_cv]['cross_validation']['accuracy_mean'],
                'test_accuracy': model_summary[best_model_test]['test_performance']['accuracy']
            },
            'predictions': self.predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def run_model_training(self) -> Tuple[Dict, Dict]:
        """รันขั้นตอนการฝึกสอนโมเดลทั้งหมด (เอา decorator ออก)"""
        self.logger.info("Starting model training pipeline...")
        
        try:
            tracker = ProgressTracker(8, "Model Training")
            
            # 1. โหลดข้อมูล
            X_train, X_test, y_train, y_test = self.load_selected_data()
            tracker.update("Loading selected data")
            
            # 2. สร้างโมเดล
            models = self.initialize_models()
            tracker.update("Initializing models")
            
            # 3. Cross-validation
            cv_results = self.cross_validation_evaluation(X_train, y_train)
            tracker.update("Cross-validation evaluation")
            
            # 4. ฝึกสอนโมเดลสุดท้าย
            trained_models = self.train_final_models(X_train, y_train)
            tracker.update("Training final models")
            
            # 5. ประเมินผลบน test set
            test_results = self.evaluate_test_performance(X_test, y_test)
            tracker.update("Test set evaluation")
            
            # 6. สร้าง confusion matrices
            confusion_matrices = self.create_confusion_matrices(y_test)
            tracker.update("Creating confusion matrices")
            
            # 7. ทดสอบนัยสำคัญทางสถิติ
            statistical_tests = self.statistical_significance_testing()
            tracker.update("Statistical significance testing")
            
            # 8. สร้างรายงานและบันทึกผลลัพธ์
            report = self.create_training_report(confusion_matrices, statistical_tests)
            
            # บันทึกไฟล์ผลลัพธ์
            save_json(report, get_output_path('evaluation', 'training_report.json'))
            save_json(cv_results, get_output_path('evaluation', 'cv_results.json'))
            save_json(test_results, get_output_path('evaluation', 'test_results.json'))
            save_json(self.predictions, get_output_path('evaluation', 'predictions.json'))
            
            # สร้าง CSV สำหรับ model performance
            performance_data = []
            for model_name, summary in report['model_summary'].items():
                row = {
                    'Model': model_name,
                    'CV_Accuracy_Mean': summary['cross_validation']['accuracy_mean'],
                    'CV_Accuracy_Std': summary['cross_validation']['accuracy_std'],
                    'CV_Precision_Mean': summary['cross_validation']['precision_mean'],
                    'CV_Recall_Mean': summary['cross_validation']['recall_mean'],
                    'CV_F1_Mean': summary['cross_validation']['f1_mean'],
                    'Test_Accuracy': summary['test_performance']['accuracy'],
                    'Test_Precision': summary['test_performance']['precision'],
                    'Test_Recall': summary['test_performance']['recall'],
                    'Test_F1': summary['test_performance']['f1_score'],
                    'Training_Time': summary['training_time'],
                    'Prediction_Time': summary['prediction_time']
                }
                performance_data.append(row)
            
            performance_df = pd.DataFrame(performance_data)
            save_data(performance_df, get_output_path('evaluation', 'model_performance.csv'))
            
            tracker.update("Saving results")
            tracker.finish()
            
            if VERBOSE:
                print_summary("Model Training Results", {
                    'Models Trained': len(trained_models),
                    'Best CV Model': report['best_models']['best_cv_accuracy'],
                    'Best Test Model': report['best_models']['best_test_accuracy'],
                    'Best CV Accuracy': f"{report['best_models']['cv_accuracy']:.4f}",
                    'Best Test Accuracy': f"{report['best_models']['test_accuracy']:.4f}",
                    'Cross-Validation': f"{CV_FOLDS}-fold",
                    'Statistical Tests': len(statistical_tests)
                })
            
            self.logger.info("Model training completed successfully")
            
            return trained_models, report
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise

def main():
    """ฟังก์ชันหลักสำหรับการทดสอบ"""
    try:
        trainer = ModelTrainer()
        models, report = trainer.run_model_training()
        
        print("✅ Model training completed successfully!")
        print(f"🤖 Models trained: {len(models)}")
        print(f"📊 Best model: {report['best_models']['best_test_accuracy']}")
        print(f"📁 Results saved to: {EVALUATION_RESULT_DIR}")
        
        return models, report
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()