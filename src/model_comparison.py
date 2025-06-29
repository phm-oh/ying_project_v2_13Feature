# ไฟล์: model_comparison.py
# Path: src/model_comparison.py
# วัตถุประสงค์: Step 4 - Model Comparison และสรุปผลลัพธ์สุดท้าย (แก้แล้ว - ลบ statistical analysis)

"""
model_comparison.py - การเปรียบเทียบโมเดลและสรุปผลลัพธ์สุดท้าย (เรียบง่าย)
"""

import pandas as pd
import numpy as np
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple, Any, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# แก้ imports เป็น relative imports
from .config import *
from .utils import *

class ModelComparator:
    """คลาสสำหรับการเปรียบเทียบโมเดลและสรุปผลลัพธ์ (เรียบง่าย)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.comparison_results = {}
        self.final_ranking = {}
        self.recommendations = {}
        
    def load_all_results(self) -> Dict:
        """โหลดผลลัพธ์ทั้งหมด"""
        self.logger.info("Loading all results for comparison...")
        
        results = {}
        
        try:
            # โหลดผลการ preprocessing
            results['preprocessing_report'] = load_json(get_output_path('preprocessing', 'normalization_report.json'))
            
            # โหลดผลการ feature selection
            results['feature_selection_report'] = load_json(get_output_path('feature_selection', 'feature_selection_report.json'))
            results['selected_features'] = load_json(get_output_path('feature_selection', 'selected_features.json'))
            
            # โหลดผลการ training
            results['training_report'] = load_json(get_output_path('evaluation', 'training_report.json'))
            results['cv_results'] = load_json(get_output_path('evaluation', 'cv_results.json'))
            results['test_results'] = load_json(get_output_path('evaluation', 'test_results.json'))
            results['performance_df'] = load_data(get_output_path('evaluation', 'model_performance.csv'))
            
            # โหลด feature scores (ถ้ามี)
            try:
                results['feature_scores'] = load_data(get_output_path('feature_selection', 'selection_scores.csv'))
            except:
                results['feature_scores'] = None
                
        except Exception as e:
            self.logger.error(f"Error loading results: {str(e)}")
            raise
        
        self.logger.info("All results loaded successfully")
        return results
    
    def analyze_feature_selection_impact(self, results: Dict) -> Dict:
        """วิเคราะห์ผลกระทบของ Feature Selection (เรียบง่าย)"""
        self.logger.info("Analyzing feature selection impact...")
        
        feature_report = results['feature_selection_report']
        comparison_results = feature_report.get('comparison_results', {})
        
        # เปรียบเทียบกับ baseline (all features)
        baseline = comparison_results.get('all_features', {})
        best_method = feature_report.get('best_method', {})
        final_selection = feature_report.get('final_selection', {})
        
        impact_analysis = {
            'baseline_performance': {
                'accuracy': baseline.get('mean_score', 0),
                'std': baseline.get('std_score', 0),
                'n_features': baseline.get('n_features_selected', 0)
            },
            'best_method_performance': {
                'method': best_method.get('name', 'Unknown'),
                'accuracy': best_method.get('accuracy', 0),
                'std': best_method.get('std', 0),
                'n_features': best_method.get('n_features', 0)
            },
            'final_selection_performance': {
                'method': final_selection.get('method_used', 'Unknown'),
                'n_features': final_selection.get('n_features_selected', 0),
                'selected_features': final_selection.get('selected_features', [])
            }
        }
        
        # คำนวณการปรับปรุง
        baseline_acc = impact_analysis['baseline_performance']['accuracy']
        best_acc = impact_analysis['best_method_performance']['accuracy']
        
        improvement = {
            'accuracy_improvement': best_acc - baseline_acc,
            'accuracy_improvement_percent': ((best_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0,
            'feature_reduction': baseline.get('n_features_selected', 0) - best_method.get('n_features', 0),
            'feature_reduction_percent': ((baseline.get('n_features_selected', 0) - best_method.get('n_features', 0)) / baseline.get('n_features_selected', 1) * 100)
        }
        
        impact_analysis['improvement'] = improvement
        
        # วิเคราะห์ feature groups
        selected_features = final_selection.get('selected_features', [])
        feature_group_analysis = {
            'core_features_selected': len([f for f in selected_features if f in CORE_FEATURES]),
            'demographic_features_selected': len([f for f in selected_features if f in DEMOGRAPHIC_FEATURES]),
            'core_features_percentage': len([f for f in selected_features if f in CORE_FEATURES]) / len(CORE_FEATURES) * 100,
            'demographic_features_percentage': len([f for f in selected_features if f in DEMOGRAPHIC_FEATURES]) / len(DEMOGRAPHIC_FEATURES) * 100
        }
        
        impact_analysis['feature_group_analysis'] = feature_group_analysis
        
        self.logger.info("Feature selection impact analysis completed")
        return impact_analysis
    
    def comprehensive_model_comparison(self, results: Dict) -> Dict:
        """เปรียบเทียบโมเดลอย่างละเอียด (เรียบง่าย)"""
        self.logger.info("Performing comprehensive model comparison...")
        
        performance_df = results['performance_df']
        cv_results = results['cv_results']
        test_results = results['test_results']
        training_report = results['training_report']
        
        models = performance_df['Model'].tolist()
        comparison = {}
        
        for model in models:
            model_data = performance_df[performance_df['Model'] == model].iloc[0]
            cv_data = cv_results.get(model, {})
            test_data = test_results.get(model, {})
            
            # ประสิทธิภาพ
            performance_metrics = {
                'cv_accuracy_mean': model_data['CV_Accuracy_Mean'],
                'cv_accuracy_std': model_data['CV_Accuracy_Std'],
                'cv_precision_mean': model_data['CV_Precision_Mean'],
                'cv_recall_mean': model_data['CV_Recall_Mean'],
                'cv_f1_mean': model_data['CV_F1_Mean'],
                'test_accuracy': model_data['Test_Accuracy'],
                'test_precision': model_data['Test_Precision'],
                'test_recall': model_data['Test_Recall'],
                'test_f1': model_data['Test_F1']
            }
            
            # ประสิทธิภาพเวลา
            time_metrics = {
                'training_time': model_data['Training_Time'],
                'prediction_time': model_data['Prediction_Time'],
                'total_time': model_data['Training_Time'] + model_data['Prediction_Time']
            }
            
            # ความเสถียร (ง่ายๆ)
            stability_metrics = {
                'cv_accuracy_cv': cv_data.get('accuracy', {}).get('test_std', 0) / cv_data.get('accuracy', {}).get('test_mean', 1),  # Coefficient of variation
                'overfitting_indicator': cv_data.get('accuracy', {}).get('train_mean', 0) - cv_data.get('accuracy', {}).get('test_mean', 0)
            }
            
            # คะแนนรวม (weighted score)
            weights = {
                'accuracy': 0.5,     # เพิ่มน้ำหนัก accuracy
                'stability': 0.2,    # ลดน้ำหนัก stability  
                'speed': 0.15,       # ลดน้ำหนัก speed
                'precision': 0.15    # ลดน้ำหนัก precision
            }
            
            # Normalize scores to 0-1
            accuracy_score = performance_metrics['test_accuracy']
            stability_score = 1 - min(stability_metrics['cv_accuracy_cv'], 1)  # Lower CV is better
            speed_score = 1 / (1 + time_metrics['total_time'])  # Faster is better
            precision_score = performance_metrics['test_precision']
            
            overall_score = (weights['accuracy'] * accuracy_score + 
                           weights['stability'] * stability_score + 
                           weights['speed'] * speed_score + 
                           weights['precision'] * precision_score)
            
            comparison[model] = {
                'performance_metrics': performance_metrics,
                'time_metrics': time_metrics,
                'stability_metrics': stability_metrics,
                'normalized_scores': {
                    'accuracy_score': accuracy_score,
                    'stability_score': stability_score,
                    'speed_score': speed_score,
                    'precision_score': precision_score
                },
                'overall_score': overall_score,
                'rank_by_accuracy': 0,  # จะคำนวณทีหลัง
                'rank_by_speed': 0,
                'rank_by_stability': 0,
                'rank_overall': 0
            }
        
        # คำนวณ ranking
        models_sorted_by_accuracy = sorted(models, key=lambda x: comparison[x]['performance_metrics']['test_accuracy'], reverse=True)
        models_sorted_by_speed = sorted(models, key=lambda x: comparison[x]['time_metrics']['total_time'])
        models_sorted_by_stability = sorted(models, key=lambda x: comparison[x]['stability_metrics']['cv_accuracy_cv'])
        models_sorted_overall = sorted(models, key=lambda x: comparison[x]['overall_score'], reverse=True)
        
        for i, model in enumerate(models_sorted_by_accuracy):
            comparison[model]['rank_by_accuracy'] = i + 1
        for i, model in enumerate(models_sorted_by_speed):
            comparison[model]['rank_by_speed'] = i + 1
        for i, model in enumerate(models_sorted_by_stability):
            comparison[model]['rank_by_stability'] = i + 1
        for i, model in enumerate(models_sorted_overall):
            comparison[model]['rank_overall'] = i + 1
        
        # สรุปการเปรียบเทียบ
        comparison_summary = {
            'best_accuracy': models_sorted_by_accuracy[0],
            'fastest_model': models_sorted_by_speed[0],
            'most_stable': models_sorted_by_stability[0],
            'best_overall': models_sorted_overall[0],
            'model_rankings': {
                'by_accuracy': models_sorted_by_accuracy,
                'by_speed': models_sorted_by_speed,
                'by_stability': models_sorted_by_stability,
                'overall': models_sorted_overall
            }
        }
        
        comparison['summary'] = comparison_summary
        
        self.logger.info("Comprehensive model comparison completed")
        return comparison
    
    def generate_recommendations(self, comparison: Dict, feature_impact: Dict) -> Dict:
        """สร้างคำแนะนำและข้อสรุป (เรียบง่าย)"""
        self.logger.info("Generating recommendations...")
        
        best_model = comparison['summary']['best_overall']
        best_accuracy_model = comparison['summary']['best_accuracy']
        fastest_model = comparison['summary']['fastest_model']
        
        recommendations = {
            'model_selection': {
                'recommended_model': best_model,
                'reason': f"ได้คะแนนรวมสูงสุด ({comparison[best_model]['overall_score']:.4f})",
                'alternatives': {
                    'for_highest_accuracy': best_accuracy_model,
                    'for_fastest_prediction': fastest_model
                }
            },
            'feature_selection': {
                'effectiveness': feature_impact['improvement']['accuracy_improvement_percent'],
                'feature_reduction': feature_impact['improvement']['feature_reduction_percent'],
                'recommended_features': feature_impact['final_selection_performance']['selected_features']
            },
            'deployment_considerations': {
                'production_model': best_model,
                'accuracy_expectation': comparison[best_model]['performance_metrics']['test_accuracy'],
                'prediction_time': comparison[best_model]['time_metrics']['prediction_time'],
                'memory_considerations': "ไม่มีข้อจำกัดพิเศษ",
                'retraining_frequency': "ทุก 6 เดือน หรือเมื่อข้อมูลเปลี่ยนแปลงมาก"
            },
            'key_findings': {
                'feature_selection_impact': f"Feature selection ช่วยปรับปรุง accuracy {feature_impact['improvement']['accuracy_improvement_percent']:.2f}% และลด features {feature_impact['improvement']['feature_reduction_percent']:.1f}%",
                'best_performing_model': f"{best_model} ให้ประสิทธิภาพดีที่สุดด้วย test accuracy {comparison[best_model]['performance_metrics']['test_accuracy']:.4f}",
                'feature_groups_importance': self._analyze_feature_groups(feature_impact['feature_group_analysis']),
                'model_comparison_summary': f"ทดสอบ {len(comparison)-1} โมเดล ด้วย {CV_FOLDS}-fold cross-validation"
            },
            'next_steps': {
                'immediate': [
                    "ทดสอบโมเดลกับข้อมูลจริง",
                    "สร้าง validation set แยกต่างหาก",
                    "ประเมินประสิทธิภาพในสภาพแวดล้อมจริง"
                ],
                'long_term': [
                    "เก็บข้อมูลเพิ่มเติมเพื่อปรับปรุงโมเดล",
                    "พัฒนา monitoring system สำหรับติดตามประสิทธิภาพ",
                    "ศึกษา interpretability ของโมเดล"
                ]
            }
        }
        
        self.logger.info("Recommendations generated")
        return recommendations
    
    def _analyze_feature_groups(self, feature_group_analysis: Dict) -> str:
        """วิเคราะห์ความสำคัญของกลุ่ม features"""
        core_pct = feature_group_analysis['core_features_percentage']
        demo_pct = feature_group_analysis['demographic_features_percentage']
        
        if core_pct > 80:
            return f"Core features มีความสำคัญสูงสุด ({core_pct:.1f}% ถูกเลือก)"
        elif demo_pct > 60:
            return f"Demographic features มีความสำคัญสูง ({demo_pct:.1f}% ถูกเลือก)"
        else:
            return f"Features จากทั้งสองกลุ่มมีความสำคัญ (Core: {core_pct:.1f}%, Demographic: {demo_pct:.1f}%)"
    
    def create_final_report(self, results: Dict, feature_impact: Dict, 
                          comparison: Dict, recommendations: Dict) -> Dict:
        """สร้างรายงานสุดท้าย (เรียบง่าย)"""
        self.logger.info("Creating final comprehensive report...")
        
        # สรุปข้อมูลหลัก
        dataset_summary = {
            'original_features': len(CORE_FEATURES) + len(DEMOGRAPHIC_FEATURES),
            'selected_features': feature_impact['final_selection_performance']['n_features'],
            'total_samples': 1789,  # จาก dataset
            'target_classes': 3,  # บัญชี, สารสนเทศ, อาหาร
            'preprocessing_method': NORMALIZATION_METHOD,
            'feature_selection_method': FEATURE_SELECTION_METHOD,
            'cv_folds': CV_FOLDS
        }
        
        # สรุปผลลัพธ์หลัก
        main_results = {
            'best_model': comparison['summary']['best_overall'],
            'best_accuracy': max([comparison[model]['performance_metrics']['test_accuracy'] 
                                for model in comparison.keys() if model != 'summary']),
            'feature_selection_improvement': feature_impact['improvement']['accuracy_improvement_percent'],
            'feature_reduction': feature_impact['improvement']['feature_reduction_percent']
        }
        
        # Performance summary table
        performance_summary = {}
        for model in comparison.keys():
            if model != 'summary':
                performance_summary[model] = {
                    'test_accuracy': comparison[model]['performance_metrics']['test_accuracy'],
                    'cv_accuracy': comparison[model]['performance_metrics']['cv_accuracy_mean'],
                    'precision': comparison[model]['performance_metrics']['test_precision'],
                    'recall': comparison[model]['performance_metrics']['test_recall'],
                    'f1_score': comparison[model]['performance_metrics']['test_f1'],
                    'training_time': comparison[model]['time_metrics']['training_time'],
                    'overall_rank': comparison[model]['rank_overall']
                }
        
        final_report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'pipeline_version': EXPERIMENT_VERSION,
                'configuration': {
                    'normalization': NORMALIZATION_METHOD,
                    'feature_selection': FEATURE_SELECTION_METHOD,
                    'models_compared': len(comparison) - 1,  # -1 for 'summary' key
                    'cv_folds': CV_FOLDS,
                    'random_state': RANDOM_STATE,
                    'statistical_analysis': False  # แสดงว่าไม่ได้ใช้
                }
            },
            'dataset_summary': dataset_summary,
            'main_results': main_results,
            'feature_selection_analysis': feature_impact,
            'model_comparison': comparison,
            'performance_summary': performance_summary,
            'recommendations': recommendations,
            'conclusions': {
                'primary_conclusion': f"{comparison['summary']['best_overall']} เป็นโมเดลที่ดีที่สุดสำหรับการแนะนำแผนกเรียน",
                'feature_selection_conclusion': f"Feature selection ช่วยปรับปรุงประสิทธิภาพ {feature_impact['improvement']['accuracy_improvement_percent']:.2f}%",
                'practical_implications': "ระบบสามารถนำไปใช้เป็นเครื่องมือช่วยในการแนะแนวได้ แต่ควรใช้ร่วมกับการปรึกษาครูแนะแนว",
                'limitations': "ผลลัพธ์ได้จากข้อมูล synthetic และอาจต้องปรับปรุงเมื่อใช้กับข้อมูลจริง"
            }
        }
        
        self.logger.info("Final comprehensive report created")
        return final_report
    
    def run_comprehensive_comparison(self) -> Dict:
        """รันการเปรียบเทียบอย่างครอบคลุม (เรียบง่าย)"""
        self.logger.info("Starting comprehensive model comparison...")
        
        try:
            tracker = ProgressTracker(6, "Model Comparison")  # ลดจาก 7 เป็น 6
            
            # 1. โหลดผลลัพธ์ทั้งหมด
            results = self.load_all_results()
            tracker.update("Loading all results")
            
            # 2. วิเคราะห์ผลกระทบของ Feature Selection
            feature_impact = self.analyze_feature_selection_impact(results)
            tracker.update("Analyzing feature selection impact")
            
            # 3. เปรียบเทียบโมเดลอย่างละเอียด
            comparison = self.comprehensive_model_comparison(results)
            tracker.update("Comprehensive model comparison")
            
            # 4. สร้างคำแนะนำ
            recommendations = self.generate_recommendations(comparison, feature_impact)
            tracker.update("Generating recommendations")
            
            # 5. สร้างรายงานสุดท้าย
            final_report = self.create_final_report(results, feature_impact, comparison, recommendations)
            tracker.update("Creating final report")
            
            # 6. บันทึกผลลัพธ์
            save_json(final_report, get_output_path('comparison', 'final_report.json'))
            save_json(comparison, get_output_path('comparison', 'model_comparison.json'))
            save_json(feature_impact, get_output_path('comparison', 'feature_selection_impact.json'))
            save_json(recommendations, get_output_path('comparison', 'recommendations.json'))
            
            # สร้าง summary CSV
            summary_data = []
            for model in comparison.keys():
                if model != 'summary':
                    row = {
                        'Model': model,
                        'Test_Accuracy': comparison[model]['performance_metrics']['test_accuracy'],
                        'CV_Accuracy': comparison[model]['performance_metrics']['cv_accuracy_mean'],
                        'Overall_Score': comparison[model]['overall_score'],
                        'Overall_Rank': comparison[model]['rank_overall'],
                        'Training_Time': comparison[model]['time_metrics']['training_time'],
                        'Stability_Score': comparison[model]['normalized_scores']['stability_score']
                    }
                    summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data).sort_values('Overall_Rank')
            save_data(summary_df, get_output_path('comparison', 'model_comparison.csv'))
            
            tracker.update("Saving results")
            tracker.finish()
            
            if VERBOSE:
                print_summary("Model Comparison Results", {
                    'Best Model': comparison['summary']['best_overall'],
                    'Best Accuracy': f"{comparison['summary']['best_accuracy']}",
                    'Feature Selection Improvement': f"{feature_impact['improvement']['accuracy_improvement_percent']:.2f}%",
                    'Feature Reduction': f"{feature_impact['improvement']['feature_reduction_percent']:.1f}%",
                    'Models Compared': len(comparison) - 1,
                    'Statistical Analysis': "Disabled (simplified)"
                })
            
            self.logger.info("Comprehensive model comparison completed successfully")
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"Model comparison failed: {str(e)}")
            raise

def main():
    """ฟังก์ชันหลักสำหรับการทดสอบ"""
    try:
        comparator = ModelComparator()
        final_report = comparator.run_comprehensive_comparison()
        
        print("✅ Model comparison completed successfully!")
        print(f"🏆 Best model: {final_report['main_results']['best_model']}")
        print(f"📊 Best accuracy: {final_report['main_results']['best_accuracy']:.4f}")
        print(f"📁 Results saved to: {COMPARISON_RESULT_DIR}")
        
        return final_report
        
    except Exception as e:
        logger.error(f"Model comparison failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()