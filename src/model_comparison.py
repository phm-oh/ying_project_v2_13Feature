# ไฟล์: model_comparison.py
# Path: src/model_comparison.py
# วัตถุประสงค์: Step 4 - Model Comparison และสรุปผลลัพธ์สุดท้าย (แก้แล้ว)

"""
model_comparison.py - การเปรียบเทียบโมเดลและสรุปผลลัพธ์สุดท้าย
"""

import pandas as pd
import numpy as np
import logging
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple, Any, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# แก้ imports เป็น relative imports
from .config import *
from .utils import *

class ModelComparator:
    """คลาสสำหรับการเปรียบเทียบโมเดลและสรุปผลลัพธ์"""
    
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
        """วิเคราะห์ผลกระทบของ Feature Selection"""
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
            'lifestyle_features_selected': len([f for f in selected_features if f in LIFESTYLE_FEATURES]),
            'core_features_percentage': len([f for f in selected_features if f in CORE_FEATURES]) / len(CORE_FEATURES) * 100,
            'demographic_features_percentage': len([f for f in selected_features if f in DEMOGRAPHIC_FEATURES]) / len(DEMOGRAPHIC_FEATURES) * 100,
            'lifestyle_features_percentage': len([f for f in selected_features if f in LIFESTYLE_FEATURES]) / len(LIFESTYLE_FEATURES) * 100
        }
        
        impact_analysis['feature_group_analysis'] = feature_group_analysis
        
        self.logger.info("Feature selection impact analysis completed")
        return impact_analysis
    
    def comprehensive_model_comparison(self, results: Dict) -> Dict:
        """เปรียบเทียบโมเดลอย่างละเอียด"""
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
            
            # ความเสถียร
            stability_metrics = {
                'cv_accuracy_cv': cv_data.get('accuracy', {}).get('test_std', 0) / cv_data.get('accuracy', {}).get('test_mean', 1),  # Coefficient of variation
                'overfitting_indicator': cv_data.get('accuracy', {}).get('train_mean', 0) - cv_data.get('accuracy', {}).get('test_mean', 0)
            }
            
            # คะแนนรวม (weighted score)
            weights = {
                'accuracy': 0.4,
                'stability': 0.25,
                'speed': 0.2,
                'precision': 0.15
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
    
    def statistical_analysis_detailed(self, results: Dict) -> Dict:
        """วิเคราะห์ทางสถิติละเอียด"""
        self.logger.info("Performing detailed statistical analysis...")
        
        cv_results = results['cv_results']
        training_report = results['training_report']
        statistical_tests = training_report.get('statistical_significance', {})
        
        models = list(cv_results.keys())
        
        detailed_analysis = {
            'descriptive_statistics': {},
            'effect_sizes': {},
            'confidence_intervals': {},
            'pairwise_comparisons': {},
            'overall_analysis': {}
        }
        
        # Descriptive statistics
        for model in models:
            scores = cv_results[model]['accuracy']['test_scores']
            detailed_analysis['descriptive_statistics'][model] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores),
                'q25': np.percentile(scores, 25),
                'q75': np.percentile(scores, 75),
                'iqr': np.percentile(scores, 75) - np.percentile(scores, 25),
                'skewness': stats.skew(scores),
                'kurtosis': stats.kurtosis(scores)
            }
        
        # Effect sizes และ confidence intervals
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i+1:], i+1):
                scores1 = cv_results[model1]['accuracy']['test_scores']
                scores2 = cv_results[model2]['accuracy']['test_scores']
                
                # Cohen's d (effect size)
                pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                                     (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                                    (len(scores1) + len(scores2) - 2))
                cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                
                # Confidence interval for difference
                diff_mean = np.mean(scores1) - np.mean(scores2)
                diff_std = np.sqrt(np.var(scores1, ddof=1)/len(scores1) + np.var(scores2, ddof=1)/len(scores2))
                t_critical = stats.t.ppf(1 - SIGNIFICANCE_LEVEL/2, len(scores1) + len(scores2) - 2)
                ci_lower = diff_mean - t_critical * diff_std
                ci_upper = diff_mean + t_critical * diff_std
                
                comparison_key = f"{model1}_vs_{model2}"
                detailed_analysis['effect_sizes'][comparison_key] = {
                    'cohens_d': cohens_d,
                    'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d))
                }
                
                detailed_analysis['confidence_intervals'][comparison_key] = {
                    'difference_mean': diff_mean,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'contains_zero': ci_lower <= 0 <= ci_upper
                }
        
        # Pairwise comparisons from statistical tests
        for comparison, test_results in statistical_tests.items():
            if '_vs_' in comparison:
                detailed_analysis['pairwise_comparisons'][comparison] = test_results
        
        # Overall analysis
        all_scores = [cv_results[model]['accuracy']['test_scores'] for model in models]
        
        # ANOVA test
        f_stat, p_value_anova = stats.f_oneway(*all_scores)
        
        # Friedman test (already done but include here)
        friedman_result = statistical_tests.get('friedman_test', {})
        
        detailed_analysis['overall_analysis'] = {
            'anova_test': {
                'f_statistic': f_stat,
                'p_value': p_value_anova,
                'significant': p_value_anova < SIGNIFICANCE_LEVEL
            },
            'friedman_test': friedman_result,
            'number_of_models': len(models),
            'total_comparisons': len(models) * (len(models) - 1) // 2,
            'significant_differences': sum(1 for comp_data in detailed_analysis['pairwise_comparisons'].values()
                                         if isinstance(comp_data, dict) and 
                                         comp_data.get('paired_ttest', {}).get('significant', False))
        }
        
        self.logger.info("Detailed statistical analysis completed")
        return detailed_analysis
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """แปลความหมาย effect size (Cohen's d)"""
        if effect_size < 0.2:
            return "Negligible"
        elif effect_size < 0.5:
            return "Small"
        elif effect_size < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def generate_recommendations(self, comparison: Dict, feature_impact: Dict, statistical_analysis: Dict) -> Dict:
        """สร้างคำแนะนำและข้อสรุป"""
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
            'limitations_and_warnings': {
                'dataset_size': f"ข้อมูลขนาด {2000} records อาจไม่เพียงพอสำหรับการใช้งานจริง",
                'synthetic_data': "ข้อมูลที่ใช้เป็น synthetic data อาจไม่สะท้อนความซับซ้อนของข้อมูลจริง",
                'generalization': "ผลลัพธ์อาจไม่สามารถ generalize ไปยังประชากรอื่นได้",
                'feature_stability': "ควรตรวจสอบความเสถียรของ features ในข้อมูลจริง"
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
        
        # สรุปข้อค้นพบสำคัญ
        key_findings = {
            'feature_selection_impact': f"Feature selection ช่วยปรับปรุง accuracy {feature_impact['improvement']['accuracy_improvement_percent']:.2f}% และลด features {feature_impact['improvement']['feature_reduction_percent']:.1f}%",
            'best_performing_model': f"{best_model} ให้ประสิทธิภาพดีที่สุดด้วย test accuracy {comparison[best_model]['performance_metrics']['test_accuracy']:.4f}",
            'statistical_significance': f"พบความแตกต่างอย่างมีนัยสำคัญใน {statistical_analysis['overall_analysis']['significant_differences']} จาก {statistical_analysis['overall_analysis']['total_comparisons']} การเปรียบเทียบ",
            'feature_groups_importance': self._analyze_feature_groups(feature_impact['feature_group_analysis'])
        }
        
        recommendations['key_findings'] = key_findings
        
        self.logger.info("Recommendations generated")
        return recommendations
    
    def _analyze_feature_groups(self, feature_group_analysis: Dict) -> str:
        """วิเคราะห์ความสำคัญของกลุ่ม features"""
        core_pct = feature_group_analysis['core_features_percentage']
        demo_pct = feature_group_analysis['demographic_features_percentage']
        life_pct = feature_group_analysis['lifestyle_features_percentage']
        
        if core_pct > 80:
            return f"Core features มีความสำคัญสูงสุด ({core_pct:.1f}% ถูกเลือก)"
        elif demo_pct > 60:
            return f"Demographic features มีความสำคัญสูง ({demo_pct:.1f}% ถูกเลือก)"
        else:
            return f"Features จากทุกกลุ่มมีความสำคัญ (Core: {core_pct:.1f}%, Demographic: {demo_pct:.1f}%, Lifestyle: {life_pct:.1f}%)"
    
    def create_final_report(self, results: Dict, feature_impact: Dict, 
                          comparison: Dict, statistical_analysis: Dict, 
                          recommendations: Dict) -> Dict:
        """สร้างรายงานสุดท้าย"""
        self.logger.info("Creating final comprehensive report...")
        
        # สรุปข้อมูลหลัก
        dataset_summary = {
            'original_features': len(CORE_FEATURES) + len(DEMOGRAPHIC_FEATURES) + len(LIFESTYLE_FEATURES),
            'selected_features': feature_impact['final_selection_performance']['n_features'],
            'total_samples': 2000,  # จาก config
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
            'feature_reduction': feature_impact['improvement']['feature_reduction_percent'],
            'statistical_significance_found': statistical_analysis['overall_analysis']['significant_differences'] > 0
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
                    'random_state': RANDOM_STATE
                }
            },
            'dataset_summary': dataset_summary,
            'main_results': main_results,
            'feature_selection_analysis': feature_impact,
            'model_comparison': comparison,
            'statistical_analysis': statistical_analysis,
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
        """รันการเปรียบเทียบอย่างครอบคลุม (เอา decorator ออก)"""
        self.logger.info("Starting comprehensive model comparison...")
        
        try:
            tracker = ProgressTracker(7, "Model Comparison")
            
            # 1. โหลดผลลัพธ์ทั้งหมด
            results = self.load_all_results()
            tracker.update("Loading all results")
            
            # 2. วิเคราะห์ผลกระทบของ Feature Selection
            feature_impact = self.analyze_feature_selection_impact(results)
            tracker.update("Analyzing feature selection impact")
            
            # 3. เปรียบเทียบโมเดลอย่างละเอียด
            comparison = self.comprehensive_model_comparison(results)
            tracker.update("Comprehensive model comparison")
            
            # 4. วิเคราะห์ทางสถิติละเอียด
            statistical_analysis = self.statistical_analysis_detailed(results)
            tracker.update("Detailed statistical analysis")
            
            # 5. สร้างคำแนะนำ
            recommendations = self.generate_recommendations(comparison, feature_impact, statistical_analysis)
            tracker.update("Generating recommendations")
            
            # 6. สร้างรายงานสุดท้าย
            final_report = self.create_final_report(results, feature_impact, comparison, 
                                                  statistical_analysis, recommendations)
            tracker.update("Creating final report")
            
            # 7. บันทึกผลลัพธ์
            save_json(final_report, get_output_path('comparison', 'final_report.json'))
            save_json(comparison, get_output_path('comparison', 'model_comparison.json'))
            save_json(feature_impact, get_output_path('comparison', 'feature_selection_impact.json'))
            save_json(statistical_analysis, get_output_path('comparison', 'statistical_analysis.json'))
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
                    'Statistical Significance': statistical_analysis['overall_analysis']['significant_differences'],
                    'Models Compared': len(comparison) - 1
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