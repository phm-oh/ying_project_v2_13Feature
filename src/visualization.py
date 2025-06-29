# ไฟล์: visualization.py
# Path: src/visualization.py
# วัตถุประสงค์: สร้างกราฟและการแสดงผลต่างๆ สำหรับการวิเคราะห์ (แก้แล้ว)

"""
visualization.py - การสร้างกราฟและการแสดงผลสำหรับการวิเคราะห์
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from typing import Dict, Tuple, Any, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# แก้ imports เป็น relative imports
from .config import *
from .utils import *

class Visualizer:
    """คลาสสำหรับการสร้างกราฟและการแสดงผล"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        setup_plot_style()
        
    def load_results_data(self) -> Dict:
        """โหลดข้อมูลผลลัพธ์ทั้งหมด"""
        self.logger.info("Loading results data for visualization...")
        
        data = {}
        
        try:
            # โหลดผลการ feature selection
            data['feature_report'] = load_json(get_output_path('feature_selection', 'feature_selection_report.json'))
            data['selected_features'] = load_json(get_output_path('feature_selection', 'selected_features.json'))
            
            # โหลดผลการฝึกสอนโมเดล
            data['training_report'] = load_json(get_output_path('evaluation', 'training_report.json'))
            data['cv_results'] = load_json(get_output_path('evaluation', 'cv_results.json'))
            data['test_results'] = load_json(get_output_path('evaluation', 'test_results.json'))
            
            # โหลดข้อมูล performance
            data['performance_df'] = load_data(get_output_path('evaluation', 'model_performance.csv'))
            
            # โหลด feature scores (ถ้ามี)
            try:
                data['feature_scores'] = load_data(get_output_path('feature_selection', 'selection_scores.csv'))
            except:
                data['feature_scores'] = None
                
        except Exception as e:
            self.logger.error(f"Error loading results data: {str(e)}")
            raise
        
        self.logger.info("Results data loaded successfully")
        return data
    
    def plot_feature_selection_comparison(self, feature_report: Dict) -> plt.Figure:
        """สร้างกราฟเปรียบเทียบวิธี Feature Selection"""
        self.logger.info("Creating feature selection comparison plot...")
        
        # เตรียมข้อมูล
        methods = []
        accuracies = []
        n_features = []
        
        comparison_results = feature_report.get('comparison_results', {})
        
        for method_name, results in comparison_results.items():
            if 'mean_score' in results:
                methods.append(method_name.replace('_', ' ').title())
                accuracies.append(results['mean_score'])
                n_features.append(results['n_features_selected'])
        
        # สร้างกราฟ
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # กราฟ 1: Accuracy Comparison
        bars1 = ax1.bar(methods, accuracies, alpha=0.8, color=sns.color_palette("Set2", len(methods)))
        ax1.set_title('Feature Selection Methods - Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cross-Validation Accuracy')
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='x', rotation=45)
        
        # เพิ่มค่าบนแท่งกราฟ
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # กราฟ 2: Number of Features
        bars2 = ax2.bar(methods, n_features, alpha=0.8, color=sns.color_palette("Set3", len(methods)))
        ax2.set_title('Feature Selection Methods - Number of Features', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Selected Features')
        ax2.tick_params(axis='x', rotation=45)
        
        # เพิ่มค่าบนแท่งกราฟ
        for bar, n_feat in zip(bars2, n_features):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{n_feat}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        save_plot(fig, 'feature_selection_comparison.png', 'feature_selection')
        
        return fig
    
    def plot_feature_importance(self, feature_scores: pd.DataFrame, top_n: int = 15) -> plt.Figure:
        """สร้างกราฟ Feature Importance"""
        self.logger.info("Creating feature importance plot...")
        
        if feature_scores is None or feature_scores.empty:
            self.logger.warning("No feature scores available")
            return None
        
        # เรียงลำดับและเลือก top features
        top_features = feature_scores.head(top_n)
        
        # สร้างกราฟ
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(top_features['feature'], top_features['importance'], 
                      alpha=0.8, color=sns.color_palette("viridis", len(top_features)))
        
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
        ax.set_xlabel('Importance Score')
        ax.invert_yaxis()  # เรียงจากมากไปน้อย
        
        # เพิ่มค่าข้างแท่งกราฟ
        for bar, score in zip(bars, top_features['importance']):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        save_plot(fig, 'feature_importance.png', 'feature_selection')
        
        return fig
    
    def plot_model_performance_comparison(self, performance_df: pd.DataFrame) -> plt.Figure:
        """สร้างกราฟเปรียบเทียบประสิทธิภาพโมเดล"""
        self.logger.info("Creating model performance comparison plot...")
        
        # เตรียมข้อมูล
        models = performance_df['Model'].tolist()
        metrics = ['CV_Accuracy_Mean', 'Test_Accuracy', 'CV_Precision_Mean', 'CV_Recall_Mean', 'CV_F1_Mean']
        metric_labels = ['CV Accuracy', 'Test Accuracy', 'CV Precision', 'CV Recall', 'CV F1-Score']
        
        # สร้างกราฟ
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # กราฟแต่ละ metric
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]
            values = performance_df[metric].tolist()
            
            bars = ax.bar(models, values, alpha=0.8, 
                         color=sns.color_palette("Set1", len(models)))
            
            ax.set_title(label, fontsize=14, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis='x', rotation=45)
            
            # เพิ่มค่าบนแท่งกราฟ
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # กราฟรวม - Radar Chart แบบง่าย
        ax = axes[5]
        x = np.arange(len(models))
        width = 0.15
        
        metrics_for_comparison = ['CV_Accuracy_Mean', 'CV_Precision_Mean', 'CV_Recall_Mean', 'CV_F1_Mean']
        colors = ['blue', 'orange', 'green', 'red']
        
        for i, metric in enumerate(metrics_for_comparison):
            values = performance_df[metric].tolist()
            ax.bar(x + i*width, values, width, label=metric.replace('CV_', '').replace('_Mean', ''),
                  alpha=0.8, color=colors[i])
        
        ax.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        save_plot(fig, 'model_performance_comparison.png', 'evaluation')
        
        return fig
    
    def plot_cross_validation_scores(self, cv_results: Dict) -> plt.Figure:
        """สร้างกราฟ Cross-Validation Scores"""
        self.logger.info("Creating cross-validation scores plot...")
        
        # เตรียมข้อมูล
        models = list(cv_results.keys())
        accuracy_scores = [cv_results[model]['accuracy']['test_scores'] for model in models]
        
        # สร้างกราฟ
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # กราฟ 1: Box Plot
        ax1.boxplot(accuracy_scores, labels=models)
        ax1.set_title('Cross-Validation Accuracy Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # กราฟ 2: Line Plot แสดง CV scores แต่ละ fold
        for i, model in enumerate(models):
            scores = cv_results[model]['accuracy']['test_scores']
            folds = range(1, len(scores) + 1)
            ax2.plot(folds, scores, marker='o', label=model, linewidth=2, markersize=6)
        
        ax2.set_title('Cross-Validation Accuracy by Fold', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Fold Number')
        ax2.set_ylabel('Accuracy Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_plot(fig, 'cross_validation_scores.png', 'evaluation')
        
        return fig
    
    def plot_confusion_matrices_enhanced(self, training_report: Dict) -> plt.Figure:
        """สร้างกราฟ Confusion Matrix ที่ปรับปรุงแล้ว"""
        self.logger.info("Creating enhanced confusion matrices...")
        
        confusion_matrices = training_report.get('confusion_matrices', {})
        predictions = training_report.get('predictions', {})
        
        if not confusion_matrices:
            self.logger.warning("No confusion matrices available")
            return None
        
        # โหลด target labels
        try:
            target_mapping_path = get_output_path('preprocessing', 'target_mapping.json')
            target_mapping = load_json(target_mapping_path)
            thai_labels = [k for k, v in sorted(target_mapping.items(), key=lambda x: x[1])]
            labels = get_display_labels(thai_labels)
        except:
            labels = get_display_labels()
        
        # สร้างกราฟ
        n_models = len(confusion_matrices)
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
            cm_array = np.array(cm)
            
            # กราฟ 1: Confusion Matrix ปกติ
            sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels,
                       ax=axes[0, idx])
            axes[0, idx].set_title(f'{model_name} - Confusion Matrix')
            axes[0, idx].set_xlabel('Predicted Label')
            axes[0, idx].set_ylabel('True Label')
            
            # กราฟ 2: Confusion Matrix เปอร์เซ็นต์
            cm_percent = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis] * 100
            sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                       xticklabels=labels, yticklabels=labels,
                       ax=axes[1, idx])
            axes[1, idx].set_title(f'{model_name} - Confusion Matrix (%)')
            axes[1, idx].set_xlabel('Predicted Label')
            axes[1, idx].set_ylabel('True Label')
            
            # คำนวณ accuracy ของแต่ละคลาส
            class_accuracies = cm_percent.diagonal()
            for i, acc in enumerate(class_accuracies):
                axes[1, idx].text(i, i, f'\n({acc:.1f}%)', 
                                ha='center', va='center', fontweight='bold', color='red')
        
        plt.tight_layout()
        save_plot(fig, 'confusion_matrices_enhanced.png', 'evaluation')
        
        return fig
    
    def plot_statistical_significance(self, training_report: Dict) -> plt.Figure:
      """สร้างกราฟแสดงการทดสอบนัยสำคัญทางสถิติ"""
      self.logger.info("Creating statistical significance plot...")
    
      statistical_tests = training_report.get('statistical_significance', {})
    
      if not statistical_tests:
        self.logger.warning("No statistical test results available")
        return None
    
    # เตรียมข้อมูลสำหรับ heatmap
      model_names = list(training_report['model_summary'].keys())
      n_models = len(model_names)
    
      if n_models < 2:
        self.logger.warning("Need at least 2 models for statistical comparison")
        return None
    
       # สร้าง matrix สำหรับ p-values
      p_value_matrix = np.ones((n_models, n_models))
      significance_matrix = np.zeros((n_models, n_models))
    
      for comparison, tests in statistical_tests.items():
        if comparison == 'friedman_test':
            continue
            
        model1, model2 = comparison.split('_vs_')
        try:
            idx1 = model_names.index(model1)
            idx2 = model_names.index(model2)
            
            # ใช้ p-value จาก paired t-test
            if 'paired_ttest' in tests:
                p_val = tests['paired_ttest']['p_value']
                is_significant = tests['paired_ttest']['significant']
                
                p_value_matrix[idx1, idx2] = p_val
                p_value_matrix[idx2, idx1] = p_val
                
                if is_significant:
                    significance_matrix[idx1, idx2] = 1
                    significance_matrix[idx2, idx1] = 1
                    
        except ValueError:
            continue
    
    # สร้างกราฟ
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # กราฟ 1: P-values heatmap
      mask = p_value_matrix == 1  # ซ่อน diagonal
      sns.heatmap(p_value_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',  # <- เปลี่ยนจาก 'd' เป็น '.3f'
               mask=mask, xticklabels=model_names, yticklabels=model_names,
               ax=ax1, cbar_kws={'label': 'P-value'})
      ax1.set_title('Statistical Significance - P-values\n(Lower is more significant)', 
                 fontsize=14, fontweight='bold')
    
    # กราฟ 2: Significance matrix
      sns.heatmap(significance_matrix, annot=True, fmt='.0f', cmap='RdYlGn',  # <- เปลี่ยนจาก 'd' เป็น '.0f'
               xticklabels=model_names, yticklabels=model_names,
               ax=ax2, cbar_kws={'label': 'Significant (1) / Not Significant (0)'})
      ax2.set_title(f'Statistical Significance Matrix\n(α = {SIGNIFICANCE_LEVEL})', 
                 fontsize=14, fontweight='bold')
    
      plt.tight_layout()
      save_plot(fig, 'statistical_significance.png', 'evaluation')
    
      return fig
   
   
    def plot_training_time_analysis(self, performance_df: pd.DataFrame) -> plt.Figure:
        """สร้างกราฟวิเคราะห์เวลาการฝึกสอน"""
        self.logger.info("Creating training time analysis plot...")
        
        models = performance_df['Model'].tolist()
        training_times = performance_df['Training_Time'].tolist()
        prediction_times = performance_df['Prediction_Time'].tolist()
        test_accuracies = performance_df['Test_Accuracy'].tolist()
        
        # สร้างกราฟ
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # กราฟ 1: Training vs Prediction Time
        bars1 = ax1.bar(models, training_times, alpha=0.7, label='Training Time', color='skyblue')
        bars2 = ax1.bar(models, prediction_times, bottom=training_times, alpha=0.7, 
                       label='Prediction Time', color='orange')
        
        ax1.set_title('Training and Prediction Time Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        
        # เพิ่มค่าบนแท่งกราฟ
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            total_time = training_times[i] + prediction_times[i]
            ax1.text(bar1.get_x() + bar1.get_width()/2., total_time + 0.1,
                    f'{total_time:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # กราฟ 2: Accuracy vs Training Time (Efficiency Plot)
        scatter = ax2.scatter(training_times, test_accuracies, s=200, alpha=0.7,
                            c=range(len(models)), cmap='viridis')
        
        # เพิ่ม labels สำหรับแต่ละจุด
        for i, model in enumerate(models):
            ax2.annotate(model, (training_times[i], test_accuracies[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax2.set_title('Model Efficiency: Accuracy vs Training Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Training Time (seconds)')
        ax2.set_ylabel('Test Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_plot(fig, 'training_time_analysis.png', 'evaluation')
        
        return fig
    
    def create_summary_dashboard(self, data: Dict) -> plt.Figure:
        """สร้าง Dashboard สรุปผลลัพธ์ทั้งหมด"""
        self.logger.info("Creating summary dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # แบ่ง layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Feature Selection Summary
        ax1 = fig.add_subplot(gs[0, :2])
        feature_report = data['feature_report']
        best_method = feature_report['best_method']['name']
        best_accuracy = feature_report['best_method']['accuracy']
        n_features = feature_report['best_method']['n_features']
        
        ax1.text(0.5, 0.7, 'Feature Selection Results', ha='center', va='center',
                fontsize=16, fontweight='bold', transform=ax1.transAxes)
        ax1.text(0.5, 0.4, f'Best Method: {best_method}', ha='center', va='center',
                fontsize=14, transform=ax1.transAxes)
        ax1.text(0.5, 0.2, f'Accuracy: {best_accuracy:.4f} | Features: {n_features}', 
                ha='center', va='center', fontsize=12, transform=ax1.transAxes)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # 2. Model Performance Summary
        ax2 = fig.add_subplot(gs[0, 2:])
        training_report = data['training_report']
        best_model = training_report['best_models']['best_test_accuracy']
        best_test_acc = training_report['best_models']['test_accuracy']
        best_cv_acc = training_report['best_models']['cv_accuracy']
        
        ax2.text(0.5, 0.7, 'Model Training Results', ha='center', va='center',
                fontsize=16, fontweight='bold', transform=ax2.transAxes)
        ax2.text(0.5, 0.4, f'Best Model: {best_model}', ha='center', va='center',
                fontsize=14, transform=ax2.transAxes)
        ax2.text(0.5, 0.2, f'Test Acc: {best_test_acc:.4f} | CV Acc: {best_cv_acc:.4f}', 
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        # 3. Model Comparison Bar Chart
        ax3 = fig.add_subplot(gs[1, :])
        performance_df = data['performance_df']
        models = performance_df['Model'].tolist()
        test_accs = performance_df['Test_Accuracy'].tolist()
        cv_accs = performance_df['CV_Accuracy_Mean'].tolist()
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, cv_accs, width, label='CV Accuracy', alpha=0.8)
        bars2 = ax3.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
        
        ax3.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # เพิ่มค่าบนแท่งกราฟ
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Feature Importance (ถ้ามี)
        if data['feature_scores'] is not None:
            ax4 = fig.add_subplot(gs[2, :2])
            top_features = data['feature_scores'].head(8)
            ax4.barh(top_features['feature'], top_features['importance'], alpha=0.8)
            ax4.set_title('Top Feature Importance', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Importance Score')
            ax4.invert_yaxis()
        
        # 5. Statistical Significance
        ax5 = fig.add_subplot(gs[2, 2:])
        stat_tests = training_report.get('statistical_significance', {})
        significant_comparisons = sum(1 for test_data in stat_tests.values() 
                                    if isinstance(test_data, dict) and 
                                    'paired_ttest' in test_data and 
                                    test_data['paired_ttest'].get('significant', False))
        total_comparisons = len([k for k in stat_tests.keys() if '_vs_' in k])
        
        ax5.text(0.5, 0.7, 'Statistical Significance', ha='center', va='center',
                fontsize=14, fontweight='bold', transform=ax5.transAxes)
        ax5.text(0.5, 0.4, f'Significant Differences: {significant_comparisons}/{total_comparisons}', 
                ha='center', va='center', fontsize=12, transform=ax5.transAxes)
        ax5.text(0.5, 0.2, f'Significance Level: α = {SIGNIFICANCE_LEVEL}', 
                ha='center', va='center', fontsize=10, transform=ax5.transAxes)
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        
        # 6. Configuration Summary
        ax6 = fig.add_subplot(gs[3, :])
        config_text = f"""
        Configuration Summary:
        • Dataset: {TARGET_COLUMN} classification with {len(data['selected_features']['selected_features'])} features
        • Feature Selection: {FEATURE_SELECTION_METHOD} selection method
        • Cross-Validation: {CV_FOLDS}-fold stratified CV
        • Models: {', '.join(models)}
        • Normalization: {NORMALIZATION_METHOD}
        • Random State: {RANDOM_STATE}
        """
        
        ax6.text(0.05, 0.5, config_text, ha='left', va='center',
                fontsize=11, transform=ax6.transAxes, fontfamily='monospace')
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        
        # เพิ่ม title หลัก
        fig.suptitle('Machine Learning Pipeline - Summary Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        save_plot(fig, 'summary_dashboard.png', 'comparison')
        
        return fig
    
    def create_all_visualizations(self) -> Dict:
        """สร้างกราฟทั้งหมด (เอา decorator ออก)"""
        self.logger.info("Creating all visualizations...")
        
        try:
            tracker = ProgressTracker(8, "Visualization")
            
            # โหลดข้อมูล
            data = self.load_results_data()
            tracker.update("Loading results data")
            
            visualizations = {}
            
            # 1. Feature Selection Comparison
            fig1 = self.plot_feature_selection_comparison(data['feature_report'])
            visualizations['feature_selection_comparison'] = fig1
            tracker.update("Feature selection comparison")
            
            # 2. Feature Importance
            fig2 = self.plot_feature_importance(data['feature_scores'])
            visualizations['feature_importance'] = fig2
            tracker.update("Feature importance")
            
            # 3. Model Performance Comparison
            fig3 = self.plot_model_performance_comparison(data['performance_df'])
            visualizations['model_performance'] = fig3
            tracker.update("Model performance comparison")
            
            # 4. Cross-Validation Scores
            fig4 = self.plot_cross_validation_scores(data['cv_results'])
            visualizations['cv_scores'] = fig4
            tracker.update("Cross-validation scores")
            
            # 5. Enhanced Confusion Matrices
            fig5 = self.plot_confusion_matrices_enhanced(data['training_report'])
            visualizations['confusion_matrices'] = fig5
            tracker.update("Confusion matrices")
            
            # 6. Statistical Significance
            fig6 = self.plot_statistical_significance(data['training_report'])
            visualizations['statistical_significance'] = fig6
            tracker.update("Statistical significance")
            
            # 7. Training Time Analysis
            fig7 = self.plot_training_time_analysis(data['performance_df'])
            visualizations['training_time'] = fig7
            tracker.update("Training time analysis")
            
            # 8. Summary Dashboard
            fig8 = self.create_summary_dashboard(data)
            visualizations['summary_dashboard'] = fig8
            tracker.update("Summary dashboard")
            
            tracker.finish()
            
            if VERBOSE:
                print_summary("Visualization Results", {
                    'Plots Created': len([v for v in visualizations.values() if v is not None]),
                    'Feature Selection Plots': 2,
                    'Model Evaluation Plots': 4,
                    'Summary Plots': 2,
                    'Output Directory': COMPARISON_RESULT_DIR
                })
            
            self.logger.info("All visualizations created successfully")
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {str(e)}")
            raise

def main():
    """ฟังก์ชันหลักสำหรับการทดสอบ"""
    try:
        visualizer = Visualizer()
        plots = visualizer.create_all_visualizations()
        
        print("✅ Visualization completed successfully!")
        print(f"📊 Plots created: {len([p for p in plots.values() if p is not None])}")
        # print(f"📁 Plots saved to: {COMPARISON_RESULT_DIR}")
        print(f"📁 Output Directory: {get_output_path('comparison', '')}")
        
        return plots
        
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()