# ไฟล์: main.py
# Path: main.py
# วัตถุประสงค์: Main entry point สำหรับรัน pipeline ทั้งหมด (แก้แล้ว)

"""
main.py - Main entry point สำหรับ Feature Selection และ Model Comparison Pipeline
"""

import sys
import os
import argparse
import time
from pathlib import Path

# เพิ่ม src path
sys.path.append(str(Path(__file__).parent / "src"))

# แก้ imports เพื่อหลีกเลี่ยง circular import
import src.config as config
import src.utils as utils
from src.data_preprocessing import DataPreprocessor
from src.feature_selection import FeatureSelector
from src.model_training import ModelTrainer
from src.model_comparison import ModelComparator
from src.visualization import Visualizer

class MLPipeline:
    """Main Pipeline Class สำหรับรัน ML Pipeline ทั้งหมด"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = utils.logger
        self.results = {}
        self.start_time = None
        
        # แสดงข้อมูลเริ่มต้น
        if self.verbose:
            self.print_pipeline_info()
    
    def print_pipeline_info(self):
        """แสดงข้อมูลของ Pipeline"""
        print("=" * 80)
        print("🚀 FEATURE SELECTION & MODEL COMPARISON PIPELINE")
        print("=" * 80)
        print(f"📊 Dataset: {config.DATA_PATH}")
        print(f"🎯 Target: {config.TARGET_COLUMN}")
        print(f"🔧 Normalization: {config.NORMALIZATION_METHOD}")
        print(f"✨ Feature Selection: {config.FEATURE_SELECTION_METHOD}")
        print(f"🤖 Models: {', '.join(config.SELECTED_MODELS)}")
        print(f"📈 CV Folds: {config.CV_FOLDS}")
        print(f"🎲 Random State: {config.RANDOM_STATE}")
        print("=" * 80)
        print()
    
    def validate_setup(self):
        """ตรวจสอบการตั้งค่าและไฟล์ที่จำเป็น"""
        self.logger.info("Validating pipeline setup...")
        
        issues = []
        
        # ตรวจสอบไฟล์ข้อมูล
        if not config.DATA_PATH.exists():
            issues.append(f"❌ Data file not found: {config.DATA_PATH}")
        else:
            self.logger.info(f"✅ Data file found: {config.DATA_PATH}")
        
        # ตรวจสอบโฟลเดอร์ output
        required_dirs = [config.RESULT_DIR, config.MODELS_DIR, config.LOGS_DIR]
        for dir_path in required_dirs:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"📁 Created directory: {dir_path}")
        
        # ตรวจสอบ configuration
        try:
            config.validate_config()
            self.logger.info("✅ Configuration validation passed")
        except ValueError as e:
            issues.append(f"❌ Configuration error: {str(e)}")
        
        # ตรวจสอบ dependencies
        try:
            import pandas as pd
            import numpy as np
            import sklearn
            import matplotlib.pyplot as plt
            import seaborn as sns
            self.logger.info("✅ All required packages available")
        except ImportError as e:
            issues.append(f"❌ Missing dependency: {str(e)}")
        
        if issues:
            for issue in issues:
                print(issue)
                self.logger.error(issue)
            raise RuntimeError("Pipeline validation failed. Please fix the issues above.")
        
        self.logger.info("Pipeline validation completed successfully")
    
    def run_step_1_preprocessing(self):
        """Step 1: Data Preprocessing"""
        print("\n🔄 STEP 1: DATA PREPROCESSING")
        print("-" * 50)
        
        try:
            preprocessor = DataPreprocessor()
            df_normalized, preprocessing_report = preprocessor.run_preprocessing()
            
            self.results['step1'] = {
                'data': df_normalized,
                'report': preprocessing_report,
                'status': 'completed'
            }
            
            print("✅ Step 1 completed successfully!")
            print(f"📊 Normalized data shape: {df_normalized.shape}")
            
        except Exception as e:
            self.logger.error(f"Step 1 failed: {str(e)}")
            self.results['step1'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def run_step_2_feature_selection(self):
        """Step 2: Feature Selection"""
        print("\n🎯 STEP 2: FEATURE SELECTION")
        print("-" * 50)
        
        try:
            selector = FeatureSelector()
            data_selected, selection_report = selector.run_feature_selection()
            
            self.results['step2'] = {
                'data': data_selected,
                'report': selection_report,
                'selected_features': selector.selected_features,
                'status': 'completed'
            }
            
            print("✅ Step 2 completed successfully!")
            print(f"✨ Selected features: {len(selector.selected_features)}")
            print(f"📈 Improvement: {selection_report.get('best_method', {}).get('accuracy', 0):.4f}")
            
        except Exception as e:
            self.logger.error(f"Step 2 failed: {str(e)}")
            self.results['step2'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def run_step_3_model_training(self):
        """Step 3: Model Training"""
        print("\n🤖 STEP 3: MODEL TRAINING")
        print("-" * 50)
        
        try:
            trainer = ModelTrainer()
            models, training_report = trainer.run_model_training()
            
            self.results['step3'] = {
                'models': models,
                'report': training_report,
                'status': 'completed'
            }
            
            best_model = training_report['best_models']['best_test_accuracy']
            best_accuracy = training_report['best_models']['test_accuracy']
            
            print("✅ Step 3 completed successfully!")
            print(f"🏆 Best model: {best_model}")
            print(f"📊 Best accuracy: {best_accuracy:.4f}")
            
        except Exception as e:
            self.logger.error(f"Step 3 failed: {str(e)}")
            self.results['step3'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def run_step_4_model_comparison(self):
        """Step 4: Model Comparison"""
        print("\n📈 STEP 4: MODEL COMPARISON")
        print("-" * 50)
        
        try:
            comparator = ModelComparator()
            final_report = comparator.run_comprehensive_comparison()
            
            self.results['step4'] = {
                'report': final_report,
                'status': 'completed'
            }
            
            print("✅ Step 4 completed successfully!")
            print(f"🎯 Final recommendation: {final_report['recommendations']['model_selection']['recommended_model']}")
            
        except Exception as e:
            self.logger.error(f"Step 4 failed: {str(e)}")
            self.results['step4'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def run_step_5_visualization(self):
        """Step 5: Visualization"""
        print("\n📊 STEP 5: VISUALIZATION")
        print("-" * 50)
        
        try:
            visualizer = Visualizer()
            plots = visualizer.create_all_visualizations()
            
            self.results['step5'] = {
                'plots': plots,
                'status': 'completed'
            }
            
            plots_created = len([p for p in plots.values() if p is not None])
            print("✅ Step 5 completed successfully!")
            print(f"📈 Plots created: {plots_created}")
            
        except Exception as e:
            self.logger.error(f"Step 5 failed: {str(e)}")
            self.results['step5'] = {'status': 'failed', 'error': str(e)}
            print("⚠️  Visualization failed, but pipeline can continue")
    
    def print_final_summary(self):
        """แสดงสรุปผลลัพธ์สุดท้าย"""
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # สรุปจากแต่ละขั้นตอน
        if self.results.get('step1', {}).get('status') == 'completed':
            preprocessing_report = self.results['step1']['report']
            print(f"📊 Data Preprocessing: ✅ {preprocessing_report.get('preprocessing_method', 'N/A')}")
        
        if self.results.get('step2', {}).get('status') == 'completed':
            selection_report = self.results['step2']['report']
            n_features = len(self.results['step2']['selected_features'])
            print(f"✨ Feature Selection: ✅ {n_features} features selected")
        
        if self.results.get('step3', {}).get('status') == 'completed':
            training_report = self.results['step3']['report']
            best_model = training_report['best_models']['best_test_accuracy']
            best_accuracy = training_report['best_models']['test_accuracy']
            print(f"🤖 Model Training: ✅ {best_model} ({best_accuracy:.4f})")
        
        if self.results.get('step4', {}).get('status') == 'completed':
            final_report = self.results['step4']['report']
            recommended_model = final_report['recommendations']['model_selection']['recommended_model']
            print(f"📈 Model Comparison: ✅ {recommended_model} recommended")
        
        if self.results.get('step5', {}).get('status') == 'completed':
            plots_created = len([p for p in self.results['step5']['plots'].values() if p is not None])
            print(f"📊 Visualization: ✅ {plots_created} plots created")
        
        print(f"\n⏱️  Total Time: {total_time:.2f} seconds")
        print(f"📁 Results saved to: {config.RESULT_DIR}")
        print(f"🤖 Models saved to: {config.MODELS_DIR}")
        print(f"📋 Logs saved to: {config.LOGS_DIR}")
        
        # แสดงไฟล์สำคัญที่สร้างขึ้น
        print(f"\n📄 Key Output Files:")
        key_files = [
            config.get_output_path('preprocessing', 'data_normalization.csv'),
            config.get_output_path('feature_selection', 'data_selection.csv'),
            config.get_output_path('evaluation', 'model_performance.csv'),
            config.get_output_path('comparison', 'final_report.json'),
            config.get_output_path('comparison', 'model_comparison.csv')
        ]
        
        for file_path in key_files:
            if file_path.exists():
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path}")
        
        print("=" * 80)
    
    def run_full_pipeline(self, skip_visualization: bool = False):
        """รัน Pipeline ทั้งหมด"""
        self.start_time = time.time()
        
        try:
            # Validation
            self.validate_setup()
            
            # รันแต่ละขั้นตอน
            self.run_step_1_preprocessing()
            self.run_step_2_feature_selection()
            self.run_step_3_model_training()
            self.run_step_4_model_comparison()
            
            if not skip_visualization:
                self.run_step_5_visualization()
            
            # สรุปผลลัพธ์
            self.print_final_summary()
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            print(f"\n❌ Pipeline failed at step: {str(e)}")
            
            # แสดงสถานะของแต่ละขั้นตอน
            print("\n📋 Pipeline Status:")
            steps = ['step1', 'step2', 'step3', 'step4', 'step5']
            step_names = ['Preprocessing', 'Feature Selection', 'Model Training', 'Model Comparison', 'Visualization']
            
            for step, name in zip(steps, step_names):
                status = self.results.get(step, {}).get('status', 'not started')
                if status == 'completed':
                    print(f"   ✅ {name}: Completed")
                elif status == 'failed':
                    error = self.results[step].get('error', 'Unknown error')
                    print(f"   ❌ {name}: Failed ({error})")
                else:
                    print(f"   ⏸️  {name}: Not started")
            
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Feature Selection และ Model Comparison Pipeline')
    parser.add_argument('--skip-visualization', action='store_true', 
                       help='ข้าม visualization step')
    parser.add_argument('--quiet', action='store_true', 
                       help='ลด verbose output')
    parser.add_argument('--step', type=str, choices=['1', '2', '3', '4', '5'], 
                       help='รันเฉพาะ step ที่ระบุ (1=preprocessing, 2=feature_selection, 3=training, 4=comparison, 5=visualization)')
    
    args = parser.parse_args()
    
    # ตั้งค่า verbosity
    verbose = not args.quiet
    
    try:
        pipeline = MLPipeline(verbose=verbose)
        
        if args.step:
            # รันเฉพาะ step ที่ระบุ
            step_num = args.step
            print(f"🎯 Running only step {step_num}...")
            
            if step_num == '1':
                pipeline.run_step_1_preprocessing()
            elif step_num == '2':
                pipeline.run_step_2_feature_selection()
            elif step_num == '3':
                pipeline.run_step_3_model_training()
            elif step_num == '4':
                pipeline.run_step_4_model_comparison()
            elif step_num == '5':
                pipeline.run_step_5_visualization()
                
            print(f"✅ Step {step_num} completed!")
            
        else:
            # รัน pipeline ทั้งหมด
            results = pipeline.run_full_pipeline(skip_visualization=args.skip_visualization)
            
            # แสดงข้อมูลสรุปสุดท้าย
            if verbose and 'step4' in results and results['step4']['status'] == 'completed':
                final_report = results['step4']['report']
                recommendations = final_report['recommendations']
                
                print(f"\n🎯 FINAL RECOMMENDATIONS:")
                print(f"   🏆 Best Model: {recommendations['model_selection']['recommended_model']}")
                print(f"   📊 Expected Accuracy: {recommendations['deployment_considerations']['accuracy_expectation']:.4f}")
                print(f"   ⚡ Prediction Time: {recommendations['deployment_considerations']['prediction_time']:.4f}s")
                print(f"   ✨ Feature Selection Improvement: {recommendations['feature_selection']['effectiveness']:.2f}%")
                
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        utils.logger.error(f"Pipeline failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)