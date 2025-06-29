# ไฟล์: domain_knowledge_evaluator.py
# Path: domain_knowledge_evaluator.py (สร้างไฟล์ใหม่ใน root directory)
# วัตถุประสงค์: ตรวจสอบความสมเหตุสมผลของ Feature Selection

"""
Domain Knowledge Evaluator - ตรวจสอบว่า Feature Selection สมเหตุสมผลหรือไม่
สำหรับการป้องกันการตั้งคำถามของนักวิชาการ
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DomainKnowledgeEvaluator:
    """คลาสสำหรับประเมินความสมเหตุสมผลของ Feature Selection"""
    
    def __init__(self):
        # Define feature categories based on education domain knowledge
        self.feature_categories = {
            'Academic_Scores': [
                'คะแนนคณิตศาสตร์', 'คะแนนคอมพิวเตอร์', 'คะแนนภาษาไทย', 
                'คะแนนวิทยาศาสตร์', 'คะแนนศิลปะ'
            ],
            'Core_Skills': [
                'ทักษะการคิดเชิงตรรกะ', 'ทักษะความคิดสร้างสรรค์', 'ทักษะการแก้ปัญหา'
            ],
            'Career_Interests': [
                'ความสนใจด้านตัวเลข', 'ความสนใจด้านเทคโนโลยี', 'ความสนใจด้านการทำอาหาร'
            ],
            'Demographics': [
                'อายุ', 'เพศ', 'รายได้ครอบครัว', 'การศึกษาของผู้ปกครอง', 'จำนวนพี่น้อง'
            ],
            'Lifestyle_Personal': [
                'ชั่วโมงการนอน', 'ความถี่การออกกำลังกาย', 'ชั่วโมงใช้โซเชียลมีเดีย',
                'ชอบอ่านหนังสือ', 'ประเภทเพลงที่ชอบ'
            ]
        }
        
        # Educational justification for each category
        self.category_justifications = {
            'Academic_Scores': {
                'relevance': 'HIGH',
                'justification': 'คะแนนวิชาหลักแสดงความสามารถและความถนัดในสาขาต่างๆ ควรเป็นปัจจัยหลักในการเลือกแผนก',
                'expected_percentage': '40-60%',
                'research_support': 'Supported by educational psychology research on academic aptitude'
            },
            'Core_Skills': {
                'relevance': 'HIGH',
                'justification': 'ทักษะการคิดเป็นพื้นฐานสำคัญในการเรียนรู้ แต่ละแผนกต้องการทักษะที่แตกต่างกัน',
                'expected_percentage': '15-25%',
                'research_support': 'Cognitive skills are strong predictors of academic success'
            },
            'Career_Interests': {
                'relevance': 'HIGH',
                'justification': 'ความสนใจด้านอาชีพเป็นตัวบ่งชี้แรงจูงใจและความตั้งใจในการเรียน',
                'expected_percentage': '15-25%',
                'research_support': 'Vocational interest theory supports this factor'
            },
            'Demographics': {
                'relevance': 'LOW',
                'justification': 'ข้อมูลประชากรอาจมีผลต่อโอกาสทางการศึกษา แต่ไม่ควรเป็นปัจจัยตัดสินใจหลัก',
                'expected_percentage': '0-10%',
                'research_support': 'Should be minimized to ensure educational equity'
            },
            'Lifestyle_Personal': {
                'relevance': 'NONE',
                'justification': 'ปัจจัยส่วนตัวเช่น การนอน การออกกำลังกาย ไม่ควรใช้ในการตัดสินใจทางการศึกษา',
                'expected_percentage': '0%',
                'research_support': 'Personal lifestyle should not determine educational opportunities'
            }
        }
    
    def evaluate_feature_selection(self, selected_features: List[str]) -> Dict:
        """ประเมินความสมเหตุสมผลของ feature selection"""
        
        evaluation = {
            'total_features': len(selected_features),
            'category_analysis': {},
            'red_flags': [],
            'recommendations': [],
            'overall_score': 0,
            'academic_justification': ""
        }
        
        # วิเคราะห์แต่ละหมวดหมู่
        for category, features in self.feature_categories.items():
            selected_in_category = [f for f in selected_features if f in features]
            percentage = len(selected_in_category) / len(selected_features) * 100 if len(selected_features) > 0 else 0
            
            analysis = {
                'features_selected': selected_in_category,
                'count': len(selected_in_category),
                'percentage': percentage,
                'justification': self.category_justifications[category],
                'status': self._evaluate_category_status(category, percentage)
            }
            
            evaluation['category_analysis'][category] = analysis
            
            # ตรวจหา red flags
            if category == 'Lifestyle_Personal' and len(selected_in_category) > 0:
                evaluation['red_flags'].append({
                    'type': 'LIFESTYLE_FACTORS_PRESENT',
                    'message': f"🚨 พบปัจจัยไลฟ์สไตล์: {selected_in_category}",
                    'severity': 'HIGH',
                    'academic_concern': 'นักวิชาการจะตั้งคำถามว่าทำไมปัจจัยส่วนตัวเข้ามาในการตัดสินใจทางการศึกษา'
                })
            
            if category == 'Demographics' and percentage > 15:
                evaluation['red_flags'].append({
                    'type': 'HIGH_DEMOGRAPHIC_BIAS',
                    'message': f"⚠️ ปัจจัยประชากรศาสตร์สูงเกินไป: {percentage:.1f}%",
                    'severity': 'MEDIUM',
                    'academic_concern': 'อาจถูกมองว่าเป็นการเลือกปฏิบัติทางสังคม'
                })
        
        # คำนวณคะแนนรวม
        evaluation['overall_score'] = self._calculate_overall_score(evaluation['category_analysis'])
        
        # สร้างคำแนะนำ
        evaluation['recommendations'] = self._generate_recommendations(evaluation)
        
        # สร้างเหตุผลทางวิชาการ
        evaluation['academic_justification'] = self._generate_academic_justification(evaluation)
        
        return evaluation
    
    def _evaluate_category_status(self, category: str, percentage: float) -> str:
        """ประเมินสถานะของแต่ละหมวดหมู่"""
        if category in ['Academic_Scores', 'Core_Skills', 'Career_Interests']:
            if percentage >= 15:
                return 'GOOD'
            elif percentage >= 10:
                return 'ACCEPTABLE'
            else:
                return 'TOO_LOW'
        
        elif category == 'Demographics':
            if percentage <= 10:
                return 'GOOD'
            elif percentage <= 20:
                return 'ACCEPTABLE'
            else:
                return 'TOO_HIGH'
        
        elif category == 'Lifestyle_Personal':
            if percentage == 0:
                return 'GOOD'
            else:
                return 'UNACCEPTABLE'
        
        return 'UNKNOWN'
    
    def _calculate_overall_score(self, category_analysis: Dict) -> float:
        """คำนวณคะแนนรวม (0-100)"""
        score = 0
        
        # Academic factors (ให้คะแนนสูง)
        academic_pct = category_analysis['Academic_Scores']['percentage']
        skills_pct = category_analysis['Core_Skills']['percentage']
        interests_pct = category_analysis['Career_Interests']['percentage']
        
        # คะแนนจากปัจจัยทางการศึกษา (70 คะแนน)
        academic_score = min(academic_pct / 50 * 30, 30)  # สูงสุด 30 คะแนน
        skills_score = min(skills_pct / 25 * 20, 20)      # สูงสุด 20 คะแนน
        interests_score = min(interests_pct / 25 * 20, 20) # สูงสุด 20 คะแนน
        
        # หักคะแนนจากปัจจัยที่ไม่เหมาะสม (30 คะแนน)
        demo_pct = category_analysis['Demographics']['percentage']
        lifestyle_pct = category_analysis['Lifestyle_Personal']['percentage']
        
        demo_penalty = max(0, (demo_pct - 10) / 10 * 15)    # หักสูงสุด 15 คะแนน
        lifestyle_penalty = lifestyle_pct / 10 * 15         # หักสูงสุด 15 คะแนน
        
        score = academic_score + skills_score + interests_score + 30 - demo_penalty - lifestyle_penalty
        
        return max(0, min(100, score))
    
    def _generate_recommendations(self, evaluation: Dict) -> List[str]:
        """สร้างคำแนะนำสำหรับการปรับปรุง"""
        recommendations = []
        
        # ตรวจสอบ red flags
        for flag in evaluation['red_flags']:
            if flag['type'] == 'LIFESTYLE_FACTORS_PRESENT':
                recommendations.append(
                    "🚨 ลบปัจจัยไลฟ์สไตล์ออกจาก feature selection - ใช้ domain knowledge filtering"
                )
            elif flag['type'] == 'HIGH_DEMOGRAPHIC_BIAS':
                recommendations.append(
                    "⚠️ ลดปัจจัยประชากรศาสตร์ - จำกัดให้เหลือเฉพาะอายุและเพศ"
                )
        
        # ตรวจสอบสัดส่วนหมวดหมู่
        category_analysis = evaluation['category_analysis']
        
        if category_analysis['Academic_Scores']['percentage'] < 40:
            recommendations.append(
                "📚 เพิ่มปัจจัยคะแนนวิชาหลัก - ควรมีอย่างน้อย 40% ของ selected features"
            )
        
        if category_analysis['Core_Skills']['percentage'] < 15:
            recommendations.append(
                "🧠 เพิ่มปัจจัยทักษะการคิด - สำคัญต่อการเรียนรู้"
            )
        
        if category_analysis['Career_Interests']['percentage'] < 15:
            recommendations.append(
                "🎯 เพิ่มปัจจัยความสนใจด้านอาชีพ - สำคัญต่อแรงจูงใจ"
            )
        
        # คำแนะนำเชิงเทคนิค
        if evaluation['overall_score'] < 70:
            recommendations.append(
                "🔧 ใช้ Enhanced Sequential Selection with Domain Knowledge Filtering"
            )
            recommendations.append(
                "📊 ปรับพารามิเตอร์ของ Feature Selection Algorithm"
            )
        
        return recommendations
    
    def _generate_academic_justification(self, evaluation: Dict) -> str:
        """สร้างเหตุผลทางวิชาการสำหรับการนำเสนอ"""
        
        academic_pct = evaluation['category_analysis']['Academic_Scores']['percentage']
        skills_pct = evaluation['category_analysis']['Core_Skills']['percentage']
        interests_pct = evaluation['category_analysis']['Career_Interests']['percentage']
        demo_pct = evaluation['category_analysis']['Demographics']['percentage']
        lifestyle_pct = evaluation['category_analysis']['Lifestyle_Personal']['percentage']
        
        total_educational = academic_pct + skills_pct + interests_pct
        total_personal = demo_pct + lifestyle_pct
        
        justification = f"""
🎓 **เหตุผลทางวิชาการสำหรับ Feature Selection**

📊 **ผลการวิเคราะห์:**
- คะแนนคุณภาพโดยรวม: {evaluation['overall_score']:.1f}/100
- ปัจจัยทางการศึกษา: {total_educational:.1f}%
- ปัจจัยส่วนตัว: {total_personal:.1f}%

📚 **หลักการทางการศึกษา:**
1. **Academic Achievement Theory**: คะแนนวิชาหลักเป็นตัวทำนายความสำเร็จในแต่ละสาขา
2. **Cognitive Skills Framework**: ทักษะการคิดเป็นพื้นฐานการเรียนรู้
3. **Vocational Interest Theory**: ความสนใจเป็นตัวขับเคลื่อนแรงจูงใจ
4. **Educational Equity Principle**: หลีกเลี่ยงปัจจัยที่อาจสร้างการเลือกปฏิบัติ

🎯 **การประยุกต์ใช้:**
- เน้นปัจจัยที่นักเรียนสามารถพัฒนาได้ (คะแนน, ทักษะ)
- หลีกเลี่ยงปัจจัยที่นักเรียนควบคุมไม่ได้ (รายได้ครอบครัว, ไลฟ์สไตล์)
- สนับสนุนหลักการความเท่าเทียมทางการศึกษา

⚖️ **ความชอบธรรม:**
การเลือก features ตามหลักการนี้สามารถป้องกันการตั้งคำถามจากนักวิชาการ
และสอดคล้องกับจริยธรรมทางการศึกษา
        """
        
        return justification.strip()
    
    def create_evaluation_visualization(self, evaluation: Dict) -> plt.Figure:
        """สร้างกราฟแสดงผลการประเมิน"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Category Distribution
        categories = list(evaluation['category_analysis'].keys())
        percentages = [evaluation['category_analysis'][cat]['percentage'] for cat in categories]
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
        
        # กรองออกหมวดหมู่ที่เป็น 0%
        non_zero_data = [(cat, pct, colors[i]) for i, (cat, pct) in enumerate(zip(categories, percentages)) if pct > 0]
        if non_zero_data:
            cats, pcts, cols = zip(*non_zero_data)
            ax1.pie(pcts, labels=cats, autopct='%1.1f%%', colors=cols)
        else:
            ax1.text(0.5, 0.5, 'No Features Selected', ha='center', va='center')
        ax1.set_title('Feature Distribution by Category')
        
        # 2. Category Status
        statuses = [evaluation['category_analysis'][cat]['status'] for cat in categories]
        status_colors = {'GOOD': 'green', 'ACCEPTABLE': 'yellow', 'TOO_LOW': 'orange', 
                        'TOO_HIGH': 'red', 'UNACCEPTABLE': 'darkred'}
        bar_colors = [status_colors.get(status, 'gray') for status in statuses]
        
        ax2.bar(range(len(categories)), percentages, color=bar_colors, alpha=0.7)
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels([cat.replace('_', '\n') for cat in categories], rotation=0, ha='center')
        ax2.set_ylabel('Percentage')
        ax2.set_title('Category Status Evaluation')
        
        # 3. Overall Score Gauge
        score = evaluation['overall_score']
        color = 'green' if score >= 80 else 'yellow' if score >= 60 else 'red'
        ax3.pie([score, 100-score], labels=['Score', 'Remaining'], autopct='%1.1f%%',
               colors=[color, 'lightgray'])
        ax3.set_title(f'Overall Quality Score: {score:.1f}/100')
        
        # 4. Red Flags Summary
        red_flag_counts = {}
        for flag in evaluation['red_flags']:
            severity = flag['severity']
            red_flag_counts[severity] = red_flag_counts.get(severity, 0) + 1
        
        if red_flag_counts:
            severities = list(red_flag_counts.keys())
            counts = list(red_flag_counts.values())
            severity_colors = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'yellow'}
            colors = [severity_colors.get(s, 'gray') for s in severities]
            
            ax4.bar(severities, counts, color=colors)
            ax4.set_title('Red Flags by Severity')
            ax4.set_ylabel('Count')
        else:
            ax4.text(0.5, 0.5, 'No Red Flags Found\n✅ Good!', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=16, color='green', fontweight='bold')
            ax4.set_title('Red Flags Summary')
        
        plt.tight_layout()
        return fig

def evaluate_current_selection():
    """ประเมิน feature selection ปัจจุบัน"""
    
    print("🔍 DOMAIN KNOWLEDGE EVALUATION")
    print("=" * 50)
    
    # ลองโหลดจากไฟล์ผลลัพธ์ ถ้ามี
    result_path = Path("result/feature_selection/selected_features.json")
    
    if result_path.exists():
        print("📁 Loading features from result file...")
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            current_selection = data.get('selected_features', [])
    else:
        print("📄 Using example from report...")
        # Features ที่ถูกเลือกปัจจุบัน (จากผลลัพธ์ที่แสดง)
        current_selection = [
            "คะแนนคณิตศาสตร์", "คะแนนคอมพิวเตอร์", "คะแนนภาษาไทย",
            "คะแนนวิทยาศาสตร์", "คะแนนศิลปะ", "ทักษะการคิดเชิงตรรกะ",
            "ทักษะความคิดสร้างสรรค์", "ความสนใจด้านตัวเลข",
            "ความสนใจด้านเทคโนโลยี", "ความสนใจด้านการทำอาหาร",
            "เพศ", "รายได้ครอบครัว", "การศึกษาของผู้ปกครอง",
            "ความถี่การออกกำลังกาย", "ชอบอ่านหนังสือ"  # 🚨 ปัญหาตรงนี้!
        ]
    
    print(f"📊 Analyzing {len(current_selection)} selected features...")
    
    evaluator = DomainKnowledgeEvaluator()
    evaluation = evaluator.evaluate_feature_selection(current_selection)
    
    print(f"\n📈 Overall Score: {evaluation['overall_score']:.1f}/100")
    print(f"🚩 Red Flags: {len(evaluation['red_flags'])}")
    
    print("\n🚨 RED FLAGS:")
    if evaluation['red_flags']:
        for flag in evaluation['red_flags']:
            print(f"   {flag['message']}")
            print(f"      Severity: {flag['severity']}")
            print(f"      Concern: {flag['academic_concern']}")
    else:
        print("   ✅ No red flags found!")
    
    print("\n💡 RECOMMENDATIONS:")
    for rec in evaluation['recommendations']:
        print(f"   {rec}")
    
    print("\n📋 CATEGORY BREAKDOWN:")
    for category, analysis in evaluation['category_analysis'].items():
        status_emoji = {
            'GOOD': '✅', 'ACCEPTABLE': '⚠️', 'TOO_LOW': '❌', 
            'TOO_HIGH': '🚫', 'UNACCEPTABLE': '🚨'
        }
        emoji = status_emoji.get(analysis['status'], '❓')
        print(f"   {emoji} {category}: {analysis['percentage']:.1f}% ({analysis['count']} features)")
        if analysis['features_selected']:
            print(f"      Features: {analysis['features_selected']}")
    
    print(evaluation['academic_justification'])
    
    # สร้างและบันทึกกราฟ
    try:
        fig = evaluator.create_evaluation_visualization(evaluation)
        fig.savefig("domain_knowledge_evaluation.png", dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved: domain_knowledge_evaluation.png")
        plt.close(fig)
    except Exception as e:
        print(f"\n⚠️ Could not create visualization: {e}")
    
    return evaluation

def test_enhanced_selection():
    """ทดสอบผลลัพธ์ที่ควรจะได้หลังแก้ไข"""
    
    print("\n" + "="*50)
    print("🧪 TESTING ENHANCED SELECTION RESULTS")
    print("="*50)
    
    # ผลลัพธ์ที่คาดหวังหลังใช้ enhanced method
    enhanced_selection = [
        # Core Academic (5/12 = 41.7%)
        "คะแนนคณิตศาสตร์", "คะแนนคอมพิวเตอร์", "คะแนนภาษาไทย",
        "คะแนนวิทยาศาสตร์", "คะแนนศิลปะ",
        
        # Core Skills (2/12 = 16.7%)
        "ทักษะการคิดเชิงตรรกะ", "ทักษะความคิดสร้างสรรค์",
        
        # Core Interests (3/12 = 25%)
        "ความสนใจด้านตัวเลข", "ความสนใจด้านเทคโนโลยี", "ความสนใจด้านการทำอาหาร",
        
        # Demographics - ขั้นต่ำ (2/12 = 16.7%)
        "อายุ", "เพศ"
        
        # ไม่มี Lifestyle หรือ Socioeconomic factors!
    ]
    
    evaluator = DomainKnowledgeEvaluator()
    evaluation = evaluator.evaluate_feature_selection(enhanced_selection)
    
    print(f"🎯 Enhanced Selection Score: {evaluation['overall_score']:.1f}/100")
    print(f"🚩 Red Flags: {len(evaluation['red_flags'])}")
    
    print("\n📊 Enhanced Category Breakdown:")
    for category, analysis in evaluation['category_analysis'].items():
        print(f"   {category}: {analysis['percentage']:.1f}% ({analysis['count']} features)")
    
    if evaluation['overall_score'] >= 80:
        print("\n✅ ENHANCED SELECTION PASSES ACADEMIC STANDARDS!")
    else:
        print(f"\n❌ Still needs improvement (score: {evaluation['overall_score']:.1f}/100)")
    
    return evaluation

if __name__ == "__main__":
    # ประเมินผลลัพธ์ปัจจุบัน
    current_eval = evaluate_current_selection()
    
    # ทดสอบผลลัพธ์ที่คาดหวัง
    enhanced_eval = test_enhanced_selection()
    
    print(f"\n" + "="*50)
    print("📈 COMPARISON SUMMARY")
    print("="*50)
    print(f"Current Score:  {current_eval['overall_score']:.1f}/100")
    print(f"Enhanced Score: {enhanced_eval['overall_score']:.1f}/100")
    print(f"Improvement:    {enhanced_eval['overall_score'] - current_eval['overall_score']:.1f} points")
    
    if enhanced_eval['overall_score'] >= 80:
        print("🎉 Ready for academic presentation!")
    else:
        print("🔧 Still needs domain knowledge refinement")