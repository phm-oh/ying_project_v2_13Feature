# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def convert_score_to_exam_format(scores, min_score=30, max_score=100, exam_points=4, min_correct=8, max_correct=25):
    """
    แปลงคะแนนเดิม (30.0-100.0) ให้เป็นคะแนนจากข้อสอบ (32, 36, 40, ..., 100)
    รักษาการกระจายและความสัมพันธ์เดิมไว้
    """
    # แปลงเป็น percentile (0-1)
    percentiles = (scores - min_score) / (max_score - min_score)
    
    # แมปไปยังจำนวนข้อที่ทำถูก (8-25 ข้อ)
    correct_answers = np.round(percentiles * (max_correct - min_correct) + min_correct).astype(int)
    
    # แปลงเป็นคะแนนขั้นสุดท้าย
    exam_scores = correct_answers * exam_points
    
    return exam_scores

def convert_skill_to_exam_format(skills, min_skill=3, max_skill=10, original_min=3, original_max=10):
    """
    แปลงทักษะ (3.0-10.0) ให้เป็นจำนวนเต็ม (3, 4, 5, ..., 10)
    """
    # แปลงเป็น percentile (0-1)
    percentiles = (skills - original_min) / (original_max - original_min)
    
    # แมปไปยังคะแนนทักษะ (3-10)
    skill_scores = np.round(percentiles * (max_skill - min_skill) + min_skill).astype(int)
    
    return skill_scores

def convert_interest_to_exam_format(interests, original_min=1, original_max=10):
    """
    แปลงความสนใจ (1.0-10.0) ให้เป็นระดับ Likert Scale (1, 3, 7, 10)
    """
    # แปลงเป็น percentile (0-1)
    percentiles = (interests - original_min) / (original_max - original_min)
    
    # แมปไปยังระดับ Likert
    # 0-0.25 -> 1, 0.25-0.5 -> 3, 0.5-0.75 -> 7, 0.75-1.0 -> 10
    likert_scores = np.zeros_like(interests, dtype=int)
    likert_scores[percentiles <= 0.25] = 1
    likert_scores[(percentiles > 0.25) & (percentiles <= 0.5)] = 3
    likert_scores[(percentiles > 0.5) & (percentiles <= 0.75)] = 7
    likert_scores[percentiles > 0.75] = 10
    
    return likert_scores

def generate_simplified_student_dataset(n_samples=1789, base_accuracy=0.80, target_after_selection=0.87, seed=42):
    """
    สร้างชุดข้อมูลนักเรียนแบบเรียบง่าย - โฟกัสที่ academic performance
    แล้วแปลงคะแนนให้สอดคล้องกับข้อสอบจริง (รักษาความสัมพันธ์เดิม)
    
    หลักการ: ข้อมูลการเลือกแผนกควรขึ้นอยู่กับ academic performance และ interests เป็นหลัก
    ลบ lifestyle factors และ validation features ออกเพื่อลดความซับซ้อน
    """
    
    np.random.seed(seed)
    departments = ['บัญชี', 'สารสนเทศ', 'อาหาร']
    
    # ========== CORE FEATURES (ใช้วิธีเดิม) ==========
    # คะแนนวิชาหลัก (1-100) - ยังใช้วิธีเดิม
    math_score = np.round(np.random.uniform(30, 100, n_samples), 1)
    computer_score = np.round(np.random.uniform(30, 100, n_samples), 1)  
    language_score = np.round(np.random.uniform(30, 100, n_samples), 1)
    science_score = np.round(np.random.uniform(30, 100, n_samples), 1)
    art_score = np.round(np.random.uniform(30, 100, n_samples), 1)
    
    # ทักษะหลัก (1-10 scale) - ยังใช้วิธีเดิม
    logical_thinking = np.round(np.random.uniform(3, 10, n_samples), 1)
    creativity_skill = np.round(np.random.uniform(3, 10, n_samples), 1)
    problem_solving = np.round(np.random.uniform(3, 10, n_samples), 1)
    
    # ความสนใจหลัก (1-10 scale) - ยังใช้วิธีเดิม
    interest_numbers = np.round(np.random.uniform(1, 10, n_samples), 1)
    interest_technology = np.round(np.random.uniform(1, 10, n_samples), 1)
    interest_cooking = np.round(np.random.uniform(1, 10, n_samples), 1)
    
    # ========== DEMOGRAPHIC FEATURES (ใช้วิธีเดิม) ==========
    np.random.seed(seed * 3 + 1000)  # ใช้ seed ต่างออกไป
    age = np.random.randint(16, 20, n_samples)
    gender = np.random.choice([0, 1], n_samples)  # 0: ชาย, 1: หญิง
    
    # สร้าง DataFrame ด้วยข้อมูลเดิม
    data_original = {
        # === CORE FEATURES ===
        'คะแนนคณิตศาสตร์': math_score,
        'คะแนนคอมพิวเตอร์': computer_score,
        'คะแนนภาษาไทย': language_score,
        'คะแนนวิทยาศาสตร์': science_score,
        'คะแนนศิลปะ': art_score,
        'ทักษะการคิดเชิงตรรกะ': logical_thinking,
        'ทักษะความคิดสร้างสรรค์': creativity_skill,
        'ทักษะการแก้ปัญหา': problem_solving,
        'ความสนใจด้านตัวเลข': interest_numbers,
        'ความสนใจด้านเทคโนโลยี': interest_technology,
        'ความสนใจด้านการทำอาหาร': interest_cooking,
        
        # === DEMOGRAPHIC FEATURES ===
        'อายุ': age,
        'เพศ': gender
    }
    
    df_original = pd.DataFrame(data_original)
    
    # ========== สร้างคะแนนความเหมาะสมด้วยข้อมูลเดิม ==========
    
    # 1. แผนกบัญชี - เน้นคณิต + ตรรกะ + ตัวเลข
    accounting_score = (
        2.0 * df_original['คะแนนคณิตศาสตร์'] +
        1.8 * df_original['ทักษะการคิดเชิงตรรกะ'] +
        1.5 * df_original['ความสนใจด้านตัวเลข'] +
        0.8 * df_original['คะแนนภาษาไทย']
    )
    
    # 2. แผนกสารสนเทศ - เน้นคอมพิวเตอร์ + เทคโนโลยี + แก้ปัญหา  
    it_score = (
        2.0 * df_original['คะแนนคอมพิวเตอร์'] +
        1.8 * df_original['ความสนใจด้านเทคโนโลยี'] +
        1.5 * df_original['ทักษะการแก้ปัญหา'] +
        1.2 * df_original['ทักษะการคิดเชิงตรรกะ'] +
        0.8 * df_original['คะแนนวิทยาศาสตร์']
    )
    
    # 3. แผนกอาหาร - เน้นความคิดสร้างสรรค์ + การทำอาหาร + ศิลปะ
    food_score = (
        2.0 * df_original['ความสนใจด้านการทำอาหาร'] +
        1.8 * df_original['ทักษะความคิดสร้างสรรค์'] +
        1.5 * df_original['คะแนนศิลปะ'] +
        1.0 * df_original['คะแนนวิทยาศาสตร์'] +
        0.8 * df_original['คะแนนภาษาไทย']
    )
    
    # เพิ่ม noise เพื่อความสมจริง
    noise_strength = (1.0 - base_accuracy) * 8
    accounting_score += np.random.normal(0, noise_strength, n_samples)
    it_score += np.random.normal(0, noise_strength, n_samples)
    food_score += np.random.normal(0, noise_strength, n_samples)
    
    # กำหนดแผนกตามคะแนนสูงสุด
    scores = pd.DataFrame({
        'บัญชี': accounting_score,
        'สารสนเทศ': it_score,
        'อาหาร': food_score
    })
    
    department = scores.idxmax(axis=1)
    
    # ========== แปลงคะแนนให้สอดคล้องกับข้อสอบ ==========
    
    # แปลงคะแนนวิชา (30.0-100.0) -> (32, 36, 40, ..., 100)
    math_exam = convert_score_to_exam_format(math_score)
    computer_exam = convert_score_to_exam_format(computer_score)
    language_exam = convert_score_to_exam_format(language_score)
    science_exam = convert_score_to_exam_format(science_score)
    art_exam = convert_score_to_exam_format(art_score)
    
    # แปลงทักษะ (3.0-10.0) -> (3, 4, 5, ..., 10)
    logical_exam = convert_skill_to_exam_format(logical_thinking)
    creativity_exam = convert_skill_to_exam_format(creativity_skill)
    problem_exam = convert_skill_to_exam_format(problem_solving)
    
    # แปลงความสนใจ (1.0-10.0) -> (1, 3, 7, 10)
    numbers_exam = convert_interest_to_exam_format(interest_numbers)
    technology_exam = convert_interest_to_exam_format(interest_technology)
    cooking_exam = convert_interest_to_exam_format(interest_cooking)
    
    # สร้าง DataFrame ขั้นสุดท้าย
    data_final = {
        # === CORE FEATURES (แปลงแล้ว) ===
        'คะแนนคณิตศาสตร์': math_exam,
        'คะแนนคอมพิวเตอร์': computer_exam,
        'คะแนนภาษาไทย': language_exam,
        'คะแนนวิทยาศาสตร์': science_exam,
        'คะแนนศิลปะ': art_exam,
        'ทักษะการคิดเชิงตรรกะ': logical_exam,
        'ทักษะความคิดสร้างสรรค์': creativity_exam,
        'ทักษะการแก้ปัญหา': problem_exam,
        'ความสนใจด้านตัวเลข': numbers_exam,
        'ความสนใจด้านเทคโนโลยี': technology_exam,
        'ความสนใจด้านการทำอาหาร': cooking_exam,
        
        # === DEMOGRAPHIC FEATURES (เหมือนเดิม) ===
        'อายุ': age,
        'เพศ': gender,
        
        # === TARGET (เหมือนเดิม) ===
        'แผนก': department
    }
    
    df = pd.DataFrame(data_final)
    
    # ========== ทดสอบประสิทธิภาพ ==========
    
    # แบ่งกลุ่ม features สำหรับทดสอบ
    core_features = [
        'คะแนนคณิตศาสตร์', 'คะแนนคอมพิวเตอร์', 'คะแนนภาษาไทย', 'คะแนนวิทยาศาสตร์', 'คะแนนศิลปะ',
        'ทักษะการคิดเชิงตรรกะ', 'ทักษะความคิดสร้างสรรค์', 'ทักษะการแก้ปัญหา',
        'ความสนใจด้านตัวเลข', 'ความสนใจด้านเทคโนโลยี', 'ความสนใจด้านการทำอาหาร'
    ]
    
    demographic_features = [
        'อายุ', 'เพศ'
    ]
    
    X = df.drop('แผนก', axis=1)
    y = df['แผนก']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    
    # ทดสอบกับ features ทั้งหมด
    model_all = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
    model_all.fit(X_train, y_train)
    y_pred_all = model_all.predict(X_test)
    accuracy_all = accuracy_score(y_test, y_pred_all)
    
    # ทดสอบกับ core features เท่านั้น
    X_core = df[core_features]
    X_train_core, X_test_core, _, _ = train_test_split(X_core, y, test_size=0.3, random_state=seed, stratify=y)
    
    model_core = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
    model_core.fit(X_train_core, y_train)
    y_pred_core = model_core.predict(X_test_core)
    accuracy_core = accuracy_score(y_test, y_pred_core)
    
    # แสดงผลลัพธ์
    print(f"สร้างชุดข้อมูลนักเรียน {n_samples} คน (แปลงให้สอดคล้องกับข้อสอบ)")
    print(f"📝 ระบบคะแนนหลังการแปลง:")
    print(f"   • วิชาหลัก: {sorted(df['คะแนนคณิตศาสตร์'].unique())}")
    print(f"   • ทักษะ: {sorted(df['ทักษะการคิดเชิงตรรกะ'].unique())}")  
    print(f"   • ความสนใจ: {sorted(df['ความสนใจด้านตัวเลข'].unique())}")
    
    print(f"\nจำนวนนักเรียนในแต่ละแผนก:")
    print(df['แผนก'].value_counts())
    
    print(f"\n📊 ผลการทดสอบ:")
    print(f"🔍 ใช้ Features ทั้งหมด ({len(X.columns)} features): {accuracy_all:.3f}")
    print(f"🎯 ใช้ Core Features เท่านั้น ({len(core_features)} features): {accuracy_core:.3f}")
    print(f"✨ ความแตกต่าง: {accuracy_core - accuracy_all:.3f}")
    
    # ตรวจสอบความสัมพันธ์
    print(f"\n🔍 ตรวจสอบความสัมพันธ์หลังการแปลง:")
    
    # คำนวณค่าเฉลี่ยตามแผนก
    dept_stats = df.groupby('แผนก')[['คะแนนคณิตศาสตร์', 'คะแนนคอมพิวเตอร์', 'คะแนนศิลปะ', 
                                     'ความสนใจด้านตัวเลข', 'ความสนใจด้านเทคโนโลยี', 'ความสนใจด้านการทำอาหาร']].mean()
    
    print("คะแนนเฉลี่ยแต่ละแผนก:")
    for dept in departments:
        stats = dept_stats.loc[dept]
        print(f"   {dept}: คณิต={stats['คะแนนคณิตศาสตร์']:.1f}, คอม={stats['คะแนนคอมพิวเตอร์']:.1f}, ศิลปะ={stats['คะแนนศิลปะ']:.1f}")
        print(f"           ตัวเลข={stats['ความสนใจด้านตัวเลข']:.1f}, เทคโนโลยี={stats['ความสนใจด้านเทคโนโลยี']:.1f}, ทำอาหาร={stats['ความสนใจด้านการทำอาหาร']:.1f}")
    
    return df, accuracy_all, accuracy_core, None, None, core_features, demographic_features

if __name__ == "__main__":
    print("=== สร้างชุดข้อมูลนักเรียนที่สอดคล้องกับข้อสอบ ===")
    print("🔄 รักษาความสัมพันธ์และการกระจายเดิม แต่แปลงคะแนนให้ตรงกับข้อสอบ")
    
    df, acc_all, acc_core, feat_imp_all, feat_imp_core, core_feat, demo_feat = generate_simplified_student_dataset()
    
    # บันทึกไฟล์
    df.to_csv('student_realistic_data.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 บันทึกไฟล์ student_realistic_data.csv เรียบร้อยแล้ว")
    print(f"📝 Total Features: {len(df.columns)-1}")
    print(f"📊 Total Rows: {len(df)}")
    print(f"\n🎯 คะแนนสอดคล้องกับข้อสอบ + รักษาความสัมพันธ์เดิม")
    print(f"✅ พร้อมใช้งานสำหรับระบบทำนายแผนกการเรียน")