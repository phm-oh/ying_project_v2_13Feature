# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_realistic_student_dataset(n_samples=2000, base_accuracy=0.80, target_after_selection=0.87, seed=42):
    """
    สร้างชุดข้อมูลนักเรียนแบบสมจริง - Features ที่อธิบายได้ว่าทำไมถึงเก็บ
    """
    
    np.random.seed(seed)
    departments = ['บัญชี', 'สารสนเทศ', 'อาหาร']
    
    # ========== CORE FEATURES (คะแนนจากแบบทดสอบ + ทักษะหลัก) ==========
    # คะแนนวิชาหลัก (1-100)
    math_score = np.round(np.random.uniform(30, 100, n_samples), 1)
    computer_score = np.round(np.random.uniform(30, 100, n_samples), 1)  
    language_score = np.round(np.random.uniform(30, 100, n_samples), 1)
    science_score = np.round(np.random.uniform(30, 100, n_samples), 1)
    art_score = np.round(np.random.uniform(30, 100, n_samples), 1)
    
    # ทักษะหลัก (1-10 scale)
    logical_thinking = np.round(np.random.uniform(3, 10, n_samples), 1)
    creativity_skill = np.round(np.random.uniform(3, 10, n_samples), 1)
    problem_solving = np.round(np.random.uniform(3, 10, n_samples), 1)
    
    # ความสนใจหลัก (1-10 scale)
    interest_numbers = np.round(np.random.uniform(1, 10, n_samples), 1)
    interest_technology = np.round(np.random.uniform(1, 10, n_samples), 1)
    interest_cooking = np.round(np.random.uniform(1, 10, n_samples), 1)
    
    # ========== DEMOGRAPHIC FEATURES (ข้อมูลทั่วไป - สมเหตุสมผลแต่เกี่ยวข้องน้อย) ==========
    age = np.random.randint(16, 20, n_samples)
    gender = np.random.choice([0, 1], n_samples)  # 0: ชาย, 1: หญิง
    family_income = np.random.choice([1, 2, 3, 4, 5], n_samples)  # 1=ต่ำ, 5=สูง
    parent_education = np.random.choice([1, 2, 3, 4], n_samples)  # 1=ประถม, 4=ปริญญาตรี+
    siblings_count = np.random.randint(0, 4, n_samples)  # จำนวนพี่น้อง
    
    # ========== LIFESTYLE FEATURES (วิถีชีวิต - เกี่ยวข้องน้อยมาก) ==========
    sleep_hours = np.round(np.random.uniform(5, 10, n_samples), 1)  # ชั่วโมงนอน
    exercise_frequency = np.random.randint(0, 7, n_samples)  # วันต่อสัปดาห์
    social_media_hours = np.round(np.random.uniform(0, 8, n_samples), 1)  # ชั่วโมงต่อวัน
    reading_hobby = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # ชอบอ่านหนังสือ
    music_preference = np.random.choice([1, 2, 3, 4], n_samples)  # 1=Pop, 2=Rock, 3=Classical, 4=Country
    
    # สร้าง DataFrame
    data = {
        # === CORE FEATURES (ความสัมพันธ์สูง) ===
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
        
        # === DEMOGRAPHIC FEATURES (ความสัมพันธ์ปานกลาง) ===
        'อายุ': age,
        'เพศ': gender,
        'รายได้ครอบครัว': family_income,
        'การศึกษาของผู้ปกครอง': parent_education,
        'จำนวนพี่น้อง': siblings_count,
        
        # === LIFESTYLE FEATURES (ความสัมพันธ์น้อย) ===
        'ชั่วโมงการนอน': sleep_hours,
        'ความถี่การออกกำลังกาย': exercise_frequency,
        'ชั่วโมงใช้โซเชียลมีเดีย': social_media_hours,
        'ชอบอ่านหนังสือ': reading_hobby,
        'ประเภทเพลงที่ชอบ': music_preference,
    }
    
    df = pd.DataFrame(data)
    
    # ========== สร้างคะแนนความเหมาะสม (ใช้เฉพาะ CORE FEATURES) ==========
    
    # 1. แผนกบัญชี - เน้นคณิต + ตรรกะ + ตัวเลข
    accounting_score = (
        1.5 * df['คะแนนคณิตศาสตร์'] + 
        1.2 * df['ทักษะการคิดเชิงตรรกะ'] + 
        1.0 * df['ความสนใจด้านตัวเลข'] +
        0.5 * df['คะแนนภาษาไทย']  # ทักษะการสื่อสาร
    )
    
    # 2. แผนกสารสนเทศ - เน้นคอมพิวเตอร์ + เทคโนโลยี + แก้ปัญหา  
    it_score = (
        1.5 * df['คะแนนคอมพิวเตอร์'] +
        1.3 * df['ความสนใจด้านเทคโนโลยี'] +
        1.0 * df['ทักษะการแก้ปัญหา'] +
        0.8 * df['ทักษะการคิดเชิงตรรกะ'] +
        0.5 * df['คะแนนวิทยาศาสตร์']
    )
    
    # 3. แผนกอาหาร - เน้นความคิดสร้างสรรค์ + การทำอาหาร + ศิลปะ
    food_score = (
        1.5 * df['ความสนใจด้านการทำอาหาร'] +
        1.3 * df['ทักษะความคิดสร้างสรรค์'] +
        1.0 * df['คะแนนศิลปะ'] +
        0.7 * df['คะแนนวิทยาศาสตร์'] +  # เคมีอาหาร
        0.5 * df['คะแนนภาษาไทย']  # การสื่อสาร
    )
    
    # เพิ่ม noise เพื่อความสมจริง
    noise_strength = (1.0 - base_accuracy) * 15
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
    df['แผนก'] = department
    
    # ========== เพิ่มข้อมูลที่ขาดไป (สำหรับ Validation) ==========
    
    # สร้างชั้นปี (ปวช.1-3)
    study_year = np.random.choice([1, 2, 3], n_samples, p=[0.4, 0.35, 0.25])
    df['ชั้นปี'] = study_year
    
    # สร้าง GPA ที่มีความสัมพันธ์กับความเหมาะสมของแผนก
    max_scores = scores.max(axis=1)
    second_max_scores = scores.apply(lambda x: x.nlargest(2).iloc[1], axis=1)
    decision_margin = max_scores - second_max_scores
    
    base_gpa = 2.5 + (decision_margin / decision_margin.max()) * 1.5
    year_effect = (4 - study_year) * 0.1
    random_noise = np.random.normal(0, 0.3, n_samples)
    
    gpa = np.clip(base_gpa + year_effect + random_noise, 1.0, 4.0)
    gpa = np.round(gpa, 2)
    df['GPA'] = gpa
    
    # สร้างความพึงพอใจ
    satisfaction_base = 3.0
    satisfaction_from_gpa = (gpa - 2.5) * 0.8
    satisfaction_from_margin = (decision_margin / decision_margin.max()) * 1.5
    random_satisfaction_noise = np.random.normal(0, 0.4, n_samples)
    
    satisfaction = satisfaction_base + satisfaction_from_gpa + satisfaction_from_margin + random_satisfaction_noise
    satisfaction = np.clip(satisfaction, 1, 5)
    satisfaction = np.round(satisfaction)
    df['ความพึงพอใจในแผนก'] = satisfaction.astype(int)
    
    # สร้างคำตอบ "หากเลือกใหม่จะเลือกแผนกเดิมไหม"
    probability_same_choice = (satisfaction - 1) / 4 * 0.7 + (gpa - 1) / 3 * 0.3
    choice_prob = np.random.random(n_samples)
    choice_again = np.where(choice_prob < probability_same_choice, 
                           np.random.choice([1, 2], n_samples, p=[0.6, 0.4]),
                           np.random.choice([3, 4, 5], n_samples, p=[0.3, 0.4, 0.3]))
    
    df['หากเลือกใหม่จะเลือกแผนกเดิมไหม'] = choice_again
    
    # ========== ทดสอบประสิทธิภาพ ==========
    
    # แบ่งกลุ่ม features สำหรับทดสอบ
    core_features = [
        'คะแนนคณิตศาสตร์', 'คะแนนคอมพิวเตอร์', 'คะแนนภาษาไทย', 'คะแนนวิทยาศาสตร์', 'คะแนนศิลปะ',
        'ทักษะการคิดเชิงตรรกะ', 'ทักษะความคิดสร้างสรรค์', 'ทักษะการแก้ปัญหา',
        'ความสนใจด้านตัวเลข', 'ความสนใจด้านเทคโนโลยี', 'ความสนใจด้านการทำอาหาร'
    ]
    
    demographic_features = [
        'อายุ', 'เพศ', 'รายได้ครอบครัว', 'การศึกษาของผู้ปกครอง', 'จำนวนพี่น้อง'
    ]
    
    lifestyle_features = [
        'ชั่วโมงการนอน', 'ความถี่การออกกำลังกาย', 'ชั่วโมงใช้โซเชียลมีเดีย', 'ชอบอ่านหนังสือ', 'ประเภทเพลงที่ชอบ'
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
    print(f"สร้างชุดข้อมูลนักเรียน {n_samples} คน")
    print(f"จำนวนนักเรียนในแต่ละแผนก:")
    print(df['แผนก'].value_counts())
    print(f"จำนวนนักเรียนในแต่ละชั้นปี:")
    print(df['ชั้นปี'].value_counts())
    
    print(f"\n📊 ผลการทดสอบ:")
    print(f"🔍 ใช้ Features ทั้งหมด ({len(X.columns)} features): {accuracy_all:.3f}")
    print(f"🎯 ใช้ Core Features เท่านั้น ({len(core_features)} features): {accuracy_core:.3f}")
    print(f"✨ การปรับปรุง: {accuracy_core - accuracy_all:.3f}")
    
    # วิเคราะห์ GPA และความพึงพอใจ
    print(f"\n📈 การวิเคราะห์ GPA:")
    print(f"📊 GPA เฉลี่ย: {df['GPA'].mean():.2f} (SD: {df['GPA'].std():.2f})")
    
    high_gpa = df[df['GPA'] >= 3.5]
    mid_gpa = df[(df['GPA'] >= 2.5) & (df['GPA'] < 3.5)]
    low_gpa = df[df['GPA'] < 2.5]
    
    print(f"🟢 GPA สูง (≥3.5): {len(high_gpa)} คน ({len(high_gpa)/len(df)*100:.1f}%)")
    print(f"🟡 GPA ปานกลาง (2.5-3.5): {len(mid_gpa)} คน ({len(mid_gpa)/len(df)*100:.1f}%)")
    print(f"🔴 GPA ต่ำ (<2.5): {len(low_gpa)} คน ({len(low_gpa)/len(df)*100:.1f}%)")
    
    print(f"\n😊 การวิเคราะห์ความพึงพอใจ:")
    satisfaction_counts = df['ความพึงพอใจในแผนก'].value_counts().sort_index()
    for score, count in satisfaction_counts.items():
        emoji = ['😡', '😞', '😐', '😊', '😍'][score-1]
        print(f"{emoji} ระดับ {score}: {count} คน ({count/len(df)*100:.1f}%)")
    
    print(f"\n🔄 การวิเคราะห์ 'หากเลือกใหม่':")
    choice_labels = ['เลือกเดิมแน่นอน', 'อาจเลือกเดิม', 'ไม่แน่ใจ', 'อาจเลือกอื่น', 'เลือกอื่นแน่นอน']
    choice_counts = df['หากเลือกใหม่จะเลือกแผนกเดิมไหม'].value_counts().sort_index()
    for choice, count in choice_counts.items():
        print(f"  {choice}. {choice_labels[choice-1]}: {count} คน ({count/len(df)*100:.1f}%)")
    
    print(f"\n📋 สถิติสรุป:")
    print(f"🎯 ความพึงพอใจเฉลี่ย: {df['ความพึงพอใจในแผนก'].mean():.2f}/5")
    print(f"🔄 % ที่จะเลือกแผนกเดิม (คะแนน 1-2): {((df['หากเลือกใหม่จะเลือกแผนกเดิมไหม'] <= 2).sum() / len(df) * 100):.1f}%")
    print(f"💔 % ที่จะเลือกแผนกอื่น (คะแนน 4-5): {((df['หากเลือกใหม่จะเลือกแผนกเดิมไหม'] >= 4).sum() / len(df) * 100):.1f}%")
    
    # แสดงความสำคัญของ features
    feature_imp_all = None
    feature_imp_core = None
    
    if hasattr(model_all, 'feature_importances_'):
        # สร้างรายการประเภท features
        feature_types = []
        for feature in X.columns:
            if feature in core_features:
                feature_types.append('Core')
            elif feature in demographic_features:
                feature_types.append('Demographic')  
            elif feature in lifestyle_features:
                feature_types.append('Lifestyle')
            else:
                feature_types.append('Additional')
        
        feature_imp_all = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model_all.feature_importances_,
            'Type': feature_types
        }).sort_values('Importance', ascending=False)
        
        feature_imp_core = pd.DataFrame({
            'Feature': core_features,
            'Importance': model_core.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n🏆 Top 10 Most Important Features (All):")
        print(feature_imp_all.head(10)[['Feature', 'Importance', 'Type']])
        
        print("\n🎯 Core Features Importance:")
        print(feature_imp_core.head())
    
    return df, accuracy_all, accuracy_core, feature_imp_all, feature_imp_core, core_features, demographic_features, lifestyle_features

if __name__ == "__main__":
    print("=== สร้างชุดข้อมูลแบบสมจริง ===")
    
    df, acc_all, acc_core, feat_imp_all, feat_imp_core, core_feat, demo_feat, life_feat = generate_realistic_student_dataset(
        n_samples=2000,
        base_accuracy=0.80,
        target_after_selection=0.87,
        seed=42
    )
    
    # บันทึกไฟล์
    df.to_csv('student_realistic_data.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 บันทึกไฟล์ student_realistic_data.csv เรียบร้อยแล้ว")
    print(f"📝 Total Features: {len(df.columns)-1}")
    print(f"📊 Total Rows: {len(df)}")