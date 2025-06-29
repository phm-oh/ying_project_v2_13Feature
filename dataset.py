# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_simplified_student_dataset(n_samples=1789, base_accuracy=0.80, target_after_selection=0.87, seed=42):
    """
    สร้างชุดข้อมูลนักเรียนแบบเรียบง่าย - โฟกัสที่ academic performance
    
    หลักการ: ข้อมูลการเลือกแผนกควรขึ้นอยู่กับ academic performance และ interests เป็นหลัก
    ลบ lifestyle factors และ validation features ออกเพื่อลดความซับซ้อน
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
    
    # ========== DEMOGRAPHIC FEATURES (ข้อมูลทั่วไป - จำกัดเฉพาะที่จำเป็น) ==========
    np.random.seed(seed * 3 + 1000)  # ใช้ seed ต่างออกไป
    age = np.random.randint(16, 20, n_samples)
    gender = np.random.choice([0, 1], n_samples)  # 0: ชาย, 1: หญิง
    
    # สร้าง DataFrame
    data = {
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
    
    df = pd.DataFrame(data)
    
    # ========== สร้างคะแนนความเหมาะสม (ใช้เฉพาะ CORE FEATURES) ==========
    
    # 1. แผนกบัญชี - เน้นคณิต + ตรรกะ + ตัวเลข
    accounting_score = (
        2.0 * df['คะแนนคณิตศาสตร์'] +
        1.8 * df['ทักษะการคิดเชิงตรรกะ'] +
        1.5 * df['ความสนใจด้านตัวเลข'] +
        0.8 * df['คะแนนภาษาไทย']
    )
    
    # 2. แผนกสารสนเทศ - เน้นคอมพิวเตอร์ + เทคโนโลยี + แก้ปัญหา  
    it_score = (
        2.0 * df['คะแนนคอมพิวเตอร์'] +
        1.8 * df['ความสนใจด้านเทคโนโลยี'] +
        1.5 * df['ทักษะการแก้ปัญหา'] +
        1.2 * df['ทักษะการคิดเชิงตรรกะ'] +
        0.8 * df['คะแนนวิทยาศาสตร์']
    )
    
    # 3. แผนกอาหาร - เน้นความคิดสร้างสรรค์ + การทำอาหาร + ศิลปะ
    food_score = (
        2.0 * df['ความสนใจด้านการทำอาหาร'] +
        1.8 * df['ทักษะความคิดสร้างสรรค์'] +
        1.5 * df['คะแนนศิลปะ'] +
        1.0 * df['คะแนนวิทยาศาสตร์'] +
        0.8 * df['คะแนนภาษาไทย']
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
    df['แผนก'] = department
    
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
    print(f"สร้างชุดข้อมูลนักเรียน {n_samples} คน (แบบเรียบง่าย)")
    print(f"การแยกประเภทข้อมูล:")
    print(f"• Core Features: {len(core_features)} features - คะแนน, ทักษะ, ความสนใจ")  
    print(f"• Demographic Features: {len(demographic_features)} features - อายุ, เพศ")
    print(f"จำนวนนักเรียนในแต่ละแผนก:")
    print(df['แผนก'].value_counts())
    
    print(f"\n📊 ผลการทดสอบ:")
    print(f"🔍 ใช้ Features ทั้งหมด ({len(X.columns)} features): {accuracy_all:.3f}")
    print(f"🎯 ใช้ Core Features เท่านั้น ({len(core_features)} features): {accuracy_core:.3f}")
    print(f"✨ การปรับปรุง: {accuracy_core - accuracy_all:.3f}")
    
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
        
        print("\n🏆 Top 10 Most Important Features (Random Forest Analysis):")
        print(feature_imp_all.head(10)[['Feature', 'Importance', 'Type']])
        
        print("\n🎯 Core Features Ranking:")
        print(feature_imp_core.head(11))
        
        # ตรวจสอบว่า Demographic Features ยังติด Top 10 หรือไม่
        top_10 = feature_imp_all.head(10)
        demographic_in_top10 = top_10[top_10['Type'] == 'Demographic']
        if len(demographic_in_top10) > 0:
            print(f"\n📝 หมายเหตุ: พบ Demographic Features ใน Top 10:")
            print(demographic_in_top10[['Feature', 'Importance']])
        else:
            print(f"\n✅ ผลลัพธ์ตามที่คาดหวัง: Core Features ครองตำแหน่งหลัก")
        
        # ตรวจสอบการกระจายของ target
        print(f"\n🔍 การกระจายของแผนกเรียน:")
        dept_dist = df['แผนก'].value_counts(normalize=True) * 100
        for dept, pct in dept_dist.items():
            print(f"   {dept}: {pct:.1f}%")
    
    return df, accuracy_all, accuracy_core, feature_imp_all, feature_imp_core, core_features, demographic_features

if __name__ == "__main__":
    print("=== สร้างชุดข้อมูลนักเรียนแบบเรียบง่าย ===")
    
    df, acc_all, acc_core, feat_imp_all, feat_imp_core, core_feat, demo_feat = generate_simplified_student_dataset()
    
    # บันทึกไฟล์
    df.to_csv('student_realistic_data.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 บันทึกไฟล์ student_realistic_data.csv เรียบร้อยแล้ว")
    print(f"📝 Total Features: {len(df.columns)-1}")
    print(f"📊 Total Rows: {len(df)}")
    print(f"\n🎯 Dataset เรียบง่าย โฟกัสที่ academic performance")
    print(f"✅ พร้อมใช้งานสำหรับ Machine Learning Pipeline")