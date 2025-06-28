# ไฟล์: __init__.py
# Path: src/__init__.py
# วัตถุประสงค์: ทำให้ src เป็น Python package (แก้แล้ว - ลบ imports เพื่อหลีกเลี่ยง circular import)

"""
src package - Feature Selection และ Model Comparison Pipeline
"""

__version__ = "1.0.0"
__author__ = "ML Pipeline Team"
__description__ = "Feature Selection และ Model Comparison สำหรับระบบแนะนำแผนกเรียน"

# ไม่ import modules เพื่อหลีกเลี่ยง circular import
# ให้ import ใน main.py แทน