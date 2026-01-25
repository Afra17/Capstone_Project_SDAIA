import os
import re
import json
from agentic_doc.parse import parse
from unstructured.partition.md import partition_md
from unstructured.cleaners.core import clean_extra_whitespace

class DocumentProcessor:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def process_pdf_to_clean_md(self, pdf_path):
        """
        المهمة: تحويل PDF إلى MD ثم تنظيفه وإصلاح العناوين في خطوة واحدة.
        """
        file_base_name = os.path.basename(pdf_path).replace(".pdf", "")
        raw_md_path = os.path.join(self.output_folder, f"{file_base_name}_raw.md")
        final_md_path = os.path.join(self.output_folder, f"{file_base_name}_final_clean.md")

        # 1. المرحلة الأولى: القراءة والتحويل (Parsing)
        print(f"🔄 جاري تحويل {file_base_name} من PDF إلى Markdown...")
        results = parse([pdf_path])
        raw_content = results[0].markdown

        with open(raw_md_path, "w", encoding="utf-8") as f:
            f.write(raw_content)

        # 2. المرحلة الثانية: التنظيف الهيكلي (Cleaning & Repairing)
        print(f"🧹 جاري تنظيف النص وإصلاح العناوين (بما في ذلك عنوان 64)...")
        cleaned_content = self._auto_clean_logic(raw_md_path)

        with open(final_md_path, "w", encoding="utf-8") as f:
            f.write(cleaned_content)

        print(f"✅ اكتملت المعالجة! الملف النظيف: {final_md_path}")
        return final_md_path

    def _auto_clean_logic(self, file_path):
        """منطق التنظيف العميق واستعادة العناوين"""
        elements = partition_md(filename=file_path)
        cleaned_text = []

        # كلمات مستبعدة (الصور والترويسات)
        image_keywords = [
            "Summary :", "logo:", "Visible Elements :", "Analysis :", 
            "Graphic Elements :", "Design & Layout :", "/1446", "/25"
        ]
        excluded_headers = [
            "المملكة العربية السعودية", "كراسة الشروط", "tenders.etimad.sa", 
            "المعتمد بموجب قرار وزير المالية", "اسم المنافسة:", "رقم الكراسة:"
        ]

        for element in elements:
            text = element.text.strip()
            
            # الفلترة بناءً على الكلمات المستبعدة
            if any(text.startswith(key) for key in image_keywords): continue 
            if any(key in text for key in excluded_headers): continue

            # تنظيف الروابط وأرقام الصفحات
            text = re.sub(r'\d+/\d+', '', text)
            text = re.sub(r'https?://\S+', '', text)
            text = clean_extra_whitespace(text)

            if text.strip():
                # --- تحسين التعرف على العناوين (Fixing Heading 64 and others) ---
                # النمط الجديد: يبحث عن رقم في بداية السطر يتبعه (شرطة أو نقطة أو مسافة)
                # مثال: "64 عنوان" أو "64 - عنوان" أو "64. عنوان"
                heading_pattern = r'^(\d+[\s\.\-].*|^القسم\s+.*|^المادة\s+.*)'
                
                if re.match(heading_pattern, text):
                    # إذا لم يكن السطر يحمل علامة العنوان #، نضيفها له
                    if not text.startswith('#'):
                        text = f"## {text}"
                
                cleaned_text.append(text)

        return "\n\n".join(cleaned_text)

# --- مثال على طريقة الاستخدام في ملفك الرئيسي ---
if __name__ == "__main__":
    PDF_INPUT = r"C:\Users\user\...\data\request1.pdf"
    OUTPUT_DIR = r"C:\Users\user\...\data\processed"
    
    processor = DocumentProcessor(OUTPUT_DIR)
    final_file = processor.process_pdf_to_clean_md(PDF_INPUT)