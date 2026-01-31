import re
import os
import json
from unstructured.partition.md import partition_md
from unstructured.cleaners.core import clean_extra_whitespace
from dotenv import load_dotenv
import os 
from openai import OpenAI 
from agentic_doc.parse import parse


class DynamicDocumentProcessor:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        # 1. كلمات ثابتة للأوصاف البصرية (مستخلصة من تجاربك)


        self.static_image_noise = [
            "Visual Elements", "Technical Details", "Logo Elements",
            "Design & Placement", "Graphic Elements", "Spatial Relationships",
            "Analysis :", "Summary :", "Logo uses", "Stylized star",
            "blue and green lines", "central arrow", "wavy circular shape",
            "logo: No visible company name", "logo:", "Placement & Dimensions :", 
            "Text Elements :", "Design & Layout :", "Layout :","Design Elements :","Design Details :", 
            "Text Fields :", "Colour Palette :", "Spatial Relationships :","Dimensions & Placement :",
            "Design & Colour : ","Primary colour:","The use of blue and green", 
            "bilingual text indicates","national symbolism","Surrounding Outline :","Layout & Placement :"
                
            
    ]
        self.static_blacklist = [
                "كراسة الشروط والمواصفات",
                "نموذج كراسة الشروط",
                "المعتمد بموجب قرار وزير المالية",
                "رقم النسخة : الأولى",
                "تاريخ الإصدار",
                "المملكة العربية السعودية",
                "اسم الإدارة :",
                "اسم النموذج :",
                "شركة تمكين للتقنيات",
                "تاريخ طرح الكراسة:",
                
            ]

        

    def get_cleaning_rules(self, raw_md_text, client):
        """الآن نطلب من الموديل التركيز فقط على الهيدرز الخاصة بهذا الملف"""
        sample = raw_md_text[:3000]
        
        prompt = f"""
        Analyze this RFP sample and identify recurring 'Document Noise' specific to this file:
        - Specific headers/footers (e.g., Department names, Tender numbers).
        - Document metadata that repeats on every page.
        
        Return ONLY a JSON with: "excluded_headers" (list).
        Sample: {sample}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        
        rules = json.loads(response.choices[0].message.content)
        print(f"🎯 قواعد التنظيف الخاصة بهذا الملف: {rules}")
        return rules

    def clean_document(self, file_path, rules):
        elements = partition_md(filename=file_path)
        cleaned_text = []
        dynamic_hdr_keys = rules.get('excluded_headers', [])

        for element in elements:
            text = element.text.strip()
            if not text: continue
            
            # --- [فلتر 1] الروابط والتواريخ والوقت (Regex) ---
            # يحذف الروابط (etimad.sa) والوقت (4:29 PM) والتواريخ المكسورة (/25)
            if re.search(r'https?://\S+|(\d{1,2}:\d{2}\s?(AM|PM))|(/\d{2})', text):
                continue

            # --- [فلتر 2] القائمة السوداء الثابتة ---
            if any(trash in text for trash in self.static_blacklist):
                continue
            
            # --- [فلتر 3] أوصاف الصور (image_keywords) ---
            if any(noise.lower() in text.lower() for noise in self.static_image_noise):
                continue
            
            # --- [فلتر 4] الهيدرز الديناميكية من الـ LLM ---
            if any(key in text for key in dynamic_hdr_keys):
                continue

            # تنظيف المسافات الزائدة
            text = clean_extra_whitespace(text)

            # استعادة العناوين (مع الحفاظ على اسم المنافسة ورقم الكراسة لأنها معلومات هامة)
            if text:
                heading_pattern = r'^(\d+[\s\.\-].*|^المادة\s+.*|^البند\s+.*)'
                if re.match(heading_pattern, text) and not text.startswith('#'):
                    text = f"## {text}"
                cleaned_text.append(text)

        return "\n\n".join(cleaned_text)


def run_full_cleaning_pipeline(pdf_input_path: str):
    """
    تأخذ مسار الـ PDF وتعيد مسار ملف الـ Markdown المنظف نهائياً.
    """
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # تحديد مجلد المخرجات بناءً على مكان ملف المدخلات
    base_folder = os.path.join(os.path.dirname(pdf_input_path), "processed")
    os.makedirs(base_folder, exist_ok=True)
    
    raw_md_path = os.path.join(base_folder, "raw_temp.md")
    final_md_path = os.path.join(base_folder, "RFP_Final_Cleaned.md")

    processor = DynamicDocumentProcessor(output_folder=base_folder)

    # 1. التحويل لـ Markdown خام
    results = parse([pdf_input_path]) 
    raw_content = results[0].markdown
    with open(raw_md_path, "w", encoding="utf-8") as f:
        f.write(raw_content)

    # 2. استخراج القواعد ذكياً
    dynamic_rules = processor.get_cleaning_rules(raw_content, client)

    # 3. التنظيف النهائي
    final_text = processor.clean_document(raw_md_path, dynamic_rules)
    with open(final_md_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    return final_md_path