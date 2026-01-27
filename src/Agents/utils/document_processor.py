import re
import os
import json
from unstructured.partition.md import partition_md
from unstructured.cleaners.core import clean_extra_whitespace
from dotenv import load_dotenv
import os 
from openai import OpenAI 



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
            "bilingual text indicates","national symbolism","Surrounding Outline :"
                
            
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

    def clean_with_unstructured(self, file_path, rules):
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



if __name__ == "__main__":
    # 1. تحميل مفتاح API
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 2. تحديد المسارات (تأكدي من صحتها في جهازك)
    # المسار للملف الخام المستخرج من الـ PDF
    raw_file_path = r"C:\Users\user\OneDrive - University of Prince Mugrin\سطح المكتب\Capstone_Project_SDAIA\src\data\RFP_raw.md"
    # مجلد الحفظ للملف النظيف
    output_dir = r"C:\Users\user\OneDrive - University of Prince Mugrin\سطح المكتب\Capstone_Project_SDAIA\src\data\processed"

    # 3. تهيئة المعالج
    processor = DynamicDocumentProcessor(output_folder=output_dir)

    print("🧠 جاري تحليل المستند لاستخراج القواعد...")
    with open(raw_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # الحصول على القواعد الديناميكية
    dynamic_rules = processor.get_cleaning_rules(content, client)

    print("🧹 جاري التنظيف (حذف أوصاف الصور والهيدرز المتكررة)...")
    final_md_text = processor.clean_with_unstructured(raw_file_path, dynamic_rules)

    # 4. حفظ النتيجة النهائية
    final_output_path = os.path.join(output_dir, "RFP_Final_Cleaned.md")
    with open(final_output_path, "w", encoding="utf-8") as f:
        f.write(final_md_text)

    print(f"✅ انتهى التنظيف بنجاح!")
    print(f"📂 الملف النظيف موجود هنا: {final_output_path}")