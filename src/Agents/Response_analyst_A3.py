import os
import json
from crewai import Agent, Task
from crewai_tools import MDXSearchTool
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

# --- 1. Universal Data Schema ---
class RequirementEvidence(BaseModel):
    requirement_name: str = Field(..., description="اسم المتطلب المستخرج من ملف المتطلبات")
    compliance_status: str = Field(..., description="حالة الامتثال (مطابق/غير مطابق/جزئي)")
    evidence_text: str = Field(..., description="النص المقتبس حرفياً من العرض كدليل")
    technical_methodology: str = Field(..., description="شرح تفصيلي للمنهجية أو الحل المقترح من قبل المورد")
    extracted_value: str = Field("غير محدد", description="أي قيم مالية، أرقام، أو تواريخ مستخرجة")

class VendorExtractionReport(BaseModel):
    vendor_name: str = Field(..., description="اسم الجهة الموردة")
    proposal_evidence: List[RequirementEvidence]

# --- 2. High-Detail Agent Class ---
class ResponseAnalyst:
    def __init__(self, llm, proposal_md_path, requirements_json_path, output_dir="./outputs"):
        self.llm = llm
        self.md_path = proposal_md_path 
        self.requirements_path = requirements_json_path
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

       # OVERRIDING DEFAULTS TO PREVENT 429 ERROR
        self.rag_tool = MDXSearchTool(
            mdx=self.md_path,
            config={
                "llm": {
                    "provider": "openai",
                    # Swap to mini: It has a much higher TPM limit than gpt-4o
                    "config": {"model": "gpt-4o-mini", "temperature": 0} 
                },
                "embedder": {
                    "provider": "openai",
                    "config": {"model": "text-embedding-3-small"}
                },
                # Precision Chunking: Smaller chunks stay under the 30k limit
                "chunk_size": 400,      
                "chunk_overlap": 50,
                "retriever": {
                    "k": 8             # High enough for detail, low enough for safety
                }
            }
        )

        self.agent = self._create_agent()
        self.task = self._create_task()

    def _create_agent(self):
        return Agent(
            role="محلل استجابة الموردين (High-Detail RAG Analyst)",
            goal="البحث العميق داخل عروض الموردين لاستخراج تفاصيل فنية ومالية شاملة.",
            backstory="""أنت مدقق بيانات دقيق جداً. تخصصك هو عدم الاكتفاء بالردود السطحية، 
            بل البحث عن "كيفية التنفيذ" (How-to) والمنهجيات الكاملة. أنت تستخدم البحث الدلالي 
            لجمع أكبر قدر ممكن من السياق لكل متطلب لضمان دقة تقييم الخبير.""",
            llm=self.llm,
            tools=[self.rag_tool],
            verbose=True,
            allow_delegation=False
        )
    
    def _create_task(self):
        # Load the strategic refined requirements
        target_reqs = "No requirements provided."
        if os.path.exists(self.requirements_path):
            with open(self.requirements_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                target_reqs = json.dumps(data.get('requirements', []), ensure_ascii=False)

        return Task(
            description=f"""
            مهمتك هي إجراء استخراج "عميق" للأدلة بناءً على هذه المتطلبات: {target_reqs}
            
            الضوابط لضمان كفاية التفاصيل:
            1. لا تتوقف عند العثور على كلمة مفتاحية؛ استخرج الفقرة المحيطة بها بالكامل لفهم السياق.
            2. ابحث عن منهجيات العمل (Technical Methodology) التي تصف الخطوات، الأدوات، والمعايير المستخدمة.
            3. في المتطلبات المالية ، استخرج الجداول أو الأرقام مع شرح البنود التابعة لها.
            4. إذا كان الرد بالإنجليزية، استخرجه كما هو في 'evidence_text' وقم بتلخيص المنهجية بالعربية.
            5. هدفك هو جعل 'technical_methodology' غنية بالمعلومات لدرجة أن Agent 4 لا يحتاج للعودة للملف الأصلي.
            """,
            expected_output="تقرير JSON غني بالتفاصيل التقنية والمالية المقتبسة والمحللة.",
            output_json=VendorExtractionReport,
            output_file=os.path.join(self.output_dir, f"evidence_{os.path.basename(self.md_path)}.json"),
            agent=self.agent
        )