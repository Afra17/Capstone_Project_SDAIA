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

        # VECTORIZATION & RAG CONFIGURATION
        # The MDXSearchTool automates the mathematical process of turning text into numbers (Vectors).
        self.rag_tool = MDXSearchTool(
            mdx=self.md_path,
            config={
                "llm": {
                    "provider": "openai",
                    "config": {"model": "gpt-4o-mini", "temperature": 0} 
                },
                "embedder": {
                    "provider": "openai",
                    # This model performs the Vectorization: converting text segments into 1536-dimensional vectors.
                    "config": {"model": "text-embedding-3-small"}
                },
                # Precision Chunking: Text is split into 400-character pieces for vector indexing.
                "chunk_size": 400,      
                "chunk_overlap": 50,
                "retriever": {
                    # Similarity Search: When searching, the tool calculates Cosine Similarity between 
                    # the query and the text chunks, returning the 'k' most relevant matches.
                    "k": 8             
                }
            }
        )

        self.agent = self._create_agent()
        self.task = self._create_task()

    def _create_agent(self):
        """Initializes the Agent with the RAG tool for deep semantic searching."""
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
        """Defines the specific extraction task using the refined requirements as the search query."""
        # 1. Load the requirements extracted by Agent 3
        target_reqs = "No requirements provided."
        if os.path.exists(self.requirements_path):
            with open(self.requirements_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # These requirements will be converted into vectors to query the proposal via Similarity Search.
                target_reqs = json.dumps(data.get('requirements', []), ensure_ascii=False)

        return Task(
            description=f"""
            مهمتك هي إجراء استخراج "عميق" للأدلة بناءً على هذه المتطلبات: {target_reqs}
            
            الضوابط لضمان كفاية التفاصيل:
            1. لا تتوقف عند العثور على كلمة مفتاحية؛ استخرج الفقرة المحيطة بها بالكامل لفهم السياق.
            2. ابحث عن منهجيات العمل (Technical Methodology) التي تصف الخطوات، الأدوات، والمعايير المستخدمة.
            3. في المتطلبات المالية ، استخرج الجداول أو الأرق مع شرح البنود التابعة لها.
            4. إذا كان الرد بالإنجليزية، استخرجه كما هو في 'evidence_text' وقم بتلخيص المنهجية بالعربية.
            5. هدفك هو جعل 'technical_methodology' غنية بالمعلومات لدرجة أن Agent 4 لا يحتاج للعودة للملف الأصلي.
            """,
            expected_output="تقرير JSON غني بالتفاصيل التقنية والمالية المقتبسة والمحللة.",
            output_json=VendorExtractionReport,
            # Dynamically names the evidence file based on the vendor proposal name
            output_file=os.path.join(self.output_dir, f"evidence_{os.path.basename(self.md_path)}.json"),
            agent=self.agent
        )