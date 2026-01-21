from crewai import Agent, Task
from pydantic import BaseModel, Field
from typing import List
#from crewai.tools import tool
from typing import List
from pydantic import BaseModel, Field
import os
from crewai_tools import FileReadTool
from agentic_doc.parse import parse
from dotenv import load_dotenv
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

load_dotenv()
VISION_AGENT_API_KEY=os.environ["VISION_AGENT_API_KEY"]


class MandatoryRequirement(BaseModel):
    requirement_text: str = Field(..., description="Text of the extracted technical or administrative requirement")

    percentage: str = Field("Not Specified", description="Percentage associated with the requirement ")

    deadline: str = Field("Not Specified", description="Date or deadline associated with this requirement")

    section_source: str = Field(..., description="Name of the department from which the requirement was extracted")

class FinalRequirementsOutput(BaseModel):
    requirements: List[MandatoryRequirement]



class Extractor_agent:
    def __init__(self, llm, output_dir="./output"):
        self.llm = llm
        self.output_dir = output_dir
        with open(r"C:\Users\user\OneDrive - University of Prince Mugrin\سطح المكتب\Capstone_Project_SDAIA\src\Agents\request1_final_clean3.md", "r", encoding="utf-8") as f:
            self.full_md_content = f.read()
        self.selected_sections = self._load_toc_data()
        
##-----------------------------------------------
        self.agent = self._create_agent()
        self.task = self._create_task()

        
    def _load_toc_data(self):
        # تأكد من صحة المسار لديك
        json_file_path = r"C:\Users\user\OneDrive - University of Prince Mugrin\سطح المكتب\Capstone_Project_SDAIA\src\outputs\selected_sections.json"
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                toc_data = json.load(f)
            return json.dumps(toc_data, ensure_ascii=False, indent=2)
        except FileNotFoundError:
            print("خطأ: لم يتم العثور على ملف toc_extracted.json")
            return 

    def _create_agent(self):
        return Agent(
            role="RFP Contextual Analyst (Arabic Specialist)",
            goal="Identify and interpret essential requirements and evaluation criteria, providing all outputs in clear, professional Arabic.",
            backstory=(
                "You are a strategic consultant with deep knowledge of Saudi Government procurement law (Etimad). "
                "You excel at analyzing complex RFP documents and extracting binding obligations. "
                "You provide your findings in professional Arabic, ensuring technical terms are accurately translated "
                "and contextually appropriate for the Saudi market."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )
    
    def _create_task(self):
        return Task(
            description=f"""
            Analyze the RFP content {self.full_md_content} focusing on these sections:
            {self.selected_sections}

            YOUR ANALYSIS MISSION (Output must be in ARABIC):
            1. **Semantic Understanding**: استخراج المتطلبات الجوهرية والمعايير التي يجب على مقدم العرض الالتزام بها.
            2. **Identify Evaluation Logic**: تحديد معايير التقييم ونقاط المفاضلة (النوعية والكمية).
            3. **Extract Hidden Obligations**: استخراج الالتزامات الضمنية الموجودة داخل نطاق العمل.

            FOR EACH ELEMENT, EXTRACT IN ARABIC:
            - **The Requirement**: نص المتطلب بأسلوب واضح ومختصر.
            - **Percentage/Value**: القيمة الرقمية أو النسبة المئوية (مثل: الضمان البنكي 1%، وزن المحتوى المحلي 20%). اكتب 'غير محدد' إذا لم توجد.
            - **Deadlines/Timelines**: التواريخ النهائية أو المواعيد الزمنية المرتبطة بالمتطلب.
            """,
            expected_output="A JSON object containing the list of proposed mandatory requirements in Arabic.",
            output_json=FinalRequirementsOutput,
            output_file=os.path.join(self.output_dir, "contextual_requirements_ar.json"),
            agent=self.agent
        )

    @property
    def get_agent(self):
        return self.agent
    
    @property
    def get_task(self):
        return self.task