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



class SelectedSection(BaseModel):
    section_name: str = Field(..., description="The exact text of the heading")
    reason: str = Field(..., description="Why is this relevant? (e.g., Mandatory Requirement)")

class ScoutOutput(BaseModel):
    selected_sections: List[SelectedSection]


class selected_agent:
    def __init__(self, llm, output_dir="./output"):
        self.llm = llm
        self.output_dir = output_dir        
##-----------------------------------------------
        self.toc_string_for_prompt = self._load_toc_data()
        self.agent = self._create_agent()
        self.task = self._create_task()
    
    def _load_toc_data(self):
        # تأكد من صحة المسار لديك
        json_file_path = r"C:\Users\user\OneDrive - University of Prince Mugrin\سطح المكتب\Capstone_Project_SDAIA\src\Agents\toc_extracted.json"
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                toc_data = json.load(f)
            return json.dumps(toc_data, ensure_ascii=False, indent=2)
        except FileNotFoundError:
            print("خطأ: لم يتم العثور على ملف toc_extracted.json")
            return ""
    
    def _create_agent(self):
        return Agent(
            role="Table of content Analysis Expert",
            goal="""Analyze RFP Table of Contents and identify sections containing:
            - Proposal evaluation criteria
            - Mandatory requirements
            - Weights and scoring
            - Deadlines
            - Disqualification conditions""",   
            backstory=""" "You are an expert Proposal Manager for government tenders (Etimad).
                "Your job is to filter the Table of Contents. 
                "You know that sections like 'Introduction' or 'Definitions' are informational and irrelevant for evaluation. 
                "However, sections like 'Scope of Work', 'Technical Specifications', 'General Provisions', 'Submission Method', and 'Evaluation Criteria' are critical.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )
    
    def _create_task(self):
        return Task(
            description=f"""Analyze the following Table of Contents (JSON/Text) from an RFP document:
            { self.toc_string_for_prompt}
            YOUR MISSION:
            1. Identify headings that imply a **mandatory requirement**, **evaluation criteria**, or **submission instruction**.
            2. Ignore general informational headings """,
            expected_output="A JSON object containing the list of selected sections that are relevant for proposal evaluation",
            output_json=ScoutOutput,
            output_file=os.path.join(self.output_dir, "selected_sections.json"),
            agent=self.agent
        )

    @property
    def get_agent(self):
        return self.agent
    
    @property
    def get_task(self):
        return self.task