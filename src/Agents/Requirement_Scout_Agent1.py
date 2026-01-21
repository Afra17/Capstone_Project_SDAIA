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
        with open(r"C:\Users\user\OneDrive - University of Prince Mugrin\سطح المكتب\Capstone_Project_SDAIA\src\Agents\request.md", "r", encoding="utf-8") as f:
            self.full_md_content = f.read()
        self.selected_sections = self._load_toc_data()
        
##-----------------------------------------------
        self.agent = self._create_agent()
        self.task = self._create_task()

        
    def _load_toc_data(self):
        # تأكد من صحة المسار لديك
        json_file_path = r"C:\Users\user\OneDrive - University of Prince Mugrin\سطح المكتب\Capstone_Project_SDAIA\src\outputs\selscted.json"
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                toc_data = json.load(f)
            return json.dumps(toc_data, ensure_ascii=False, indent=2)
        except FileNotFoundError:
            print("خطأ: لم يتم العثور على ملف toc_extracted.json")
            return 

    def _create_agent(self):
        return Agent(
            role="RFP Contextual Analyst",
            goal="Identify and interpret essential requirements and evaluation criteria by understanding the semantic meaning of the text .",
            backstory=(
        """You are a strategic consultant with deep knowledge of Saudi Government procurement law. 
                You don't just look for keywords; you analyze the 'intent' of each paragraph. 
                You can distinguish between a general description and a binding obligation. 
                Even if a requirement is phrased subtly, you can identify it as a mandatory element or a scoring criterion.
            """),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )
    
    def _create_task(self):
        return Task(
            description=f"""
            Analyze the RFP content{self.full_md_content} within the following selected headings:
            {self.selected_sections}
            YOUR ANALYSIS MISSION:
            1. **Semantic Understanding**: Read each paragraph to understand its purpose. If the text defines a standard the bidder must meet or a condition for success, mark it as an 'Essential Requirement'.
            2. **Identify Evaluation Logic**: Spot criteria that influence the 'Scoring' . Look for qualitative measures (e.g., 'quality of previous work', 'methodology effectiveness') not just quantitative ones.
            3. **Extract Hidden Obligations**: Sometimes requirements are implied within descriptions of the 'Scope of Work'. If a task is described as a part of the project, it is a requirement.
            
            EXTRACT THE FOLLOWING FOR EACH IDENTIFIED ELEMENT:
            - **The Requirement**: The core essence of what is being asked (paraphrased for clarity if needed).
            - **Percentage/Value**: Any numerical weight or financial ratio associated with it (e.g., 5% Guarantee, 20% Local Content weight). Write 'Not Specified' if absent.
            - **Deadlines/Timelines**: Any temporal constraint (e.g., 'within 10 days', 'before the bid closing').
            """,
            expected_output="A JSON object containing the list of proposed mandatory requirements",
            output_json=FinalRequirementsOutput,
            output_file=os.path.join(self.output_dir, "contextual_requirements.json"),
            agent=self.agent
        )

    @property
    def get_agent(self):
        return self.agent
    
    @property
    def get_task(self):
        return self.task