"""
Agent2 : Contextual Analyst (The Extractor)
Input: Full text of the sections selected by the Scout, isolated from the main document.
Task: Deeply analyzes the text to identify specific mandatory obligations, values, and deadlines.
Output: A structured JSON report of actionable requirements in professional Arabic.
"""

from crewai import Agent, Task
from pydantic import BaseModel, Field
from typing import List
from crewai.tools import tool
import os
from dotenv import load_dotenv
import json
import re
load_dotenv()
VISION_AGENT_API_KEY=os.environ["VISION_AGENT_API_KEY"]


class MandatoryRequirement(BaseModel):
    requirement_text: str = Field(..., description="Text of the extracted technical or administrative requirement")
    percentage: str = Field("Not Specified", description="Percentage associated with the requirement ")
    deadline: str = Field("Not Specified", description="Date or deadline associated with this requirement")
    category: str = Field(..., description="Requirement classification: (Technical, Administrative, Financial)")    
    section_name: str =Field(..., description="section name of the requerment")
class FinalRequirementsOutput(BaseModel):
    requirements: List[MandatoryRequirement]



class Extractor_agent:
    def __init__(self, llm, md_path, scout_json_path, output_dir="./output"):
        self.llm = llm
        self.md_path = md_path
        self.scout_json_path = scout_json_path
        self.output_dir = output_dir
        self.extraction_tool = self._setup_extraction_tool()
##-----------------------------------------------
        self.agent = self._create_agent()
        self.task = self._create_task()


    def _setup_extraction_tool(self):
        md_file = self.md_path
        @tool("rfp_text_extractor_tool")
        def rfp_text_extractor_tool(titles: list):
            """
            Extracts full content from the Markdown RFP for a given list of section titles.
            Input: ['Title 1', 'Title 2', ...]
            """
            import re
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                extracted_data = []
                for title in titles:
                    clean_title = title.replace('#', '').strip()
                    # الباترن للبحث عن النص
                    pattern = rf"(?ms)^#*\s*.*{re.escape(clean_title)}.*?\n(.*?)(?=\n#|\n\d+\s*-\s|\Z)"
                    match = re.search(pattern, content)
                    
                    if match:
                        text_content = match.group(1).strip()
                        extracted_data.append(f"SECTION: {title}\nCONTENT:\n{text_content}")
                    else:
                        extracted_data.append(f"SECTION: {title}\nCONTENT: [Text not found]")

                return "\n\n" + "="*30 + "\n\n".join(extracted_data)
            except Exception as e:
                return f"Error: {str(e)}"

        return rfp_text_extractor_tool

    def _create_agent(self):
        return Agent(
            role="RFP Contextual Analyst",
            goal="Extract deep requirements from the provided full text sections.",
            backstory="Specialist in translating complex RFP text into clear, actionable Arabic requirements.",
            llm=self.llm,
            verbose=True,
            tools=[self.extraction_tool], 
            allow_delegation=False,
        )
    
    def _create_task(self):
        return Task(
            description="""
            This tool analyzes the headings that appear in the search results. Your task is to analyze (deconstruct) the text to extract any commitment, criterion, or condition that determines the success or acceptance of the proposal(s).
            Smart business rules:
            1. Microscopy: Scan the text paragraph by paragraph, sentence by sentence. Do not ignore the details that come in the context of the transmitted speech, and extract the “essence of the requirement” from them.
            2. Semantic comprehension: Understand the meaning, not just the words. Any sentence that refers to (necessity, obligation, standard, integrity, compatibility, or schedule) is a “prerequisite” that must be extracted.
            3. Comprehensiveness: It is not limited to technical aspects. Extract requirements (legal, regulatory, administrative, financial, and operational) with the same accuracy.
            4. Logical connection: If a standard is mentioned (such as an ISO, building code, or system), consider it an immediate “success condition” and draw the associated implications.             
            5. Numerical Analysis: Carefully search for any percentages or deadlines associated with the requirement and document them in the designated fields.
            6. Qualitative Classification: Classify each requirement into one of three categories:
            - Technical: Everything related to implementation, quality, and specifications.
            - Administrative: Licenses, team, methodology, and submission process.
            - Financial: Guarantees, payments, penalties, and financial percentages.
            NOTE:Formulate the output in professional Arabic.
            """,
            expected_output="A JSON object containing the list of proposed mandatory requirements in Arabic.",
            output_json=FinalRequirementsOutput,
            output_file=os.path.join(self.output_dir, "contextual_requirements_ar.json"),
            agent=self.agent
        )


    
    def set_context_dependency(self, dependency_task):
        """Set context dependency for the task"""
        self.task.context = [dependency_task]
        return self
    

    @property
    def get_agent(self):
        return self.agent
    
    @property
    def get_task(self):
        return self.task