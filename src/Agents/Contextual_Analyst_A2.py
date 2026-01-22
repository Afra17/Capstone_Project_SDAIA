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
from typing import List
from pydantic import BaseModel, Field
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

class FinalRequirementsOutput(BaseModel):
    requirements: List[MandatoryRequirement]



class Extractor_agent:
    def __init__(self, llm, md_path, scout_json_path, output_dir="./output"):
        self.llm = llm
        self.md_path = md_path
        self.scout_json_path = scout_json_path
        self.output_dir = output_dir
##-----------------------------------------------
        self.agent = self._create_agent()
        self.task = self._create_task()

    @tool("rfp_text_extractor_tool")
    def cut_full_text(self, titles: list):
        """
        Extracts the full text content of specific sections from the RFP markdown file.
        Input should be a list of exact section titles found during the scout phase.
        """
        with open(self.md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        extracted = []
        for title in titles:
                pattern = rf"(?ms)^#*\s*{re.escape(title)}.*?(?=\n#|\n\d+\s*-\s|\Z)"
                match = re.search(pattern, content)
                if match: 
                    extracted.append(match.group(0))
                else:
                    extracted.append(f"--- [Warning: Section '{title}' not found in the file] ---")
        return "\n\n---\n\n".join(extracted)

    def _create_agent(self):
        return Agent(
            role="RFP Contextual Analyst",
            goal="Extract deep requirements from the provided full text sections.",
            backstory="Specialist in translating complex RFP text into clear, actionable Arabic requirements.",
            llm=self.llm,
            verbose=True,
            tools=[self.cut_full_text], 
            allow_delegation=False,
        )
    
    def _create_task(self):
        return Task(
            description=f"""Analyze the following FULL TEXT from selected RFP sections:
                        Task: Extract all mandatory requirements, percentages, and deadlines in professional Arabic.""",
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