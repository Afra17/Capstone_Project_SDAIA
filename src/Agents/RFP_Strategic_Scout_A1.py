"""
Agent 1:RFP Strategic Scout (The Filter)
Input: Table of Contents (TOC) with smart contextual previews of each section.
Task: Evaluates and scores sections to identify those containing critical requirements or evaluation criteria.
Output: A filtered JSON list of high-priority section titles for further analysis.
"""

from crewai import Agent, Task
from pydantic import BaseModel, Field
from typing import List
from crewai.tools import tool
from pydantic import BaseModel, Field
import os
from crewai_tools import FileReadTool
from agentic_doc.parse import parse
from dotenv import load_dotenv
import json
import re

load_dotenv()
VISION_AGENT_API_KEY=os.environ["VISION_AGENT_API_KEY"]



class SelectedSection(BaseModel):
    section_name: str = Field(..., description="The exact text of the heading")
    relevance_score: int = Field(..., description="Importance level from 0-100")
    
class ScoutOutput(BaseModel):
    selected_sections: List[SelectedSection]


class selected_agent:
    def __init__(self, llm, md_path,output_dir="./output"):
        self.llm = llm
        self.md_path = md_path      
        self.output_dir = output_dir  
##-----------------------------------------------
        self.agent = self._create_agent()
        self.task = self._create_task()

    @tool("rfp_sections_preview_tool")
    def _get_contextual_toc(self, segment_size=600):
        """
        Reads the markdown file and extracts the full content of the selected sections.
        Input should be a list of section titles.
        """
        if not os.path.exists(self.md_path):
            return "Error: File not found."
            
        with open(self.md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # تقسيم النص بناءً على العناوين
        sections = re.split(r'\n(?=#+\s|\d+\s*-\s)', content)
        smart_toc = []
        
        for sec in sections:
            lines = sec.strip().split('\n')
            if not lines:
                continue
                
            title = lines[0].strip()
            body = " ".join([l.strip() for l in lines[1:] if l.strip()])
            
            # إذا كان النص طويلاً جداً، نأخذ عينات من أماكن مختلفة
            if len(body) > segment_size * 2:
                start_part = body[:segment_size]
                # أخذ عينة من المنتصف
                mid_point = len(body) // 2
                mid_part = body[mid_point : mid_point + segment_size]
                # أخذ عينة من النهاية
                end_part = body[-segment_size:]
                
                preview = f"[START]: {start_part} ... [MIDDLE]: {mid_part} ... [END]: {end_part}"
            else:
                preview = body
                
            smart_toc.append({
                "title": title,
                "preview": preview,
                "total_section_length": len(body) # نعطي العميل فكرة عن حجم القسم
            })
        return smart_toc
        
    
    def _create_agent(self):
        return Agent(
            role="RFP Strategic Scout",
            goal="Identify critical sections for proposal evaluation using titles and previews.",
            backstory="""""You are an expert Proposal Manager for government tenders (Etimad).
                "Your job is to filter the Table of Contents. 
                "You know that sections like 'Introduction' or 'Definitions' are informational and irrelevant for evaluation. 
                "However, sections like 'Scope of Work', 'Technical Specifications', 'General Provisions', 'Submission Method', and 'Evaluation Criteria' are critical.""",
            llm=self.llm,
            verbose=True,
            tools=[self._get_contextual_toc],
            allow_delegation=False,
        )
    
    def _create_task(self):
        return Task(
            description=f"""Analyze the following RFP sections and their previews
            YOUR MISSION:
            1. Identify headings that imply a **mandatory requirement**, **evaluation criteria**, or **submission instruction**.
            2. Ignore general informational headings
            3. Select sections with score >= 60.
            Be strict but don't miss technical/legal content.""",
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
