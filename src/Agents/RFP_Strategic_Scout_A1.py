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
from dotenv import load_dotenv

load_dotenv()


class SelectedSection(BaseModel):
    # ✅ ADDED ONLY (description stricter - no code logic change)
    section_name: str = Field(..., description="The exact text of the heading (copy verbatim, do NOT translate)")
    relevance_score: int = Field(..., description="Importance level from 0-100")
    reasoning: str = Field(..., description="Why is this section important for evaluation?")


class ScoutOutput(BaseModel):
    selected_sections: List[SelectedSection]


class selected_agent:
    def __init__(self, llm, md_path, output_dir="./output"):
        self.llm = llm
        self.md_path = md_path
        self.output_dir = output_dir
        self.preview_tool = self._setup_preview_tool()

        ##-----------------------------------------------
        self.agent = self._create_agent()
        self.task = self._create_task()

    def _setup_preview_tool(self):
        # نأخذ المسار لمتغير محلي لاستخدامه داخل الأداة
        md_path_local = self.md_path

        @tool("rfp_sections_preview_tool")
        def rfp_sections_preview_tool(segment_size: int = 600):
            """
            Reads the markdown file and extracts the full content of the selected sections.
            Input should be a list of section titles.
            """

            import os
            import re

            if not os.path.exists(self.md_path):
                return "Error: File not found."

            with open(md_path_local, "r", encoding="utf-8") as f:
                content = f.read()

            # تقسيم النص بناءً على العناوين
            sections = re.split(r"\n(?=#+\s|\d+\s*-\s)", content)
            smart_toc = []

            for sec in sections:
                lines = sec.strip().split("\n")
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

                smart_toc.append(
                    {
                        "title": title,
                        "preview": preview,
                        "total_section_length": len(body),
                    }
                )
            return smart_toc

        return rfp_sections_preview_tool

    def _create_agent(self):
        return Agent(
            role="RFP Strategic Scout",
            goal="Identify critical sections for proposal evaluation using titles and previews.",
            backstory="""You are a Senior RFP Strategist with deep expertise in Government & Corporate Tenders. 
                Your core skill is 'Structural Decomposition' of procurement documents. 
                You distinguish between two types of content:
                1. **The Administrative Shell:** (Definitions, Legal Boilerplate, General Provisions) which are mandatory but don't differentiate one bidder from another.
                2. **The Execution Core:** (Scope, Methodology, Technical Obligations, KPIs) which are the 'Value Drivers' where the actual evaluation and scoring happen.
                Your mindset is to seek out 'Obligation-Heavy' sections. If a section dictates the 'Lifecycle' of the project—defining what must be built, managed, or provided—it 
                is your highest priority. You ignore the 'Static' and hunt for the 'Dynamic'.

                🔒 LANGUAGE RULES (STRICT):
                - The RFP document is Arabic. Output section_name MUST remain Arabic.
                - NEVER translate headings to English.
                - section_name MUST be copied verbatim exactly from the provided titles (rfp_sections_preview_tool output).
                - If a title is Arabic, you MUST return it exactly as-is (same characters/punctuation).
                """,
            llm=self.llm,
            verbose=True,
            tools=[self.preview_tool],
            allow_delegation=False,
        )

    def _create_task(self):
        return Task(
            description=f"""Analyze the Table of Contents and the contextual previews provided. 
            Your mission is to select headings that represent the 'Actionable Core' of this RFP.
            **Use these Strategic Evaluation Rules:**
            1. **The Performance Rule:** Does this section describe a task, a deliverable, or a standard that the vendor must meet or exceed? (Select it).
            2. **The Scoring Rule:** Does this section contain criteria or methodologies that will be used to grade the technical proposal? (Select it).
            3. **The Resource Rule:** Does it specify required expertise, manpower, or specialized equipment? (Select it).
            4. **The Noise Rule:** If the section is purely informational (Introduction, About us) or purely legal (Dispute Resolution, Termination), 
            give it a low score (<40) and ignore it.

            ✅ OUTPUT CONSTRAINTS (MUST FOLLOW):
            - The document is Arabic → return section_name in Arabic.
            - DO NOT translate section titles.
            - section_name MUST be copied EXACTLY from rfp_sections_preview_tool "title" field.
            - Do NOT invent a title. If unclear, skip it.
            """,
            expected_output="A JSON object containing the list of selected sections that are relevant for proposal evaluation",
            output_json=ScoutOutput,
            output_file=os.path.join(self.output_dir, "selected_sections.json"),
            agent=self.agent,
        )

    @property
    def get_agent(self):
        return self.agent

    @property
    def get_task(self):
        return self.task
