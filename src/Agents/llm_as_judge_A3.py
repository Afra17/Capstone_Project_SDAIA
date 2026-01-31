"""
Agent 3: Strategic Requirement Auditor (The Judge)
Input: A raw, comprehensive JSON list of all extracted obligations and tasks from Agent 2.
Task: Critically filters the list to eliminate administrative noise and legal boilerplate, retaining only competitive, high-impact technical and financial criteria.
Output: A refined, prioritized JSON report in professional Arabic, featuring strategic reasoning and impact weighting (1-5) for each requirement.
"""
from crewai import Agent, Task
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()

class FinalRequirement(BaseModel):
    requirement_text: str = Field(..., description="Text of the extracted technical or administrative requirement")
    percentage: str = Field("Not Specified", description="Percentage associated with the requirement ")
    deadline: str = Field("Not Specified", description="Date or deadline associated with this requirement")
    category: str = Field(..., description="Requirement classification: (Technical, Administrative, Financial)")    
    section_name: str =Field(..., description="section name of the requerment")
    impact_weight: int = Field(..., ge=1, le=5, description="From 1-5, how much does this affect the final technical score?")
    reasoning: str = Field(..., description="Strategic explanation: Why is this requirement competitive and not just administrative noise?")
class FinalRequirementsOutput(BaseModel):
    requirements: List[FinalRequirement]



class judge_agent:
    def __init__(self, llm, output_dir="./output"):
        self.llm = llm
        self.output_dir = output_dir
##-----------------------------------------------
        self.agent = self._create_agent()
        self.task = self._create_task()


    def _create_agent(self):
        return Agent(
    role="Strategic Requirement Auditor",
    goal="Audit the requirement classifications to ensure only truly 'Evaluative' items are marked True.",
    backstory="""You are a cynical procurement expert. Your job is to filter out 'Administrative Noise'. 
    You believe that many requirements marked as 'Technical' or 'True' are actually just legal 
    formalities (like General Safety, Saudi Laws, or Mandatory Lists). 
    You only allow a 'True' status for things that involve technical expertise, 
    methodology, or specific deliverables that differentiate bidders.""",
    llm=self.llm,
    verbose=True,
    allow_delegation=False,

)
    
    def _create_task(self):
        return Task(
           description="""
            "Analyze the provided requirements list. Your mission is to "Purge" the list by keeping ONLY 'Meaningful' requirements.
            A Requirement is 'Meaningful' ONLY if:
            It is Competitive: Bidders can provide different levels of quality/expertise for it.
            It is Technical/Operational: It describes the core mission (Design, Build, Code, Manage).
            It has Scoring Value: It directly influences the technical or financial score.
            DISCARD (Mute) anything that is:
            General Legal Obligations (Following Saudi laws, tax compliance).
            Generic Safety/Quality promises without specific metrics.
            Administrative procedures (How to submit, how the committee opens envelopes).
            Standard Eligibility (Commercial register, Zakat certificate).
            "Be Extremely Skeptical: If a requirement is a 'Promise to follow the law' or a 'Routine notification', it is NOT evaluative. 
            A requirement is ONLY evaluative if it describes a Deliverable or a Methodology that can be better in Proposal A than in Proposal B."
            **For each retained requirement:**
            - Assign an `impact_weight` (1-5) based on its criticality to the project success.
            - Write a `reasoning` in Arabic explaining why this is a 'competitive' requirement.
            NOTE:Formulate the output in professional Arabic.""",
            expected_output="A refined JSON containing only the 'High-Signal' requirements that justify the RFP evaluation.",
            output_json=FinalRequirementsOutput,
            output_file=os.path.join(self.output_dir, "strategic_refined_requirements.json"),
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