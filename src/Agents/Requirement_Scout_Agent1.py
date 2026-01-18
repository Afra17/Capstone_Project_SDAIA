from crewai import Agent, Task
from pydantic import BaseModel, Field
from typing import List
from crewai.tools import tool
from typing import List
from pydantic import BaseModel, Field
import os





no_keywords=10
class Requirement_Scout_agent (BaseModel):

     queries: List[str] = Field(...,title="",
                               min_length=1, max_length=no_keywords)




# Requirement_Scout_agent.py
class Requirement_Scout_agent1:
    def __init__(self, llm, output_dir="./output"):
        self.llm = llm
        self.output_dir = output_dir
        self.agent = self._create_agent()
        self.task = self._create_task()
    
    def _create_agent(self):
        return Agent(
            role="",
            goal="\n".join([""
                
            ]),
            backstory="",
            llm=self.llm,
            verbose=True
        )
    
    def _create_task(self):
        return Task(
            description="\n".join([
                ""
            ]),
            expected_output="",
            output_json="file",
            output_file=os.path.join(self.output_dir, ".json"),
            agent=self.agent
        )

    @property
    def get_agent(self):
        return self.agent
    
    @property
    def get_task(self):
        return self.task