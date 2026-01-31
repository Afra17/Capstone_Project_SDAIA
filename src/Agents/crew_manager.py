from crewai import Crew, Process, LLM
from src.Agents.RFP_Strategic_Scout_A1 import selected_agent
from src.Agents.Contextual_Analyst_A2 import Extractor_agent
from src.Agents.llm_as_judge_A3 import judge_agent

import os

# تعريف الـ LLM خارج الدالة لزيادة الكفاءة
openai_llm = LLM(model="openai/gpt-4o-mini", temperature=0.1)

def run_rfp_full_crew(md_path: str, output_dir: str):
    """
    هذه الدالة هي المحرك الذي سيستدعيه الـ FastAPI
    """
    # 1. إنشاء الوكلاء بالمسارات الديناميكية التي وصلت من الـ API
    selected = selected_agent(llm=openai_llm, md_path=md_path, output_dir=output_dir)
    extractor = Extractor_agent(llm=openai_llm, md_path=md_path, output_dir=output_dir)
    judge = judge_agent(llm=openai_llm, output_dir=output_dir)

    # 2. إعداد التبعية (الربط بين المهام)
    extractor.set_context_dependency(selected.get_task)
    judge.set_context_dependency(extractor.get_task)

    # 3. تكوين الطاقم
    crew = Crew(
        agents=[selected.get_agent, extractor.get_agent, judge.get_agent],
        tasks=[selected.get_task, extractor.get_task, judge.get_task],
        process=Process.sequential,
        max_rpm=2
    )

    # 4. التنفيذ وإرجاع النتيجة
    return crew.kickoff()