from crewai import Crew, Process, LLM
from src.Agents.RFP_Strategic_Scout_A1 import selected_agent
from src.Agents.Contextual_Analyst_A2 import Extractor_agent
from src.Agents.llm_as_judge_A3 import judge_agent

# --- ADDED AGENT 4 IMPORT ---
from src.Agents.Response_analyst_A4 import ResponseAnalyst 
# ----------------------------

import os

# تعريف الـ LLM خارج الدالة لزيادة الكفاءة
openai_llm = LLM(model="openai/gpt-4o-mini", temperature=0.1)

def run_rfp_full_crew(md_path: str, output_dir: str):
    """
    هذه الدالة هي المحرك الذي سيستدعيه الـ FastAPI
    """
    # 1. إنشاء الوكلاء (Colleague's original logic)
    selected = selected_agent(llm=openai_llm, md_path=md_path, output_dir=output_dir)
    extractor = Extractor_agent(llm=openai_llm, md_path=md_path, output_dir=output_dir)
    judge = judge_agent(llm=openai_llm, output_dir=output_dir)

    # --- ADDED AGENT 4 INITIALIZATION ---
    # Agent 4 uses the judge's output file to perform the RAG search
    judge_output_path = os.path.join(output_dir, "strategic_refined_requirements.json")
    
    analyst = ResponseAnalyst(
        llm=openai_llm, 
        proposal_md_path=md_path, 
        requirements_json_path=judge_output_path, 
        output_dir=output_dir
    )
    # ------------------------------------

    # 2. إعداد التبعية (Original Flow A1 -> A2 -> A3)
    extractor.set_context_dependency(selected.get_task)
    judge.set_context_dependency(extractor.get_task) # Judge remains tied to Agent 2

    # --- ADDED AGENT 4 CONTEXT ---
    # Agent 4 waits for the Judge (Agent 3) to finish writing the JSON file
    analyst.agent.task.context = [judge.get_task]
    # -----------------------------

    # 3. تكوين الطاقم (Original Agents + Your Agent 4)
    crew = Crew(
        agents=[
            selected.get_agent, 
            extractor.get_agent, 
            judge.get_agent,
            analyst.agent # Agent 4 added here
        ],
        tasks=[
            selected.get_task, 
            extractor.get_task, 
            judge.get_task,
            analyst.task  # Agent 4 task added here
        ],
        process=Process.sequential,
        max_rpm=2
    )

    # 4. التنفيذ وإرجاع النتيجة
    return crew.kickoff()