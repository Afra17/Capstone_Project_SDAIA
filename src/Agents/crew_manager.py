from crewai import Crew, Process, LLM
from src.Agents.RFP_Strategic_Scout_A1 import selected_agent
from src.Agents.Contextual_Analyst_A2 import Extractor_agent
from src.Agents.llm_as_judge_A3 import judge_agent
from src.Agents.Response_analyst_A4 import ResponseAnalyst
import os

# Keep LLM global for efficiency
openai_llm = LLM(model="openai/gpt-4o-mini", temperature=0.1)


def run_rfp_crew(md_path: str, output_dir: str):
    """
    RFP pipeline ONLY:
    A1 -> A2 -> A3
    Output: strategic_refined_requirements.json (and other artifacts)
    """
    selected = selected_agent(llm=openai_llm, md_path=md_path, output_dir=output_dir)
    extractor = Extractor_agent(llm=openai_llm, md_path=md_path, output_dir=output_dir)
    judge = judge_agent(llm=openai_llm, output_dir=output_dir)

    # Dependencies: A1 -> A2 -> A3
    extractor.set_context_dependency(selected.get_task)
    judge.set_context_dependency(extractor.get_task)

    crew = Crew(
        agents=[selected.get_agent, extractor.get_agent, judge.get_agent],
        tasks=[selected.get_task, extractor.get_task, judge.get_task],
        process=Process.sequential,
        max_rpm=2,
    )

    return crew.kickoff()


def run_vendor_only_crew(proposal_md_path: str, output_dir: str, requirements_json_path: str = None):
    """
    Vendor pipeline ONLY:
    A4 (Response Analyst)
    Input: cleaned vendor proposal md + strategic_refined_requirements.json
    Output: evidence / mapping json created by A4
    """
    if requirements_json_path is None:
        requirements_json_path = os.path.join(output_dir, "strategic_refined_requirements.json")

    if not os.path.exists(requirements_json_path):
        raise FileNotFoundError(
            f"Requirements JSON not found: {requirements_json_path}. "
            "Upload/process the RFP first to generate strategic_refined_requirements.json"
        )

    analyst = ResponseAnalyst(
        llm=openai_llm,
        proposal_md_path=proposal_md_path,
        requirements_json_path=requirements_json_path,
        output_dir=output_dir,
    )

    crew = Crew(
        agents=[analyst.agent],
        tasks=[analyst.task],
        process=Process.sequential,
        max_rpm=2,
    )

    return crew.kickoff()
