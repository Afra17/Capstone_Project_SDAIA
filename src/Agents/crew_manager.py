from crewai import Crew, Process, LLM
from src.Agents.RFP_Strategic_Scout_A1 import selected_agent
from src.Agents.Contextual_Analyst_A2 import Extractor_agent
from src.Agents.llm_as_judge_A3 import judge_agent
from src.Agents.Response_analyst_A4 import ResponseAnalyst

# ✅ A5 + A6 (your real class names)
from src.Agents.Scoring_A5 import VendorScoringAgent
from src.Agents.Moderator_A6 import VendorModerator

import os
import json
import glob
from typing import Optional, Dict, Any

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


# ==========================================================
# ✅ NEW: FULL vendor pipeline wrapper (A4 -> A5 -> A6)
# (No need to force A5/A6 into Crew tasks; we run them safely in Python)
# ==========================================================

def _latest_file(patterns) -> Optional[str]:
    """
    Return most recently modified file among glob patterns.
    """
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]


def _find_vendor_evidence_json(output_dir: str, proposal_md_path: str) -> str:
    """
    A4 writes an evidence JSON somewhere in output_dir.
    We locate the latest evidence json related to this vendor proposal.
    """
    base = os.path.splitext(os.path.basename(proposal_md_path))[0]

    patterns = [
        os.path.join(output_dir, f"*{base}*.json"),
        os.path.join(output_dir, f"*evidence*{base}*.json"),
        os.path.join(output_dir, f"*Evidence*{base}*.json"),
        os.path.join(output_dir, f"evidence*.json"),
        os.path.join(output_dir, f"*extraction*.json"),
    ]

    path = _latest_file(patterns)
    if not path:
        raise FileNotFoundError(
            f"Could not find vendor evidence JSON after A4. "
            f"Searched in: {output_dir} using patterns: {patterns}"
        )
    return path


def run_vendor_full_pipeline_A4_A5_A6(
    proposal_md_path: str,
    output_dir: str,
    requirements_json_path: str = None,
    apply_adjustments: bool = True,
) -> Dict[str, Any]:
    """
    Vendor FULL pipeline:
      A4 -> A5 -> A6

    Returns bundle:
      {
        "vendor_evidence_path": ...,
        "scorecard_path": ...,
        "verified_report_path": ...,
        "verified_report": { ...dict... }
      }
    """
    if requirements_json_path is None:
        requirements_json_path = os.path.join(output_dir, "strategic_refined_requirements.json")

    if not os.path.exists(requirements_json_path):
        raise FileNotFoundError(
            f"Requirements JSON not found: {requirements_json_path}. "
            "Upload/process the RFP first to generate strategic_refined_requirements.json"
        )

    os.makedirs(output_dir, exist_ok=True)

    # --------------------
    # A4: Evidence extraction
    # --------------------
    run_vendor_only_crew(
        proposal_md_path=proposal_md_path,
        output_dir=output_dir,
        requirements_json_path=requirements_json_path,
    )

    vendor_evidence_path = _find_vendor_evidence_json(output_dir, proposal_md_path)

    # --------------------
    # A5: Scoring (PER VENDOR)
    # --------------------
    scorer = VendorScoringAgent(
        llm=openai_llm,
        golden_template_path=requirements_json_path,
        vendor_evidence_path=vendor_evidence_path,
        output_dir=output_dir,
    )
    scorecard_path, scorecard_data = scorer.run()

    # Ensure scorecard saved (some agents write automatically, but we enforce)
    try:
        with open(scorecard_path, "w", encoding="utf-8") as f:
            json.dump(scorecard_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # --------------------
    # A6: Moderator (PER VENDOR)
    # --------------------
    with open(requirements_json_path, "r", encoding="utf-8") as f:
        golden = json.load(f)
    with open(vendor_evidence_path, "r", encoding="utf-8") as f:
        evidence = json.load(f)

    moderator = VendorModerator(llm=openai_llm, apply_adjustments=apply_adjustments)
    verified_report = moderator.verify(golden=golden, evidence=evidence, scorecard=scorecard_data)

    # Write verified report
    vendor_base = os.path.splitext(os.path.basename(proposal_md_path))[0]
    verified_report_path = os.path.join(output_dir, f"verified_report_{vendor_base}.json")
    with open(verified_report_path, "w", encoding="utf-8") as f:
        json.dump(verified_report, f, ensure_ascii=False, indent=2)

    return {
        "vendor_evidence_path": vendor_evidence_path,
        "scorecard_path": scorecard_path,
        "verified_report_path": verified_report_path,
        "verified_report": verified_report,
    }
