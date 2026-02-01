import os
from crewai import Crew, Process
from langchain_openai import ChatOpenAI
from Response_analyst_A3 import ResponseAnalyst

def run_vendor_b_test():
    # --- Paths ---
    VENDOR_B_MD = r"D:\Capstone_Project_SDAIA\src\data\processed\Vendor_B.extraction.md"
    REQUIREMENTS_JSON = r"D:\Capstone_Project_SDAIA\src\outputs\strategic_refined_requirements.json"
    OUTPUT_DIR = r"D:\Capstone_Project_SDAIA\src\outputs"

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    print(f"--- Starting Analysis for Vendor b ---")
    
    # Initialize Agent 3
    analyst_system = ResponseAnalyst(
        llm=llm,
        proposal_md_path=VENDOR_B_MD,
        requirements_json_path=REQUIREMENTS_JSON,
        output_dir=OUTPUT_DIR
    )

    # --- THE FIX: Wrap in a Crew to ensure tool execution and file saving ---
    crew = Crew(
        agents=[analyst_system.agent],
        tasks=[analyst_system.task],
        process=Process.sequential,
        verbose=True
    )

    print(f"Step 1: Running RAG Extraction...")
    result = crew.kickoff() # Kickoff handles the logic flow correctly

    # --- Validation ---
    evidence_file = os.path.join(OUTPUT_DIR, f"evidence_{os.path.basename(VENDOR_B_MD)}.json")
    if os.path.exists(evidence_file):
        print(f"Success! Evidence JSON created at: {evidence_file}")
    else:
        print(f"Error: Evidence file was not generated at {evidence_file}")

if __name__ == "__main__":
    run_vendor_b_test()