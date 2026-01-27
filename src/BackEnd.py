from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Any, Dict
import os
import shutil
import tempfile
from crewai import LLM
from dotenv import load_dotenv

from src.supabase_config import get_supabase_client, upload_to_supabase
from src.Agents.utils.document_processor import DynamicDocumentProcessor
from src.Agents.RFP_Strategic_Scout_A1 import selected_agent


load_dotenv()

app = FastAPI(title="Capstone RFP Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase_client = get_supabase_client()


def save_upload_to_temp(upload: UploadFile) -> str:
    """Save an uploaded file to a temporary directory and return its path."""
    suffix = os.path.splitext(upload.filename or "")[1]
    if suffix.lower() not in [".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    tmp_dir = tempfile.mkdtemp(prefix="rfp_backend_")
    tmp_path = os.path.join(tmp_dir, upload.filename)

    with open(tmp_path, "wb") as out_file:
        shutil.copyfileobj(upload.file, out_file)

    return tmp_path


def clean_rfp_to_markdown(pdf_or_docx_path: str) -> str:
    """
    Use DynamicDocumentProcessor to convert and clean the RFP into a markdown file.

    Returns the path to the cleaned markdown file.
    """
    base_folder = os.path.join(os.path.dirname(pdf_or_docx_path), "processed")
    os.makedirs(base_folder, exist_ok=True)

    raw_md_path = os.path.join(base_folder, "rfp_raw.md")
    final_md_path = os.path.join(base_folder, "RFP_Final_Cleaned.md")

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    processor = DynamicDocumentProcessor(output_folder=base_folder)

    from agentic_doc.parse import parse

    results = parse([pdf_or_docx_path])
    raw_content = results[0].markdown

    with open(raw_md_path, "w", encoding="utf-8") as f:
        f.write(raw_content)

    dynamic_rules = processor.get_cleaning_rules(raw_content, client)

    if not hasattr(processor, "clean_document"):
        raise RuntimeError("DynamicDocumentProcessor must implement `clean_document` method.")

    final_text = processor.clean_document(raw_md_path, dynamic_rules)

    with open(final_md_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    return final_md_path


def run_agent1_on_markdown(md_path: str) -> Dict[str, Any]:
    """
    Instantiate Agent 1 (selected_agent) and run its task on the cleaned markdown file.

    Returns a dict with the agent output path and (optionally) parsed content.
    """

    openai_llm = LLM(
    model="openai/gpt-4o-mini", 
    temperature=0.1
)

    scout = selected_agent(llm=openai_llm, md_path=md_path, output_dir=os.path.join(os.path.dirname(md_path), "outputs"))

    task = scout.get_task

    result = task.execute()

    output_file = task.output_file
    parsed_output: Any = None
    try:
        import json

        if output_file and os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                parsed_output = json.load(f)
    except Exception:
        parsed_output = None

    return {
        "agent_raw_result": str(result),
        "output_file": output_file,
        "parsed_output": parsed_output,
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload-rfp")
async def upload_rfp(file: UploadFile = File(...)):
    """
    Upload a single RFP (PDF/DOCX), store it in Supabase, clean it,
    and run Agent 1 on the cleaned markdown.
    """
    try:
        local_path = save_upload_to_temp(file)

        supabase_url = upload_to_supabase(
            client=supabase_client,
            file_path=local_path,
            bucket="documents",
            destination_path=f"rfps/{os.path.basename(local_path)}",
        )

        cleaned_md_path = clean_rfp_to_markdown(local_path)

        agent_result = run_agent1_on_markdown(cleaned_md_path)

        return {
            "message": "RFP processed successfully",
            "rfp_supabase_url": supabase_url,
            "cleaned_markdown_path": cleaned_md_path,
            "agent_result": agent_result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-proposals")
async def upload_proposals(files: List[UploadFile] = File(...)):
    """
    Upload up to 5 proposal files (PDF/DOCX) and store them in Supabase.
    Returns the list of public URLs.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one proposal file is required.")

    if len(files) > 5:
        raise HTTPException(status_code=400, detail="You can upload a maximum of 5 proposal files.")

    urls: List[str] = []

    for upload in files:
        local_path = save_upload_to_temp(upload)

        url = upload_to_supabase(
            client=supabase_client,
            file_path=local_path,
            bucket="documents",
            destination_path=f"proposals/{os.path.basename(local_path)}",
        )
        urls.append(url)

    return {
        "message": "Proposals uploaded successfully",
        "proposal_urls": urls,
    }

