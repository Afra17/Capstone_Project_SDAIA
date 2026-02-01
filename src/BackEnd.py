from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Any, Dict
import os
import shutil
import tempfile
from fastapi import BackgroundTasks, UploadFile, File
import uuid 
from crewai import LLM
from dotenv import load_dotenv

# --- COLLEAGUE'S IMPORTS ---
from src.supabase_config import get_supabase_client, upload_to_supabase
from src.Agents.utils.document_processor import run_full_cleaning_pipeline
from src.Agents.crew_manager import run_rfp_full_crew

# --- YOUR ADDED IMPORTS ---
# This brings in your chunked cleaner to handle the proposals
from src.Agents.utils.proposal_processor import run_pipeline 

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

@app.post("/upload-rfp")
async def upload_rfp(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    local_path = save_upload_to_temp(file)
    background_tasks.add_task(start_pipeline, job_id, local_path)
    
    return {
        "message": "Process started in background",
        "job_id": job_id,
        "status_check_url": f"/status/{job_id}"
    }

async def start_pipeline(job_id: str, local_path: str):
    try:
        cleaned_md_path = run_full_cleaning_pipeline(local_path)
        output_dir = os.path.join(os.path.dirname(cleaned_md_path), "outputs")
        # Note: Agent 4 is now inside run_rfp_full_crew
        result = run_rfp_full_crew(cleaned_md_path, output_dir)
        print(f"Job {job_id} finished successfully!")
    except Exception as e:
        print(f"Error in job {job_id}: {str(e)}")

@app.post("/upload-vendors")
async def upload_vendors(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Handles multiple vendor proposals. The user uploads all files, 
    and we process them one by one in the background.
    """
    job_ids = []
    for file in files:
        job_id = f"{file.filename.split('.')[0]}_{str(uuid.uuid4())[:4]}"
        local_path = save_upload_to_temp(file)
        
        # Start the specific vendor pipeline using your Response Analyst logic
        background_tasks.add_task(start_vendor_pipeline, job_id, local_path)
        job_ids.append(job_id)

    return {
        "message": f"Successfully started processing {len(job_ids)} vendors.",
        "job_ids": job_ids
    }

async def start_vendor_pipeline(job_id: str, local_path: str):
    """
    Background task for vendor proposals using your specialized cleaning 
    and the Response Analyst (Agent 4).
    """
    try:
        processed_dir = r"D:\Capstone_Project_SDAIA\src\data\processed"
        os.makedirs(processed_dir, exist_ok=True)
        
        cleaned_md_path = run_pipeline(local_path, processed_dir)
        
        # 2. Run the Full Crew (A1, A2, A3, and your A4)
        output_dir = r"D:\Capstone_Project_SDAIA\src\outputs"
        result = run_rfp_full_crew(cleaned_md_path, output_dir)
        
        print(f"✅ Vendor Job {job_id} finished successfully!")
    except Exception as e:
        print(f"❌ Error in Vendor Job {job_id}: {str(e)}")