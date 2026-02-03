from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import os
import shutil
import tempfile
import uuid
from dotenv import load_dotenv

from src.Agents.utils.document_processor import run_full_cleaning_pipeline
from src.Agents.utils.proposal_processor import run_pipeline

# ✅ Correct functions
from src.Agents.crew_manager import run_rfp_crew, run_vendor_only_crew

load_dotenv()

app = FastAPI(title="Capstone RFP Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Simple in-memory job status
# -------------------------
job_status: Dict[str, str] = {}


def project_root() -> str:
    """
    Assuming this file is at: <root>/src/BackEnd.py
    Then project root is: <root>
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1]
    if suffix.lower() not in [".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    tmp_dir = tempfile.mkdtemp(prefix="rfp_backend_")
    tmp_path = os.path.join(tmp_dir, upload.filename or "uploaded_file")

    with open(tmp_path, "wb") as out_file:
        shutil.copyfileobj(upload.file, out_file)

    return tmp_path


@app.get("/status/{job_id}")
def status(job_id: str):
    return {"job_id": job_id, "status": job_status.get(job_id, "unknown")}


@app.post("/upload-rfp")
async def upload_rfp(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    local_path = save_upload_to_temp(file)

    job_status[job_id] = "queued"
    background_tasks.add_task(start_rfp_pipeline, job_id, local_path)

    return {
        "message": "RFP processing started in background",
        "job_id": job_id,
        "status_check_url": f"/status/{job_id}",
    }


def start_rfp_pipeline(job_id: str, local_path: str):
    try:
        job_status[job_id] = "running"

        # 1) Clean RFP to markdown
        cleaned_md_path = run_full_cleaning_pipeline(local_path)

        # 2) Outputs directory (stable)
        root = project_root()
        output_dir = os.path.join(root, "src", "outputs")
        os.makedirs(output_dir, exist_ok=True)

        # 3) Run RFP crew (A1-A3 ONLY)
        run_rfp_crew(cleaned_md_path, output_dir)

        job_status[job_id] = "done"
        print(f"✅ RFP Job {job_id} finished successfully!")
    except Exception as e:
        job_status[job_id] = f"failed: {str(e)}"
        print(f"❌ Error in RFP job {job_id}: {str(e)}")


@app.post("/upload-vendors")
async def upload_vendors(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    job_ids = []

    for file in files:
        job_id = f"{(file.filename or 'vendor').split('.')[0]}_{str(uuid.uuid4())[:4]}"
        local_path = save_upload_to_temp(file)

        job_status[job_id] = "queued"
        background_tasks.add_task(start_vendor_pipeline, job_id, local_path)

        job_ids.append(job_id)

    return {
        "message": f"Successfully started processing {len(job_ids)} vendors.",
        "job_ids": job_ids,
    }


def start_vendor_pipeline(job_id: str, local_path: str):
    try:
        job_status[job_id] = "running"

        root = project_root()

        processed_dir = os.path.join(root, "src", "data", "processed")
        os.makedirs(processed_dir, exist_ok=True)

        output_dir = os.path.join(root, "src", "outputs")
        os.makedirs(output_dir, exist_ok=True)

        # Ensure RFP requirements exist
        requirements_json = os.path.join(output_dir, "strategic_refined_requirements.json")
        if not os.path.exists(requirements_json):
            raise FileNotFoundError(
                "strategic_refined_requirements.json not found. "
                "Upload/process the RFP first via /upload-rfp."
            )

        # 1) Clean vendor proposal
        cleaned_vendor_md_path = run_pipeline(local_path, processed_dir)

        # 2) Run vendor-only crew (A4 ONLY)
        run_vendor_only_crew(cleaned_vendor_md_path, output_dir, requirements_json_path=requirements_json)

        job_status[job_id] = "done"
        print(f"✅ Vendor Job {job_id} finished successfully!")
    except Exception as e:
        job_status[job_id] = f"failed: {str(e)}"
        print(f"❌ Error in Vendor job {job_id}: {str(e)}")
