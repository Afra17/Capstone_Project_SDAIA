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

# ✅ RFP pipeline (A1-A3)
from src.Agents.crew_manager import run_rfp_crew

# ✅ FULL Vendor pipeline (A4 -> A5 -> A6)
from src.Agents.crew_manager import run_vendor_full_pipeline_A4_A5_A6

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
# Simple in-memory job status + results
# -------------------------
job_status: Dict[str, str] = {}
job_results: Dict[str, Dict] = {}


def project_root() -> str:
    """
    Robust project root detector:
    - If this file is inside <root>/src/ -> return <root>
    - If this file is in <root>/ -> return <root>
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(here).lower() == "src":
        return os.path.dirname(here)
    return here


def save_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1]
    if suffix.lower() not in [".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    tmp_dir = tempfile.mkdtemp(prefix="rfp_backend_")
    tmp_path = os.path.join(tmp_dir, upload.filename or "uploaded_file")

    with open(tmp_path, "wb") as out_file:
        shutil.copyfileobj(upload.file, out_file)

    # ✅ Close Windows file handle
    try:
        upload.file.close()
    except Exception:
        pass

    return tmp_path


@app.get("/status/{job_id}")
def status(job_id: str):
    return {"job_id": job_id, "status": job_status.get(job_id, "unknown")}


@app.get("/result/{job_id}")
def get_result(job_id: str):
    return {"job_id": job_id, "result": job_results.get(job_id, None)}


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
    tmp_dir = os.path.dirname(local_path)
    try:
        job_status[job_id] = "running"

        # 1) Clean RFP to markdown
        cleaned_md_path = run_full_cleaning_pipeline(local_path)

        # 2) Outputs directory
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

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
    tmp_dir = os.path.dirname(local_path)
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

        # 2) Run FULL vendor pipeline: A4 -> A5 -> A6 (PER VENDOR)
        result_bundle = run_vendor_full_pipeline_A4_A5_A6(
            proposal_md_path=cleaned_vendor_md_path,
            output_dir=output_dir,
            requirements_json_path=requirements_json,
            apply_adjustments=True,
        )

        # 3) Store results for /result/{job_id}
        job_results[job_id] = {
            "vendor_evidence_path": result_bundle.get("vendor_evidence_path"),
            "scorecard_path": result_bundle.get("scorecard_path"),
            "verified_report_path": result_bundle.get("verified_report_path"),
            "verified_report": result_bundle.get("verified_report"),
        }

        job_status[job_id] = "done"
        print(f"✅ Vendor Job {job_id} finished successfully!")

    except Exception as e:
        job_status[job_id] = f"failed: {str(e)}"
        print(f"❌ Error in Vendor job {job_id}: {str(e)}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
