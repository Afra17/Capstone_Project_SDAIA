from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Any, Dict
import os
import shutil
import tempfile
from fastapi import BackgroundTasks, UploadFile, File
import uuid # لتوليد رقم طلب فريد
from crewai import LLM
from dotenv import load_dotenv
from src.supabase_config import get_supabase_client, upload_to_supabase
from src.Agents.utils.document_processor import run_full_cleaning_pipeline
from src.Agents.crew_manager import run_rfp_full_crew
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
    # 1. توليد رقم فريد لهذه العملية
    job_id = str(uuid.uuid4())
    
    # 2. حفظ الملف محلياً بشكل مؤقت (سريع)
    local_path = save_upload_to_temp(file)
    
    # 3. إرسال العملية لـ "الخلفية" لتجنب الانتظار
    # سننشئ دالة وسيطة تسمى 'start_pipeline'
    background_tasks.add_task(start_pipeline, job_id, local_path)
    
    # 4. الرد على المستخدم فوراً
    return {
        "message": "Process started in background",
        "job_id": job_id,
        "status_check_url": f"/status/{job_id}" # مثال لمسار مستقبلي
    }

async def start_pipeline(job_id: str, local_path: str):
    """
    هذه الدالة تعمل في الخلفية. المستخدم لا يراها ولا ينتظرها.
    """
    try:
        # أ. تحديث حالة Job في Supabase إلى "Processing"
        # update_job_status(job_id, "cleaning")

        # ب. التنظيف
        cleaned_md_path = run_full_cleaning_pipeline(local_path)
        output_dir = os.path.join(os.path.dirname(cleaned_md_path), "outputs")

        # ج. تشغيل الـ Crew كاملاً
        # update_job_status(job_id, "running_agents")
        result = run_rfp_full_crew(cleaned_md_path, output_dir)

        # د. حفظ النتيجة النهائية في Supabase وتغيير الحالة لـ "Completed"
        # save_final_result(job_id, result)
        print(f"Job {job_id} finished successfully!")

    except Exception as e:
        # في حال حدث خطأ، سجل ذلك في قاعدة البيانات
        # update_job_status(job_id, "failed", error=str(e))
        print(f"Error in job {job_id}: {str(e)}")