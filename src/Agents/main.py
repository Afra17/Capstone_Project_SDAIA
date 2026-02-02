import os
import shutil

from crewai import Crew, Process, LLM  # استخدام LLM الأصلي لتجاوز مشاكل التعريف
from Response_analyst_A4 import ResponseAnalyst


def run_raw_vendor_b_test():


    # 1) Paths
    VENDOR_FILE = r"D:\Capstone_Project_SDAIA\src\data\processed\Vendor_B.extraction.md"
    REQUIREMENTS_JSON = r"D:\Capstone_Project_SDAIA\src\outputs\strategic_refined_requirements.json"
    OUTPUT_DIR = r"D:\Capstone_Project_SDAIA\src\data\processed"

    # 2) Quick validations (ADDED)
    if not os.path.exists(VENDOR_FILE):
        raise FileNotFoundError(f"Vendor file not found: {VENDOR_FILE}")
    if not os.path.exists(REQUIREMENTS_JSON):
        raise FileNotFoundError(f"Requirements JSON not found: {REQUIREMENTS_JSON}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== DEBUG PATHS ===")
    print("VENDOR_FILE:", VENDOR_FILE)
    print("REQUIREMENTS_JSON:", REQUIREMENTS_JSON)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("===================\n")

    # 3) Init CrewAI LLM
    llm_instance = LLM(model="openai/gpt-4o-mini", temperature=0)

    # 4) Clean previous outputs for full isolation
    evidence_output = os.path.join(OUTPUT_DIR, f"evidence_{os.path.basename(VENDOR_FILE)}.json")
    if os.path.exists(evidence_output):
        os.remove(evidence_output)
        print("🗑️ تم حذف التقرير القديم:", evidence_output)

    # IMPORTANT:
    # Your ResponseAnalyst currently saves FAISS as:
    # faiss_index_{os.path.basename(self.md_path)}_{hash}
    # So we delete ANY matching folder that starts with that name.
    base_faiss_prefix = f"faiss_index_{os.path.basename(VENDOR_FILE)}"

    deleted_any = False
    for name in os.listdir(OUTPUT_DIR):
        full_path = os.path.join(OUTPUT_DIR, name)
        if os.path.isdir(full_path) and name.startswith(base_faiss_prefix):
            shutil.rmtree(full_path, ignore_errors=True)
            deleted_any = True
            print(f"🗑️ تم حذف قاعدة بيانات FAISS القديمة لضمان العزل: {full_path}")

    if not deleted_any:
        print("ℹ️ لم يتم العثور على مجلد FAISS قديم مطابق للحذف (هذا طبيعي لأول تشغيل).")

    # 5) Initialize Agent 4 system (ResponseAnalyst)
    analyst_system = ResponseAnalyst(
        llm=llm_instance,
        proposal_md_path=VENDOR_FILE,
        requirements_json_path=REQUIREMENTS_JSON,
        output_dir=OUTPUT_DIR,
        vendor_label="Vendor b"  # إجبار الاسم في التقرير
    )

    # 6) Run Crew
    crew = Crew(
        agents=[analyst_system.agent],
        tasks=[analyst_system.task],
        process=Process.sequential,
        verbose=False
    )

    print("جاري بناء قاعدة بيانات FAISS واستخراج الأدلة... يرجى الانتظار.")
    crew.kickoff()

    # 7) Final check
    if os.path.exists(evidence_output):
        print(f"✅ تم النجاح! ملف النتائج المعزول موجود هنا: {evidence_output}")
    else:
        print("❌ لم يتم العثور على ملف المخرجات.")
        print("   - تأكدي أن CrewAI انتهى بدون Errors.")
        print("   - تأكدي من صلاحيات الكتابة داخل OUTPUT_DIR.")


if __name__ == "__main__":
    run_raw_vendor_b_test()
