import os
import sys
import json
from pathlib import Path
import time
import requests
import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Your runners live here:
import os
os.environ["CREWAI_TELEMETRY_DISABLED"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

#### 1
import shutil

if "ran_pipeline" not in st.session_state:
    st.session_state["ran_pipeline"] = False

def clear_outputs():
    for p in [AGENT5_DIR, AGENT6_DIR]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
####

# --- Directories (match your project layout) ---
OUTPUTS_DIR = SRC_DIR / "outputs"
AGENT5_DIR = OUTPUTS_DIR / "agent5_scoring"
AGENT6_DIR = OUTPUTS_DIR / "agent6_moderator"

OUTPUTS_DIR.mkdir(exist_ok=True)
AGENT5_DIR.mkdir(parents=True, exist_ok=True)
AGENT6_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_json(path: Path, default):
    try:
        return load_json(path)
    except Exception:
        return default


# Backend(FastAPI)
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")


def _poll_job(job_id: str, timeout_sec: int = 900, sleep_sec: float = 2.0):
    """Poll /status/{job_id} until done/failed or timeout."""
    start = time.time()
    while True:
        r = requests.get(f"{BACKEND_BASE_URL}/status/{job_id}", timeout=30)
        r.raise_for_status()
        status = r.json().get("status", "unknown")

        if status == "done":
            return "done"
        if isinstance(status, str) and status.startswith("failed"):
            raise RuntimeError(status)

        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Timeout waiting for job {job_id}")

        time.sleep(sleep_sec)


def _get_result(job_id: str) -> dict:
    r = requests.get(f"{BACKEND_BASE_URL}/result/{job_id}", timeout=30)
    r.raise_for_status()
    return r.json().get("result") or {}


def _upload_rfp_to_backend(uploaded_file):
    """Calls POST /upload-rfp with the uploaded file."""
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")
    }
    r = requests.post(f"{BACKEND_BASE_URL}/upload-rfp", files=files, timeout=60)
    r.raise_for_status()
    return r.json()["job_id"]


def _upload_vendors_to_backend(uploaded_files):
    """Calls POST /upload-vendors with multiple uploaded files."""
    files = []
    for f in uploaded_files:
        files.append(("files", (f.name, f.getvalue(), f.type or "application/octet-stream")))

    r = requests.post(f"{BACKEND_BASE_URL}/upload-vendors", files=files, timeout=120)
    r.raise_for_status()
    return r.json().get("job_ids", [])


def _write_verified_outputs_from_backend(vendor_job_ids: list):
    """
    Fetch backend verified reports and write them into:
      outputs/agent6_moderator/verified_*.json
      outputs/agent6_moderator/verified_leaderboard.json

    So your existing UI reads them normally.
    """
    leaderboard = []

    for jid in vendor_job_ids:
        result = _get_result(jid)
        verified_report = result.get("verified_report") or {}

        vendor_name = verified_report.get("vendor_name", jid)
        score = verified_report.get("verified_total_score_percent", 0)
        rec = verified_report.get("recommendation", "—")
        crit = verified_report.get("critical_missing", [])

        # Save verified report file
        safe_vendor = "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in vendor_name)
        out_path = AGENT6_DIR / f"verified_{safe_vendor}_{jid}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(verified_report, f, ensure_ascii=False, indent=2)

        leaderboard.append({
            "vendor_name": vendor_name,
            "verified_total_score_percent": score,
            "recommendation": rec,
            "critical_missing_count": len(crit) if isinstance(crit, list) else 0,
            "verified_file": str(out_path)
        })

    # sort and write leaderboard
    leaderboard.sort(key=lambda x: x["verified_total_score_percent"], reverse=True)
    lb_path = AGENT6_DIR / "verified_leaderboard.json"
    with open(lb_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)

    return leaderboard
#endof backend



# UI
st.set_page_config(page_title="RFP Vendor Evaluation", layout="wide")
st.markdown("""
<style>
/* Keep background white */
.stApp { background: white; }

/* Sidebar primary button -> green */
section[data-testid="stSidebar"] button[kind="primary"] {
    background-color: #2BB673 !important;   /* green */
    border: 1px solid #2BB673 !important;
    color: white !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] button[kind="primary"]:hover {
    background-color: #239a61 !important;
    border: 1px solid #239a61 !important;
}

/* Headings accent (blue) */
h1, h2, h3 {
    color: #00416B !important;  /* blue */
}
</style>
""", unsafe_allow_html=True)

#Header
LOGO_PATH = r"D:\Capstone_Project_SDAIA\src\assets\SDAIA-Logo-2.jpg"

header_left, header_right = st.columns([6, 1])
with header_left:
    st.title("Vendor Evaluation Dashboard")
    st.caption("A clear executive view of vendor compliance, evidence strength, and critical gaps.")
with header_right:
    logo = Image.open(LOGO_PATH)

    st.image(
        logo,
        use_container_width=False,
        width=160
    )


# Sidebar accepts PDF , JSON
st.sidebar.title("Inputs")

st.sidebar.subheader("1) Upload RFP")
golden_file = st.sidebar.file_uploader(
    "Upload RFP file",
    type=["pdf", "json"],
    key="golden"
)

st.sidebar.subheader("2) Upload Proposals")
evidence_files = st.sidebar.file_uploader(
    "Upload vendor proposal files",
    type=["pdf", "json"],
    accept_multiple_files=True,
    key="evidence"
)
st.sidebar.markdown("""
<style>
section[data-testid="stSidebar"] .stButton > button {
    width: 170px !important;
    height: 42px !important;
    padding: 6px 18px !important;
    margin: 0 auto !important;
    display: block !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

sp1, mid, sp2 = st.sidebar.columns([1, 2, 3])
with mid:
    run_now = st.button("Run Evaluation", type="primary")

st.sidebar.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)


# Save uploads into src/outputs
golden_path = OUTPUTS_DIR / "strategic_refined_requirements.json"

if golden_file:
    save_uploaded_file(golden_file, golden_path)
    st.sidebar.success("RFP saved")

if evidence_files:
    for ef in evidence_files:
        save_uploaded_file(ef, OUTPUTS_DIR / ef.name)
    st.sidebar.success(f"{len(evidence_files)} Proposals saved")



# Run pipeline (BACKEND)
if run_now:
    st.session_state["ran_pipeline"] = True

    if not golden_file or not evidence_files:
        st.error("Upload BOTH: (1) RFP  and (2) at least one Proposal .")
    else:
        # backend supports PDF/DOCX only — keep UI same, but guard here
        if Path(golden_file.name).suffix.lower() not in [".pdf", ".docx"]:
            st.error("Backend accepts RFP as PDF/DOCX only. Please upload RFP PDF/DOCX.")
            st.stop()

        bad_vendor = [f.name for f in evidence_files if Path(f.name).suffix.lower() not in [".pdf", ".docx"]]
        if bad_vendor:
            st.error(f"Backend accepts proposals as PDF/DOCX only. These are not supported: {bad_vendor}")
            st.stop()

        with st.status("Running pipeline...", expanded=True) as status:
            # 1) Upload RFP
            st.write("Uploading RFP to backend...")
            rfp_job_id = _upload_rfp_to_backend(golden_file)
            st.write(f"RFP job started: {rfp_job_id}")

            st.write("Processing RFP (background)...")
            _poll_job(rfp_job_id)
            st.write(" RFP processing done.")

            # 2) Upload vendors
            st.write("Uploading vendor proposals to backend...")
            vendor_job_ids = _upload_vendors_to_backend(evidence_files)
            if not vendor_job_ids:
                st.error("No vendor jobs returned from backend.")
                st.stop()

            st.write(f"Vendor jobs started: {vendor_job_ids}")

            # 3) Poll each vendor job
            for jid in vendor_job_ids:
                st.write(f"Processing vendor: {jid} ...")
                _poll_job(jid)
                st.write(f" Done: {jid}")

            # 4) Write outputs into your current folders so UI stays unchanged
            st.write("Writing verified outputs for dashboard...")
            _write_verified_outputs_from_backend(vendor_job_ids)
            st.write("Outputs saved for Streamlit UI.")

            status.update(label="Done", state="complete")



# Load results (if exist)
verified_lb_path = AGENT6_DIR / "verified_leaderboard.json"
agent5_lb_path = AGENT5_DIR / "leaderboard_agent5.json"

###4
# verified_lb = safe_read_json(verified_lb_path, default=[])
# agent5_lb = safe_read_json(agent5_lb_path, default=[])
#uncomment  this below and comment out the above 
if st.session_state["ran_pipeline"]:
    verified_lb = safe_read_json(verified_lb_path, default=[])
    agent5_lb = safe_read_json(agent5_lb_path, default=[])
else:
    st.info("Upload files then click **Run Evaluation**.")
    st.stop()



# Executive Summary
st.subheader("Executive Leaderboard")
if verified_lb:
    df = pd.DataFrame(verified_lb)

    # Keep original keys, but DISPLAY nicer labels
    df = df.rename(columns={
        "vendor_name": "Vendor",
        "verified_total_score_percent": "Final Score (%)",
        "recommendation": "Decision",
        "critical_missing_count": "Critical Missing (Count)",
    })

    show_cols = ["Vendor", "Final Score (%)", "Decision", "Critical Missing (Count)"]
    df = df[show_cols].sort_values("Final Score (%)", ascending=False)

    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No verified leaderboard yet. Upload files and click Run Evaluation.")

st.divider()



# Vendor report view
st.subheader("Vendor Reports")

if not verified_lb:
    st.stop()

vendor_names = [x["vendor_name"] for x in verified_lb]
selected_vendor = st.selectbox("Choose Vendor", vendor_names)

row = next((x for x in verified_lb if x["vendor_name"] == selected_vendor), None)
if not row:
    st.error("Vendor not found in verified leaderboard.")
    st.stop()

verified_file = Path(row["verified_file"])
report = load_json(verified_file)

# Top KPI cards
c1, c2, c3 = st.columns(3)
c1.metric("Verified Score %", report.get("verified_total_score_percent", 0))
c2.metric("Recommendation", report.get("recommendation", "—"))
c3.metric("Critical Missing", len(report.get("critical_missing", [])))

# Summary and Notes
left, right = st.columns([2, 1])

with left:
    st.markdown("### Executive Summary")
    st.write(report.get("final_summary", ""))

with right:
    st.markdown("### Audit Notes")
    for n in report.get("audit_notes", []):
        st.write(f"- {n}")

# Strengths / Gaps
st.markdown("### Strengths vs Gaps")
scol, gcol = st.columns(2)
with scol:
    st.markdown("**Strengths**")
    for s in report.get("strengths", []):
        st.write(f"- {s}")
with gcol:
    st.markdown("**Gaps**")
    for g in report.get("gaps", []):
        st.write(f"- {g}")

# Critical missing
st.markdown("### Critical Missing Requirements")
crit = report.get("critical_missing", [])
if crit:
    for x in crit:
        st.write(f"- {x}")
else:
    st.success("No critical missing requirements")

st.divider()

# Requirements table (audit-friendly)
st.markdown("### Requirements Scoring Table")
rows = report.get("table_rows", [])
if rows:
    df_rows = pd.DataFrame(rows)
    table_cols = [
        "category", "impact_weight", "requirement_text",
        "original_normalized_score", "verified_normalized_score",
        "original_weighted_points", "verified_weighted_points",
        "verified_match", "verified_strength", "confidence",
        "flags", "adjustment_reason_ar", "evidence_quote"
    ]
    df_rows = df_rows[[c for c in table_cols if c in df_rows.columns]]
    st.dataframe(df_rows, use_container_width=True, hide_index=True)
else:
    st.info("No table rows found in this report.")


# Download JSON
st.download_button(
    label="Download Verified Report JSON",
    data=json.dumps(report, ensure_ascii=False, indent=2),
    file_name=f"verified_report_{selected_vendor}.json",
    mime="application/json"
)

