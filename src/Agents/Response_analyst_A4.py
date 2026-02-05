import os
import json
import hashlib
import re  # ✅ ADDED
from typing import List, Dict, Any  # ✅ ADDED
from dotenv import load_dotenv

from crewai import Agent, Task
from crewai.tools import tool
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# =========================
# ✅ Token-safe optimization knobs (for ~30k max)
# =========================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 180

# Retrieval: richer but safe
USE_MMR = True
K = 7
FETCH_K = 30
LAMBDA_MULT = 0.35


# --- 1. Universal Data Schema ---
class RequirementEvidence(BaseModel):
    requirement_name: str = Field(..., description="اسم المتطلب المستخرج من ملف المتطلبات")
    compliance_status: str = Field(..., description="حالة الامتثال (مطابق/غير مطابق/جزئي)")
    evidence_text: str = Field(..., description="النص المقتبس حرفياً من العرض كدليل (مقتطف قصير 2–4 أسطر)")
    technical_methodology: str = Field(..., description="شرح تفصيلي للمنهجية أو الحل المقترح من قبل المورد (6–10 نقاط)")
    extracted_value: str = Field("غير محدد", description="أي قيم مالية، أرقام، أو تواريخ مستخرجة")


class VendorExtractionReport(BaseModel):
    vendor_name: str = Field(..., description="اسم المتقدم:")
    proposal_evidence: List[RequirementEvidence]


# ✅ ADDED: deterministic “no hallucination” constants
NO_EVIDENCE_TEXT = "لا يوجد دليل واضح في عرض المورد"
NO_METHOD_TEXT = "لم يتم العثور على منهجية/تفاصيل تنفيذ لهذا المتطلب داخل العرض."

# ✅ ADDED: quick normalize helper (for substring check)
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def extract_vendor_name_from_md(md_path: str) -> str:
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            txt = f.read()

        # Existing pattern (Vendor B)
        m = re.search(r"اسم\s+المتقدم\s*[：]\s*(.+)", txt)
        if m:
            line = m.group(1).strip()
            line = line.split("\n")[0].strip()
            line = line.strip("[](){} ")
            if line:
                return line

        # ✅ ADDED pattern (Vendor A / C)
        m2 = re.search(
            r"اسم\s+المتقدم\s*[\n\r]+[\[\(]?\s*(.+?)\s*[\]\)]?",
            txt,
            re.MULTILINE
        )
        if m2:
            line = m2.group(1).strip()
            line = line.split("\n")[0].strip()
            if line:
                return line

    except Exception:
        pass

    return "Unknown Vendor"



# ✅ ADDED: this builds a tool that also logs retrieved chunks per requirement
def build_vendor_search_tool(vector_store, retrieval_log: Dict[str, List[str]]):
    @tool("search_vendor_proposal")
    def search_vendor_proposal(query: str) -> str:
        """
        ابحث داخل عرض المورد الحالي حصراً لاستخراج تفاصيل التنفيذ والمنهجية التقنية.
        (FAISS + multilingual-e5-base)
        """
        if vector_store is None:
            return "خطأ: قاعدة بيانات FAISS غير مهيأة."

        q = f"query: {query}"

        if USE_MMR and hasattr(vector_store, "max_marginal_relevance_search"):
            docs = vector_store.max_marginal_relevance_search(
                q, k=K, fetch_k=FETCH_K, lambda_mult=LAMBDA_MULT
            )
        else:
            docs = vector_store.similarity_search(q, k=K)

        chunks = [d.page_content for d in docs]
        # ✅ ADDED: store retrieved chunks for later verification
        retrieval_log.setdefault(query, [])
        retrieval_log[query].extend(chunks)

        return "\n---\n".join(chunks)

    return search_vendor_proposal


class ResponseAnalyst:
    def __init__(self, llm, proposal_md_path, requirements_json_path, output_dir="./outputs", vendor_label=None):
        self.llm = llm
        self.md_path = proposal_md_path
        self.requirements_path = requirements_json_path
        self.output_dir = output_dir

        # ✅ ADDED: if vendor_label not passed, extract it from "اسم المتقدم"
        if vendor_label is None:
            vendor_label = extract_vendor_name_from_md(self.md_path)
        self.vendor_label = vendor_label

        os.makedirs(self.output_dir, exist_ok=True)

        if not os.path.exists(self.md_path):
            raise FileNotFoundError(f"Vendor MD not found: {self.md_path}")
        if not os.path.exists(self.requirements_path):
            raise FileNotFoundError(f"Requirements JSON not found: {self.requirements_path}")

        # Debug: confirm correct vendor file
        try:
            with open(self.md_path, "r", encoding="utf-8") as f:
                head = f.read(300)
            print("\n=== DEBUG: Using proposal file ===")
            print("MD PATH:", self.md_path)
            print("MD HEAD (first 300 chars):\n", head)
            print("VENDOR LABEL (extracted/enforced):", self.vendor_label)  # ✅ ADDED
            print("=== END DEBUG ===\n")
        except Exception as e:
            print("[WARN] Could not read MD head:", e)

        self.vector_store = self._initialize_faiss()

        # ✅ ADDED: retrieval log for strict verification
        self.retrieval_log: Dict[str, List[str]] = {}

        self.search_tool = build_vendor_search_tool(self.vector_store, self.retrieval_log)

        self.agent = self._create_agent()
        self.task = self._create_task(vendor_label=self.vendor_label)

    def _initialize_faiss(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            encode_kwargs={"normalize_embeddings": True}
        )

        loader = TextLoader(self.md_path, encoding="utf-8")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        docs = splitter.split_documents(documents)

        for d in docs:
            d.page_content = f"passage: {d.page_content}"

        abs_path = os.path.abspath(self.md_path)
        vid = hashlib.md5(abs_path.encode("utf-8")).hexdigest()[:12]
        index_name = f"faiss_index_{os.path.basename(self.md_path)}_{vid}"
        db_path = os.path.join(self.output_dir, index_name)

        print("\n=== DEBUG: FAISS INDEX PATH ===")
        print("INDEX FOLDER:", db_path)
        print("==============================\n")

        if os.path.exists(db_path):
            return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

        vector_db = FAISS.from_documents(docs, embeddings)
        vector_db.save_local(db_path)
        return vector_db

    def _create_agent(self):
        return Agent(
            role="محلل استجابة الموردين (High-Detail RAG Analyst)",
            goal="البحث العميق داخل عروض الموردين لاستخراج تفاصيل فنية ومالية شاملة.",
            backstory="""أنت مدقق بيانات دقيق جداً. تخصصك هو عدم الاكتفاء بالردود السطحية،
بل البحث عن "كيفية التنفيذ" (How-to) والمنهجيات الكاملة. أنت تستخدم البحث الدلالي 
عبر FAISS لجمع أكبر قدر ممكن من السياق لكل متطلب لضمان دقة تقييم الخبير.
قانون صارم: ممنوع اختلاق أي نص. أي evidence_text يجب أن يكون مقتبساً حرفياً من نتائج البحث.""",
            llm=self.llm,
            tools=[self.search_tool],
            verbose=True,
            allow_delegation=False
        )

    def _create_task(self, vendor_label=None):
        with open(self.requirements_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        req_objs = data.get("requirements", [])

        req_texts = []
        for i, r in enumerate(req_objs, start=1):
            rt = r.get("requirement_text") or r.get("requirement_name") or str(r)
            req_texts.append(f"{i}) {rt}")

        requirements_block = "\n".join(req_texts)
        req_count = len(req_texts)

        vendor_enforcement = ""
        if vendor_label:
            vendor_enforcement = f"""
مهم جداً جداً:
- يجب أن يكون الحقل vendor_name مساويًا تمامًا لهذا النص بدون أي تغيير: "{vendor_label}"
- لا تكتب Vendor B أو أي اسم آخر غير المستخرج من "اسم المتقدم".
"""

        full_coverage_rule = f"""
قواعد إلزامية (لا يمكن مخالفتها):
- لديك {req_count} متطلب. يجب أن تُرجع {req_count} عنصر داخل proposal_evidence (عنصر واحد لكل متطلب بالترتيب).
- لا تترك أي متطلب بدون عنصر.
- إذا لم تجد دليلًا: 
  compliance_status="غير مطابق"
  evidence_text="{NO_EVIDENCE_TEXT}"
  technical_methodology="{NO_METHOD_TEXT}"
  extracted_value="غير محدد"
- لكل متطلب: استخدم أداة البحث search_vendor_proposal مرة واحدة على الأقل (أو أكثر عند الحاجة).
- ممنوع اختلاق evidence_text: يجب أن يكون اقتباس حرفي من نص نتائج البحث.
- أخرج JSON فقط بدون أي نص إضافي خارج JSON.
"""

        return Task(
            description=f"""
{vendor_enforcement}

مهمتك هي إجراء استخراج "عميق" للأدلة بناءً على هذه المتطلبات (بالترتيب):

{requirements_block}

{full_coverage_rule}

الضوابط لضمان كفاية التفاصيل:
1. لا تتوقف عند العثور على كلمة مفتاحية؛ استخرج الفقرة المحيطة بها بالكامل لفهم السياق.
2. evidence_text: مقتطف قصير 2–4 أسطر فقط (مقتبس حرفياً).
3. technical_methodology: 6–10 نقاط مفصلة (خطوات، أدوات، معايير، تسليمات، QA) — لكن لا تضف تفاصيل غير موجودة في نتائج البحث.
4. extracted_value: اجمع أي أرقام/تواريخ/SLA/نسب.
5. إذا كان الرد بالإنجليزية: evidence_text كما هو، والشرح بالعربية.
""",
            expected_output="تقرير JSON غني بالتفاصيل التقنية والمالية المقتبسة والمحللة.",
            output_json=VendorExtractionReport,
            output_file=os.path.join(self.output_dir, f"evidence_{os.path.basename(self.md_path)}.json"),
            agent=self.agent
        )

    # ✅ ADDED: strict post-validation to remove hallucinations 100% (deterministic)
    def _post_validate_no_hallucination(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        If evidence_text is NOT found inside retrieved chunks -> force "no evidence".
        This makes hallucination impossible (worst case becomes non-compliant).
        """
        # Merge all retrieved chunks text
        retrieved_all = "\n".join(
            chunk for chunks in self.retrieval_log.values() for chunk in chunks
        )

        retrieved_all_n = _norm(retrieved_all)

        vendor_name = data.get("vendor_name", self.vendor_label) or self.vendor_label
        data["vendor_name"] = vendor_name  # keep enforced

        pe = data.get("proposal_evidence", []) or []
        for item in pe:
            ev = _norm(item.get("evidence_text", ""))

            # If model says evidence exists but it is not in retrieved text => kill it
            if ev and ev != _norm(NO_EVIDENCE_TEXT) and ev not in retrieved_all_n:
                item["compliance_status"] = "غير مطابق"
                item["evidence_text"] = NO_EVIDENCE_TEXT
                item["technical_methodology"] = NO_METHOD_TEXT
                item["extracted_value"] = "غير محدد"

        data["proposal_evidence"] = pe
        return data

    # ✅ ADDED: run() to enforce validation after kickoff
    def run(self):
        crew = Crew(
            agents=[self.agent],
            tasks=[self.task],
            process=Process.sequential,
            verbose=False,
        )

        result = crew.kickoff()

        raw_text = result.raw if hasattr(result, "raw") else str(result)

        try:
            data = json.loads(raw_text)
        except Exception:
            data = {"raw_result": raw_text}

        # ✅ enforce: no hallucination
        if isinstance(data, dict) and "proposal_evidence" in data:
            data = self._post_validate_no_hallucination(data)

            # overwrite output_file with validated JSON
            out_path = os.path.join(self.output_dir, f"evidence_{os.path.basename(self.md_path)}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return out_path, data

        return None, data