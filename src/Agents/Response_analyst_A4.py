import os
import json
import hashlib
from typing import List
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
    vendor_name: str = Field(..., description="اسم الجهة الموردة")
    proposal_evidence: List[RequirementEvidence]


def build_vendor_search_tool(vector_store):
    @tool("search_vendor_proposal")
    def search_vendor_proposal(query: str) -> str:
        """
        ابحث داخل عرض المورد الحالي حصراً لاستخراج تفاصيل التنفيذ والمنهجية التقنية.
        (FAISS + multilingual-e5-base)
        """
        if vector_store is None:
            return "خطأ: قاعدة بيانات FAISS غير مهيأة."

        # E5 query prefix
        q = f"query: {query}"

        if USE_MMR and hasattr(vector_store, "max_marginal_relevance_search"):
            docs = vector_store.max_marginal_relevance_search(
                q, k=K, fetch_k=FETCH_K, lambda_mult=LAMBDA_MULT
            )
        else:
            docs = vector_store.similarity_search(q, k=K)

        # Return only the content (keep it concise to save tokens)
        return "\n---\n".join([d.page_content for d in docs])

    return search_vendor_proposal


# --- 2. High-Detail Agent Class ---
class ResponseAnalyst:
    def __init__(self, llm, proposal_md_path, requirements_json_path, output_dir="./outputs", vendor_label=None):
        self.llm = llm
        self.md_path = proposal_md_path
        self.requirements_path = requirements_json_path
        self.output_dir = output_dir
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
            print("=== END DEBUG ===\n")
        except Exception as e:
            print("[WARN] Could not read MD head:", e)

        self.vector_store = self._initialize_faiss()
        self.search_tool = build_vendor_search_tool(self.vector_store)

        self.agent = self._create_agent()
        self.task = self._create_task(vendor_label=self.vendor_label)

    def _initialize_faiss(self):
        # E5 embeddings: normalize for cosine similarity
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

        # E5 passage prefix
        for d in docs:
            d.page_content = f"passage: {d.page_content}"

        # Per-vendor isolated index folder (hash full absolute path)
        abs_path = os.path.abspath(self.md_path)
        vid = hashlib.md5(abs_path.encode("utf-8")).hexdigest()[:12]
        index_name = f"faiss_index_{os.path.basename(self.md_path)}_{vid}"
        db_path = os.path.join(self.output_dir, index_name)

        print("\n=== DEBUG: FAISS INDEX PATH ===")
        print("INDEX FOLDER:", db_path)
        print("==============================\n")

        # Load cache if exists
        if os.path.exists(db_path):
            return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

        # Build + save
        vector_db = FAISS.from_documents(docs, embeddings)
        vector_db.save_local(db_path)
        return vector_db

    def _create_agent(self):
        return Agent(
            role="محلل استجابة الموردين (High-Detail RAG Analyst)",
            goal="البحث العميق داخل عروض الموردين لاستخراج تفاصيل فنية ومالية شاملة.",
            backstory="""أنت مدقق بيانات دقيق جداً. تخصصك هو عدم الاكتفاء بالردود السطحية،
بل البحث عن "كيفية التنفيذ" (How-to) والمنهجيات الكاملة. أنت تستخدم البحث الدلالي 
عبر FAISS لجمع أكبر قدر ممكن من السياق لكل متطلب لضمان دقة تقييم الخبير.""",
            llm=self.llm,
            tools=[self.search_tool],
            verbose=True,
            allow_delegation=False
        )

    def _create_task(self, vendor_label=None):
        # ✅ Load requirements
        with open(self.requirements_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        req_objs = data.get("requirements", [])

        # ✅ ADDED: compact numbered list (prevents token blow + forces iteration)
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
- لا تكتب Vendor B أو أي اسم آخر.
"""

        # ✅ ADDED: hard rule to force ALL requirements coverage
        full_coverage_rule = f"""
قواعد إلزامية (لا يمكن مخالفتها):
- لديك {req_count} متطلب. يجب أن تُرجع {req_count} عنصر داخل proposal_evidence (عنصر واحد لكل متطلب بالترتيب).
- لا تترك أي متطلب بدون عنصر.
- إذا لم تجد دليلًا: 
  compliance_status="غير مطابق"
  evidence_text="لا يوجد دليل واضح في عرض المورد"
  technical_methodology="لم يتم العثور على منهجية/تفاصيل تنفيذ لهذا المتطلب داخل العرض."
  extracted_value="غير محدد"
- لكل متطلب: استخدم أداة البحث search_vendor_proposal مرة واحدة على الأقل (أو أكثر عند الحاجة).
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
2. evidence_text: مقتطف قصير 2–4 أسطر فقط.
3. technical_methodology: 6–10 نقاط مفصلة (خطوات، أدوات، معايير، تسليمات، QA).
4. extracted_value: اجمع أي أرقام/تواريخ/SLA/نسب.
5. إذا كان الرد بالإنجليزية: evidence_text كما هو، والشرح بالعربية.
""",
            expected_output="تقرير JSON غني بالتفاصيل التقنية والمالية المقتبسة والمحللة.",
            output_json=VendorExtractionReport,
            output_file=os.path.join(self.output_dir, f"evidence_{os.path.basename(self.md_path)}.json"),
            agent=self.agent
        )
