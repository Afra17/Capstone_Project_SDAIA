import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, Process

load_dotenv()


STATUS_TO_SCORE = {
    "مطابق": 1.0,
    "مطبق": 1.0,         
    "جزئي": 0.6,
    "جزئياً": 0.6,
    "غير مطابق": 0.0,
    "غير واضح": 0.0,
}

NO_EVIDENCE_PHRASES = [
    "لا يوجد دليل واضح",
    "لم يتم العثور",
    "لا يوجد دليل",
]


# Evidence quality multiplier 
QUALITY_MULTIPLIER = {
    ("strong", "direct"): 1.00,
    ("medium", "direct"): 0.90,
    ("weak", "direct"): 0.75,

    ("strong", "indirect"): 0.85,
    ("medium", "indirect"): 0.75,
    ("weak", "indirect"): 0.60,

    # off-topic should be penalized hard regardless of strength claims
    ("strong", "off_topic"): 0.50,
    ("medium", "off_topic"): 0.50,
    ("weak", "off_topic"): 0.50,
}


def _norm_strength(s: str) -> str:
    s = (s or "").strip().lower()
    if s in ["strong", "قوي", "مرتفع"]:
        return "strong"
    if s in ["medium", "متوسط"]:
        return "medium"
    return "weak"


def _norm_match(m: str) -> str:
    m = (m or "").strip().lower()
    if m in ["direct", "مباشر"]:
        return "direct"
    if m in ["indirect", "غير مباشر"]:
        return "indirect"
    return "off_topic"


# Output Schemas
class RequirementScore(BaseModel):
    requirement_text: str
    category: str
    impact_weight: int

    compliance_status: str

    # computed deterministically to make it more reviewable
    normalized_score: float = 0.0
    weighted_points: float = 0.0


    evidence_quote: str = Field(..., description="اقتباس 2–4 أسطر من evidence_text")
    scoring_reason: List[str] = Field(..., description="2–4 نقاط قصيرة تشرح سبب التقييم بالاستناد للدليل فقط")
    risk_note: str = Field("", description="ملاحظة مخاطر فقط إذا الدليل ضعيف/عام/مفقود")

    #LLM classification for quality (converted deterministically)
    evidence_strength: str = Field(..., description="weak | medium | strong")
    evidence_match: str = Field(..., description="direct | indirect | off_topic")


class VendorScorecard(BaseModel):
    vendor_name: str

    # computed deterministically
    total_score_percent: float = 0.0
    total_weighted_points: float = 0.0
    max_possible_points: float = 0.0

    # LLM summaries
    strengths: List[str] = Field(default_factory=list, description="Top 3 strengths")
    gaps: List[str] = Field(default_factory=list, description="Top 3 gaps")
    critical_missing: List[str] = Field(default_factory=list, description="requirements with 0 score and high weight")

    scored_requirements: List[RequirementScore]


class VendorScoringAgent:
    def __init__(
        self,
        llm,
        golden_template_path: str,
        vendor_evidence_path: str,
        output_dir: str,
        critical_weight_threshold: int = 4,
    ):
        self.llm = llm
        self.golden_path = golden_template_path
        self.vendor_path = vendor_evidence_path
        self.output_dir = output_dir
        self.critical_weight_threshold = critical_weight_threshold

        os.makedirs(self.output_dir, exist_ok=True)

        self.agent = self._create_agent()
        self.task = self._create_task()

    def _load_inputs(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        with open(self.golden_path, "r", encoding="utf-8") as f:
            golden = json.load(f)
        with open(self.vendor_path, "r", encoding="utf-8") as f:
            vendor = json.load(f)
        return golden, vendor

    def _create_agent(self):
        return Agent(
            role="خبير تقييم عروض منافسات (Vendor Scorer)",
            goal="إنتاج بطاقة تقييم قابلة للتدقيق بناءً على الأدلة، مع تمييز جودة الدليل وملاءمته للمتطلب.",
            backstory=(
                "أنت مقيم مناقصات صارم. لا تُدخل معلومات غير موجودة في الدليل. "
                "تفرّق بين: (دليل مباشر قوي) و(دليل عام/تسويقي) و(دليل خارج الموضوع). "
                "لا تحسب الدرجات النهائية—سيتم احتسابها برمجياً بشكل حتمي."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def _create_task(self):
        golden, vendor = self._load_inputs()

        vendor_name = vendor.get("vendor_name", "Unknown Vendor")
        golden_reqs = golden.get("requirements", [])
        vendor_evs = vendor.get("proposal_evidence", [])

        # Evidence mapped by requirement_name (usually exact)
        ev_map = {ev.get("requirement_name"): ev for ev in vendor_evs}

        # Build compact bundle for LLM
        bundle = []
        for req in golden_reqs:
            r_text = req.get("requirement_text", "")
            ev = ev_map.get(
                r_text,
                {
                    "compliance_status": "غير واضح",
                    "evidence_text": "لا يوجد دليل واضح في عرض المورد",
                    "technical_methodology": "",
                    "extracted_value": "غير محدد",
                },
            )

            bundle.append(
                {
                    "requirement_text": r_text,
                    "category": req.get("category", "Unknown"),
                    "impact_weight": req.get("impact_weight", 1),
                    "strategic_reasoning": req.get("reasoning", ""),
                    "vendor_evidence": {
                        "compliance_status": ev.get("compliance_status", "غير واضح"),
                        "evidence_text": ev.get("evidence_text", ""),
                        "technical_methodology": ev.get("technical_methodology", ""),
                        "extracted_value": ev.get("extracted_value", "غير محدد"),
                    },
                }
            )

        bundle_json = json.dumps(bundle, ensure_ascii=False)

        description = f"""
        أنت محكم تقييم عروض مناقصات حكومية صارم.

        تقيّم المورد: "{vendor_name}"

        لكل متطلب أخرج فقط الحقول التالية (بدون حساب درجات رقمية):

        1) compliance_status (كما ورد في vendor_evidence إن وجد).
        2) evidence_quote: اقتبس من evidence_text فقط.
        3) scoring_reason: 2–4 نقاط عربية تعتمد على الدليل حصراً.
        4) risk_note: اكتبها فقط إذا الدليل عام/ضعيف/تسويقي/مفقود.
        5) evidence_strength: weak | medium | strong
        6) evidence_match: direct | indirect | off_topic


        ──────────────── قواعد مطلقة ────────────────

        ❗ لا تغيّر requirement_text إطلاقاً.  
        ❗ لا تختلق أدلة.  
        ❗ لا تحسب أي درجات رقمية.

        إذا evidence_text فارغ أو يحتوي "لا يوجد دليل واضح":
        → evidence_strength = weak  
        → evidence_match = off_topic  
        → واكتب risk_note.


        ──────────────── تصنيف evidence_match ────────────────

        A) متطلبات ISO/IEC 27001:
        direct فقط إذا ورد ISO أو 27001 صراحة.
        غير ذلك = indirect.

        B) متطلبات ISO/IEC 9001:
        direct فقط إذا ورد ISO أو 9001 صراحة.
        غير ذلك = indirect.

        C) وثيقة مراجعة أمنية + موقع تخزين البيانات:
        direct فقط إذا ذُكرت (وثيقة/تقرير) + (موقع التخزين/داخل المملكة/region/datacenter).
        بدون موقع = indirect.

        D) تقييم مصادر البيانات والتكامل:
        direct فقط إذا ذُكرت APIs / ETL / Connectors / REST / GraphQL / DB links.
        عبارات عامة = indirect.


        ──────────────── تصنيف evidence_strength ────────────────

        strong = خطوات متعددة + أدوات + معايير + تفاصيل تنفيذ.  
        medium = دعم جزئي بدون تفاصيل كافية.  
        weak = جملة عامة / وعد تسويقي / لا يوجد دليل.


        ──────────────── المدخلات ────────────────
        {bundle_json}

        أخرج JSON مطابق للـ schema فقط.
        """

        return Task(
            description=description,
            expected_output="VendorScorecard JSON بدون حساب الدرجات النهائية (سيتم احتسابها برمجياً).",
            output_json=VendorScorecard,
            output_file=os.path.join(self.output_dir, f"scorecard_{os.path.basename(self.vendor_path)}"),
            agent=self.agent,
        )

    @staticmethod
    def _is_no_evidence(text: str) -> bool:
        if not text:
            return True
        for p in NO_EVIDENCE_PHRASES:
            if p in text:
                return True
        return False

    @staticmethod
    def _normalize_status(status: str, evidence_text: str) -> str:
        s = (status or "").strip()
        # If evidence clearly says no evidence -> force non-compliant
        if VendorScoringAgent._is_no_evidence(evidence_text):
            return "غير مطابق"
        # Normalize common synonym
        if s == "مطبق":
            return "مطابق"
        # Unknown labels
        if s not in STATUS_TO_SCORE:
            return "غير واضح"
        return s

    def _compute_scores(self, scorecard: Dict[str, Any]) -> Dict[str, Any]:
        scored = scorecard.get("scored_requirements", [])

        total_weighted = 0.0
        max_possible = 0.0
        critical_missing = []

        for r in scored:
            w = int(r.get("impact_weight", 1))
            status = r.get("compliance_status", "غير واضح")
            evidence_quote = r.get("evidence_quote", "")

            # Normalize status using evidence
            status = self._normalize_status(status, evidence_quote)
            r["compliance_status"] = status

            base = STATUS_TO_SCORE.get(status, 0.0)

            strength = _norm_strength(r.get("evidence_strength", "weak"))
            match = _norm_match(r.get("evidence_match", "off_topic"))
            mult = QUALITY_MULTIPLIER.get((strength, match), 0.6)

            # Hard override: no evidence => 0
            if self._is_no_evidence(evidence_quote):
                base = 0.0
                mult = 0.0

            final_norm = base * mult
            weighted = final_norm * w

            r["normalized_score"] = round(final_norm, 3)
            r["weighted_points"] = round(weighted, 3)

            total_weighted += weighted
            max_possible += w

            if final_norm == 0.0 and w >= self.critical_weight_threshold:
                critical_missing.append(r.get("requirement_text", ""))

        total_pct = (total_weighted / max_possible) * 100 if max_possible else 0.0

        scorecard["total_weighted_points"] = round(total_weighted, 3)
        scorecard["max_possible_points"] = round(max_possible, 3)
        scorecard["total_score_percent"] = round(total_pct, 2)
        scorecard["critical_missing"] = [x for x in critical_missing if x]

        # Ensure strengths/gaps exist even if model omitted them
        scorecard.setdefault("strengths", [])
        scorecard.setdefault("gaps", [])

        return scorecard

    def run(self):
        crew = Crew(
            agents=[self.agent],
            tasks=[self.task],
            process=Process.sequential,
            verbose=False,
        )

        result = crew.kickoff()

        # Extract raw text
        if hasattr(result, "raw") and isinstance(result.raw, str):
            raw_text = result.raw
        else:
            raw_text = str(result)

        # Parse JSON
        try:
            data = json.loads(raw_text)
        except Exception:
            data = {"raw_result": raw_text}

        # Compute deterministic scoring if valid
        if isinstance(data, dict) and "scored_requirements" in data and "vendor_name" in data:
            data = self._compute_scores(data)

        out_path = os.path.join(self.output_dir, f"scorecard_{os.path.basename(self.vendor_path)}")
        return out_path, data
