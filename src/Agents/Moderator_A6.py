import os, json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process



# Output Schemas (Streamlit suitable )
class VerifiedRequirementScore(BaseModel):
    requirement_text: str
    category: str
    impact_weight: int

    original_normalized_score: float
    verified_normalized_score: float
    original_weighted_points: float
    verified_weighted_points: float

    evidence_quote: str
    verified_match: str  # direct | indirect | off_topic
    verified_strength: str  # strong | medium | weak
    confidence: str  # High | Medium | Low
    flags: List[str] = Field(default_factory=list)

    adjustment_reason_ar: str = ""


class ScoreAdjustment(BaseModel):
    requirement_text: str
    from_score: float
    to_score: float
    reason_ar: str
    flags: List[str] = Field(default_factory=list)


class VerifiedVendorReport(BaseModel):
    vendor_name: str

    original_total_score_percent: float
    verified_total_score_percent: float
    max_possible_points: float

    original_total_weighted_points: float
    verified_total_weighted_points: float

    recommendation: str  # مؤهل فنيًا | يحتاج استيضاح | غير مؤهل
    final_summary: str   # executive summary (Arabic)
    audit_notes: List[str] = Field(default_factory=list)

    vendor_profile_snapshot: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    critical_missing: List[str] = Field(default_factory=list)

    # Table-like section for Streamlit
    table_rows: List[VerifiedRequirementScore] = Field(default_factory=list)
    score_adjustments: List[ScoreAdjustment] = Field(default_factory=list)


# -----------------------------
# Utilities (domain-agnostic)
# -----------------------------
def _s(x: Any) -> str:
    return x if isinstance(x, str) else ""


def _missing(text: str) -> bool:
    t = " ".join(_s(text).split()).strip()
    return (not t) or ("لا يوجد دليل واضح" in t)


def _recompute_totals(rows: List[VerifiedRequirementScore]) -> Dict[str, float]:
    max_points = sum(float(r.impact_weight) for r in rows) or 0.0
    total = sum(float(r.verified_normalized_score) * float(r.impact_weight) for r in rows) or 0.0
    pct = (total / max_points) * 100.0 if max_points else 0.0
    return {
        "max_points": round(max_points, 3),
        "total_points": round(total, 3),
        "pct": round(pct, 2),
    }


def _recommend(score_pct: float, critical_missing_count: int) -> str:
    if score_pct >= 75 and critical_missing_count == 0:
        return "مؤهل فنيًا"
    if score_pct >= 55:
        return "يحتاج استيضاح"
    return "غير مؤهل"


# -----------------------------
# Moderator (reviews ONE vendor)
# -----------------------------
class VendorModerator:
    """
    General Moderator:
    - One LLM call (batch) to verify match/strength/confidence and suggest score changes.
    - Deterministic enforcement:
        * missing evidence => verified score must be 0
        * totals recomputed in code
    """

    def __init__(self, llm, apply_adjustments: bool = True):
        self.llm = llm
        self.apply_adjustments = apply_adjustments
        self.agent = Agent(
            role="مدقق مستقل ومحكم درجات",
            goal="تدقيق اتساق الدرجة مع الدليل وإنتاج تقرير عربي واضح ومناسب للعرض.",
            backstory="مدقق صارم ومحايد. يعمل وفق Rubric عام يصلح لأي مجال. لا يختلق أدلة.",
            llm=self.llm,
            allow_delegation=False,
            verbose=False,
        )

    def verify(self, golden: Dict[str, Any], evidence: Dict[str, Any], scorecard: Dict[str, Any]) -> Dict[str, Any]:
        vendor_name = scorecard.get("vendor_name") or evidence.get("vendor_name") or "Unknown Vendor"

        # Build batch bundle for LLM (short)
        ev_map = {e.get("requirement_name", ""): e for e in (evidence.get("proposal_evidence", []) or [])}
        bundle = []
        for r in (scorecard.get("scored_requirements", []) or []):
            rt = _s(r.get("requirement_text", ""))
            ev = ev_map.get(rt, {})
            bundle.append({
                "requirement_text": rt,
                "impact_weight": int(r.get("impact_weight", 1)),
                "category": _s(r.get("category", "Unknown")),
                "original_normalized_score": float(r.get("normalized_score", 0.0)),
                "evidence_quote": _s(r.get("evidence_quote", "")),
                "original_evidence_text": _s(ev.get("evidence_text", "")),
            })

        task = Task(
            description=f"""
أنت مدقق مستقل. راجع كل متطلب حسب النص والاقتباس فقط (بدون معرفة خارجية).

لكل عنصر في inputs:
- verified_match: direct|indirect|off_topic
- verified_strength: strong|medium|weak
- confidence: High|Medium|Low
- flags: قائمة قصيرة (missing_evidence, generic, mismatch, inflated_score, unclear)
- suggested_normalized_score: ضع رقم 0..1 فقط إذا ترى أن الدرجة الأصلية غير عادلة بوضوح، وإلا اتركها null
- adjustment_reason_ar: سبب مختصر إذا اقترحت تعديل

قواعد:
- لا تختلق أدلة.
- إذا الاقتباس أو النص الأصلي فارغ أو فيه "لا يوجد دليل واضح" => match=off_topic, strength=weak, confidence=Low, واقترح score=0 إذا الأصل >0.

بعدها أعطِ:
- strengths (3-5)
- gaps (3-5)
- audit_notes (3-6)
- vendor_profile_snapshot (2-4 ملاحظات قصيرة من طبيعة الدليل فقط)
- final_summary (6-10 أسطر عربية رسمية)

inputs:
{json.dumps(bundle, ensure_ascii=False)}

أخرج JSON فقط:
{{
 "per_requirement":[{{"requirement_text":"..","verified_match":"..","verified_strength":"..","confidence":"..","flags":[..],"suggested_normalized_score":null,"adjustment_reason_ar":""}}],
 "vendor_profile_snapshot":[..],
 "strengths":[..],
 "gaps":[..],
 "audit_notes":[..],
 "final_summary":".."
}}
""".strip(),
            expected_output="Moderator JSON",
            agent=self.agent
        )

        out = Crew(agents=[self.agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()
        raw = out.raw if hasattr(out, "raw") else str(out)

        try:
            mod = json.loads(raw)
        except Exception:
            mod = {
                "per_requirement": [],
                "vendor_profile_snapshot": [],
                "strengths": [],
                "gaps": [],
                "audit_notes": ["فشل استخراج JSON من مخرجات المدقق."],
                "final_summary": raw
            }

        per_map = {x.get("requirement_text", ""): x for x in (mod.get("per_requirement", []) or []) if isinstance(x, dict)}

        rows: List[VerifiedRequirementScore] = []
        adjustments: List[ScoreAdjustment] = []
        critical_missing: List[str] = []

        for r in (scorecard.get("scored_requirements", []) or []):
            rt = _s(r.get("requirement_text", ""))
            w = int(r.get("impact_weight", 1))

            orig = float(r.get("normalized_score", 0.0))
            orig_wp = round(orig * w, 3)

            mq = _s(r.get("evidence_quote", ""))
            ev_text = _s(ev_map.get(rt, {}).get("evidence_text", ""))

            m = per_map.get(rt, {})
            v_match = _s(m.get("verified_match", "indirect"))
            v_strength = _s(m.get("verified_strength", "medium"))
            conf = _s(m.get("confidence", "Medium"))
            flags = list(dict.fromkeys((m.get("flags", []) or [])))

            suggested = m.get("suggested_normalized_score", None)
            reason = _s(m.get("adjustment_reason_ar", ""))

            # Deterministic enforcement: missing evidence => score 0
            if (_missing(mq) or _missing(ev_text)) and orig > 0:
                suggested = 0.0
                reason = "لا يوجد دليل واضح/مقتبس، لذلك يجب أن تكون الدرجة صفر."
                if "missing_evidence" not in flags:
                    flags.append("missing_evidence")

            verified = orig
            if self.apply_adjustments and suggested is not None:
                verified = float(suggested)
                adjustments.append(ScoreAdjustment(
                    requirement_text=rt,
                    from_score=round(orig, 3),
                    to_score=round(verified, 3),
                    reason_ar=reason or "تم تعديل الدرجة وفق قوة/ملاءمة الدليل.",
                    flags=flags
                ))

            verified_wp = round(verified * w, 3)

            if w >= 4 and (v_match == "off_topic" or v_strength == "weak" or verified <= 0.25):
                # "critical" here means high impact + weak/off-topic/low score
                critical_missing.append(rt)

            rows.append(VerifiedRequirementScore(
                requirement_text=rt,
                category=_s(r.get("category", "Unknown")),
                impact_weight=w,
                original_normalized_score=round(orig, 3),
                verified_normalized_score=round(verified, 3),
                original_weighted_points=orig_wp,
                verified_weighted_points=verified_wp,
                evidence_quote=mq,
                verified_match=v_match,
                verified_strength=v_strength,
                confidence=conf,
                flags=flags,
                adjustment_reason_ar=reason if (self.apply_adjustments and suggested is not None) else ""
            ))

        totals = _recompute_totals(rows)

        original_total_wp = float(scorecard.get("total_weighted_points", sum(x.original_weighted_points for x in rows)))
        original_pct = float(scorecard.get("total_score_percent", (original_total_wp / (totals["max_points"] or 1)) * 100.0))

        rec = _recommend(totals["pct"], len(critical_missing))

        report = VerifiedVendorReport(
            vendor_name=vendor_name,
            original_total_score_percent=round(original_pct, 2),
            verified_total_score_percent=totals["pct"],
            max_possible_points=totals["max_points"],
            original_total_weighted_points=round(original_total_wp, 3),
            verified_total_weighted_points=totals["total_points"],
            recommendation=rec,
            final_summary=_s(mod.get("final_summary", "")),
            audit_notes=mod.get("audit_notes", []) or [],
            vendor_profile_snapshot=mod.get("vendor_profile_snapshot", []) or [],
            strengths=mod.get("strengths", []) or [],
            gaps=mod.get("gaps", []) or [],
            critical_missing=list(dict.fromkeys(critical_missing)),
            table_rows=rows,
            score_adjustments=adjustments
        ).model_dump()

        return report
