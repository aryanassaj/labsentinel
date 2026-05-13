"""
FastAPI server for the Early Cancer Signal Detector.

Endpoints:
  GET  /health                   — model status
  POST /predict                  — single patient risk report
  POST /predict/report           — downloadable PDF risk report
  POST /predict/batch            — batch predictions (list of patients)
  POST /predict/worklist         — batch predict, return only elevated/high patients
  GET  /model/feature-importance — top features by |SHAP| (population level)
  GET  /model/info               — model metadata + CV scores
  GET  /reference-ranges         — lab normal ranges for frontend forms

Run: uvicorn api.main:app --reload --port 8000
"""

import io
import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from functools import lru_cache
from typing import Any, Optional

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from api.schemas import LabPanel, RiskReport, DriverFeature, HealthResponse
from src.features.lab_features import FEATURE_DISPLAY_NAMES, REF_RANGES

MODELS_DIR = Path(__file__).parents[1] / "models_saved"
FRONTEND_DIR = Path(__file__).parents[1] / "frontend"
LOGS_DIR = Path(__file__).parents[1] / "logs"

# ── Audit logger setup (Fix 6) ───────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_audit_logger = logging.getLogger("labsentinel.audit")
_audit_logger.setLevel(logging.INFO)
if not _audit_logger.handlers:
    _audit_handler = RotatingFileHandler(
        LOGS_DIR / "predictions.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    _audit_handler.setFormatter(logging.Formatter("%(message)s"))
    _audit_logger.addHandler(_audit_handler)


# ── API key auth dependency (Fix 3) ─────────────────────────────────────────

def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """FastAPI dependency: validates X-API-Key header when LABSENTINEL_API_KEY is set."""
    required = os.environ.get("LABSENTINEL_API_KEY")
    if required and x_api_key != required:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Pass X-API-Key header.",
        )


app = FastAPI(
    title="Early Cancer Signal Detector",
    description=(
        "Detects pre-diagnostic cancer patterns from standard lab panels "
        "(CBC + metabolic panel) using NHANES-trained XGBoost + SHAP explainability."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lazy model loading ───────────────────────────────────────────────────────

_explainer_cache: dict = {}


def get_explainer(target: str = "any_cancer"):
    """Load explainer, cache in memory."""
    if target not in _explainer_cache:
        try:
            from src.models.explain import CancerRiskExplainer
            _explainer_cache[target] = CancerRiskExplainer(target)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not ready: {exc}. Run python -m src.models.train first.",
            )
    return _explainer_cache[target]


def build_clinical_summary(result: dict, lab_input: LabPanel) -> str:
    """Generate a clinician-friendly one-paragraph summary of the findings."""
    tier = result["risk_tier"]
    pct  = result["risk_percent"]
    age  = lab_input.age
    sex_str = "male" if lab_input.sex == "male" else "female"

    tier_phrases = {
        "low":      "no statistically significant pre-diagnostic cancer signal",
        "moderate": "a moderately elevated pre-diagnostic signal",
        "elevated": "an elevated pre-diagnostic cancer signal",
        "high":     "a high-confidence pre-diagnostic cancer signal",
    }

    drivers = result.get("top_drivers", [])
    # Only include drivers with a proper human-readable display name
    driver_names = [
        d["display_name"] for d in drivers[:5]
        if d["display_name"] != d["name"]  # skip if no proper display name
    ][:3]
    driver_str = ", ".join(driver_names) if driver_names else "no single dominant feature"

    summary = (
        f"A {age:.0f}-year-old {sex_str} patient's routine lab panel shows "
        f"{tier_phrases.get(tier, 'an uncertain signal')} "
        f"(model estimated risk: {pct:.1f}%). "
    )

    if drivers:
        summary += f"The primary laboratory drivers are: {driver_str}. "

    if tier in ("elevated", "high"):
        summary += (
            "Clinical recommendation: Consider targeted cancer screening appropriate for age and sex, "
            "repeat CBC and metabolic panel in 3\u20136 months, and clinical correlation with symptoms and family history. "
        )
    elif tier == "moderate":
        summary += (
            "Clinical recommendation: Ensure age-appropriate cancer screening is current. "
            "Consider monitoring lab trends at the next scheduled visit. "
        )
    else:
        summary += "Routine monitoring per standard guidelines is appropriate. "

    summary += (
        "Note: This tool identifies statistical patterns from the NHANES US population study. "
        "It is a clinical decision support tool \u2014 not a diagnostic test."
    )
    return summary


# ── PDF Report Generation ────────────────────────────────────────────────────

def generate_pdf_report(result: dict, lab_input: LabPanel) -> bytes:
    """Generate a clinical PDF report using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER

    # ── Color palette ──
    DARK_BLUE   = colors.HexColor("#0d6efd")
    LIGHT_GRAY  = colors.HexColor("#f8f9fa")
    MED_GRAY    = colors.HexColor("#6c757d")
    BORDER_GRAY = colors.HexColor("#e2e6ea")
    GREEN_C     = colors.HexColor("#16a34a")
    AMBER_C     = colors.HexColor("#d97706")
    ORANGE_C    = colors.HexColor("#ea580c")
    RED_C       = colors.HexColor("#dc2626")
    DARK_TEXT   = colors.HexColor("#212529")

    tier_colors = {
        "low":      GREEN_C,
        "moderate": AMBER_C,
        "elevated": ORANGE_C,
        "high":     RED_C,
    }

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
    )

    base_styles = getSampleStyleSheet()

    def make_style(base_name, **kw):
        return ParagraphStyle(
            f"cs_{abs(hash(frozenset(kw.items())))}",
            parent=base_styles[base_name],
            **kw,
        )

    title_style   = make_style("Heading1", fontSize=16, textColor=DARK_BLUE, spaceAfter=2)
    sub_style     = make_style("Normal",   fontSize=9,  textColor=MED_GRAY,  spaceAfter=6)
    heading_style = make_style("Heading2", fontSize=11, textColor=DARK_TEXT, spaceBefore=10, spaceAfter=4)
    body_style    = make_style("Normal",   fontSize=9,  textColor=DARK_TEXT, leading=14, spaceAfter=6)
    caption_style = make_style("Normal",   fontSize=8,  textColor=DARK_TEXT, leading=12)
    footer_style  = make_style("Normal",   fontSize=7,  textColor=MED_GRAY,  alignment=TA_CENTER, leading=11)

    now = datetime.now()
    tier = result["risk_tier"]
    pct  = result["risk_percent"]
    tier_color = tier_colors.get(tier, MED_GRAY)
    tier_label = tier.upper()
    patient_id = getattr(lab_input, "patient_id", None) or ""

    story = []

    # ── Header ──
    story.append(Paragraph("LabSentinel Early Cancer Signal Report", title_style))
    story.append(Paragraph(
        f"Generated: {now.strftime('%B %d, %Y')} at {now.strftime('%H:%M:%S')}",
        sub_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER_GRAY, spaceAfter=8))

    # ── Patient section ──
    story.append(Paragraph("Patient Information", heading_style))
    patient_rows = [
        ["Age", f"{lab_input.age:.0f} years"],
        ["Biological Sex", lab_input.sex.capitalize()],
    ]
    if patient_id:
        patient_rows.insert(0, ["Patient ID", str(patient_id)])

    pt_table = Table(patient_rows, colWidths=[45 * mm, 120 * mm])
    pt_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1),  LIGHT_GRAY),
        ("FONTNAME",      (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",     (0, 0), (-1, -1), DARK_TEXT),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(pt_table)
    story.append(Spacer(1, 8))

    # ── Risk Score section ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_GRAY, spaceAfter=6))
    story.append(Paragraph("Risk Assessment", heading_style))

    score_style = make_style(
        "Normal", fontSize=32, textColor=tier_color,
        fontName="Helvetica-Bold", leading=36, spaceAfter=2,
    )
    badge_style = make_style(
        "Normal", fontSize=10, textColor=colors.white,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )

    score_table = Table(
        [[Paragraph(f"{pct:.1f}%", score_style), Paragraph(f"  {tier_label} RISK  ", badge_style)]],
        colWidths=[50 * mm, 50 * mm],
    )
    score_table.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND",   (1, 0), (1, 0),   tier_color),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (1, 0), (1, 0),   6),
        ("BOTTOMPADDING",(1, 0), (1, 0),   6),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 8))

    # ── Clinical Summary ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_GRAY, spaceAfter=6))
    story.append(Paragraph("Clinical Summary", heading_style))
    story.append(Paragraph(result.get("clinical_summary", ""), body_style))
    story.append(Spacer(1, 6))

    # ── Helper: feature table ──
    def add_feature_table(features, caption):
        if not features:
            return
        story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_GRAY, spaceAfter=6))
        story.append(Paragraph(caption, heading_style))
        rows = [[
            Paragraph("<b>Factor</b>",       caption_style),
            Paragraph("<b>Value</b>",         caption_style),
            Paragraph("<b>Direction</b>",     caption_style),
            Paragraph("<b>Clinical Note</b>", caption_style),
        ]]
        for feat in features:
            dir_color = RED_C if feat.get("direction") == "risk" else GREEN_C
            rows.append([
                Paragraph(feat.get("display_name", feat.get("name", "")), caption_style),
                Paragraph(f"{feat.get('value', 0):.2f}", caption_style),
                Paragraph(
                    feat.get("direction", "").capitalize(),
                    make_style("Normal", fontSize=8, textColor=dir_color, fontName="Helvetica-Bold"),
                ),
                Paragraph(feat.get("interpretation", ""), caption_style),
            ])
        tbl = Table(rows, colWidths=[42 * mm, 22 * mm, 24 * mm, 77 * mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  LIGHT_GRAY),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
            ("GRID",          (0, 0), (-1, -1), 0.4, BORDER_GRAY),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 6))

    add_feature_table(result.get("top_drivers",        []), "Risk-Elevating Factors")
    add_feature_table(result.get("protective_factors", []), "Protective Factors")

    # ── Cancer Subtype Signals ──
    subtype_risks = result.get("subtype_risks", {})
    if subtype_risks:
        story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_GRAY, spaceAfter=6))
        story.append(Paragraph("Cancer Subtype Signals", heading_style))
        sub_rows = [[
            Paragraph("<b>Cancer Type</b>", caption_style),
            Paragraph("<b>Signal %</b>",    caption_style),
        ]]
        for name, val in sorted(subtype_risks.items(), key=lambda x: -x[1]):
            sig_color = GREEN_C if val < 10 else AMBER_C if val < 20 else RED_C
            sub_rows.append([
                Paragraph(name.capitalize(), caption_style),
                Paragraph(
                    f"{val:.1f}%",
                    make_style("Normal", fontSize=8, textColor=sig_color, fontName="Helvetica-Bold"),
                ),
            ])
        sub_tbl = Table(sub_rows, colWidths=[80 * mm, 40 * mm])
        sub_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  LIGHT_GRAY),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
            ("GRID",          (0, 0), (-1, -1), 0.4, BORDER_GRAY),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ]))
        story.append(sub_tbl)
        story.append(Spacer(1, 6))

    # ── Missing Labs ──
    missing = result.get("missing_labs", [])
    if missing:
        story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_GRAY, spaceAfter=6))
        story.append(Paragraph("Missing Key Labs", heading_style))
        story.append(Paragraph(
            f"Consider adding: {', '.join(missing)}. These labs improve model accuracy.",
            make_style(
                "Normal", fontSize=9,
                textColor=colors.HexColor("#7d4e00"),
                backColor=colors.HexColor("#fff3cd"),
                leading=14,
            ),
        ))
        story.append(Spacer(1, 6))

    # ── Footer ──
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_GRAY, spaceAfter=6))
    story.append(Paragraph(
        "RESEARCH TOOL \u2014 NOT FOR CLINICAL DIAGNOSIS. "
        "Trained on NHANES 1999\u20132018 (CDC). For clinical decision support only.",
        footer_style,
    ))
    story.append(Paragraph(
        "Model v1.0.0 \u00b7 LabSentinel \u00b7 XGBoost + SHAP \u00b7 53,448 NHANES participants",
        footer_style,
    ))

    doc.build(story)
    return buf.getvalue()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    model_loaded = (MODELS_DIR / "model_any_cancer.json").exists()
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "target": "any_cancer",
    }


@app.post("/predict", response_model=RiskReport, dependencies=[Depends(verify_api_key)])
def predict(lab_panel: LabPanel):
    """
    Generate a cancer risk report for a single patient.

    Accepts a standard CBC + metabolic panel and returns:
    - Risk score (0-100%)
    - Risk tier (low/moderate/elevated/high) — determined by calibrated threshold
    - Top driving features with SHAP values + clinical interpretation
    - Protective factors
    - Clinical summary paragraph

    All lab values are optional (model handles missing data natively).
    At minimum, provide: age, sex, wbc, hgb, plt, albumin.
    """
    exp = get_explainer("any_cancer")
    lab_dict = lab_panel.to_model_dict()

    result = exp.predict(lab_dict)

    # Get subtype risks
    subtype_risks = {}
    for subtype in ["colorectal", "lung", "breast", "prostate", "leukemia", "lymphoma"]:
        model_path = MODELS_DIR / f"model_{subtype}.json"
        if model_path.exists():
            try:
                sub_exp = get_explainer(subtype)
                sub_result = sub_exp.predict(lab_dict)
                subtype_risks[subtype] = round(sub_result["risk_percent"], 1)
            except Exception:
                pass

    # Build clinical summary
    clinical_summary = build_clinical_summary(result, lab_panel)

    # Audit log (Fix 6)
    import datetime as _dt
    sex_str = "male" if lab_panel.sex == "male" else "female"
    _audit_logger.info(
        "%sZ | /predict | patient_id=%s | age=%s | sex=%s | risk_tier=%s | risk_pct=%s",
        _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        lab_panel.patient_id,
        lab_panel.age,
        sex_str,
        result["risk_tier"],
        result["risk_percent"],
    )

    # Convert to response schema (Fix 1: include threshold fields; Fix 2: patient_id)
    return RiskReport(
        risk_score=result["risk_score"],
        risk_percent=result["risk_percent"],
        risk_tier=result["risk_tier"],
        top_drivers=[
            DriverFeature(**d) for d in result["top_drivers"]
        ],
        protective_factors=[
            DriverFeature(**d) for d in result["protective_factors"]
        ],
        missing_labs=result["missing_labs"],
        expected_value=result["expected_value"],
        subtype_risks=subtype_risks,
        clinical_summary=clinical_summary,
        calibrated_threshold=result["threshold"],
        operating_sensitivity=result["operating_sensitivity"],
        operating_specificity=result["operating_specificity"],
        patient_id=lab_panel.patient_id,
    )


@app.post("/predict/report")
def predict_report(lab_panel: LabPanel):
    """
    Generate a cancer risk report and return it as a downloadable PDF clinical document.
    """
    exp = get_explainer("any_cancer")
    lab_dict = lab_panel.to_model_dict()
    result = exp.predict(lab_dict)

    subtype_risks: dict[str, float] = {}
    for subtype in ["colorectal", "lung", "breast", "prostate", "leukemia", "lymphoma"]:
        model_path = MODELS_DIR / f"model_{subtype}.json"
        if model_path.exists():
            try:
                sub_result = get_explainer(subtype).predict(lab_dict)
                subtype_risks[subtype] = round(sub_result["risk_percent"], 1)
            except Exception:
                pass

    result["subtype_risks"] = subtype_risks
    result["clinical_summary"] = build_clinical_summary(result, lab_panel)

    pdf_bytes = generate_pdf_report(result, lab_panel)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=labsentinel_report.pdf"},
    )


@app.post("/predict/worklist")
def predict_worklist(patients: list[LabPanel]) -> dict:
    """
    Run batch prediction and return only elevated/high risk patients.
    Useful for overnight worklist generation — submit all patients with recent labs,
    receive back only those warranting clinical review.
    """
    if len(patients) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size limited to 500 patients per request.",
        )

    exp = get_explainer("any_cancer")
    flagged: list[dict[str, Any]] = []
    total = len(patients)

    for i, patient in enumerate(patients):
        try:
            result = exp.predict(patient.to_model_dict())
            tier = result["risk_tier"]
            if tier in ("elevated", "high"):
                flagged.append({
                    "index": i,
                    "patient_id": getattr(patient, "patient_id", None),
                    "age": patient.age,
                    "sex": patient.sex,
                    "risk_percent": result["risk_percent"],
                    "risk_tier": tier,
                    "top_driver": (
                        result["top_drivers"][0]["display_name"]
                        if result["top_drivers"] else None
                    ),
                })
        except Exception:
            pass  # Skip failed patients in worklist mode

    flagged.sort(key=lambda x: x["risk_percent"], reverse=True)

    return {
        "flagged": len(flagged),
        "total": total,
        "patients": flagged,
    }


@app.post("/predict/batch", dependencies=[Depends(verify_api_key)])
def predict_batch(patients: list[LabPanel]) -> list[dict[str, Any]]:
    """Batch predict for multiple patients. Returns list of risk summaries."""
    if len(patients) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size limited to 500 patients per request.",
        )

    exp = get_explainer("any_cancer")
    results = []
    for i, patient in enumerate(patients):
        try:
            result = exp.predict(patient.to_model_dict())
            row: dict[str, Any] = {
                "index":       i,
                "risk_percent": result["risk_percent"],
                "risk_tier":   result["risk_tier"],
                "top_driver":  result["top_drivers"][0]["display_name"] if result["top_drivers"] else None,
            }
            if patient.patient_id is not None:
                row["patient_id"] = patient.patient_id
            results.append(row)
        except Exception as exc:
            results.append({"index": i, "error": str(exc)})

    return results


@app.post("/parse-labs-pdf")
async def parse_labs_pdf(file: UploadFile = File(...)):
    """
    Extract lab values from an uploaded PDF and return them as a structured JSON
    object ready to auto-fill the frontend form.

    Requires ANTHROPIC_API_KEY environment variable.
    Accepts standard lab report PDFs (Quest, LabCorp, Epic, hospital systems).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ANTHROPIC_API_KEY is not set. Set it in your environment to enable PDF parsing.",
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload a PDF file.",
        )

    # Read file bytes
    pdf_bytes = await file.read()
    if len(pdf_bytes) > 20 * 1024 * 1024:  # 20 MB guard
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PDF exceeds 20 MB size limit.",
        )

    # Extract text from PDF
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        pdf_text = "\n".join(pages_text).strip()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to read PDF: {exc}",
        )

    if not pdf_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from this PDF. It may be a scanned image — try a text-based PDF.",
        )

    # Limit text to first ~6000 chars to stay well within token limits
    pdf_text_truncated = pdf_text[:6000]

    prompt = f"""You are a medical data extraction assistant. Extract laboratory values from the following lab report text and return them as a single JSON object.

Return ONLY a valid JSON object — no explanation, no markdown, no code fences.

The JSON keys and their expected units are:
- age: number (years) — patient age if present
- sex: "male" or "female" — patient sex if present
- patient_id: string — MRN, patient ID, or accession number if present

CBC:
- wbc: number (10^9/L, also called x10^3/uL or K/uL — same value)
- rbc: number (10^12/L, also x10^6/uL or M/uL)
- hgb: number (g/dL)
- hct: number (%) — hematocrit
- plt: number (10^9/L, same as K/uL)
- mcv: number (fL)
- mch: number (pg)
- mchc: number (g/dL)
- rdw: number (%)
- neut_pct: number (%) — neutrophil percentage
- lymph_pct: number (%) — lymphocyte percentage
- mono_pct: number (%) — monocyte percentage
- eos_pct: number (%) — eosinophil percentage
- baso_pct: number (%) — basophil percentage

Metabolic / CMP:
- glucose: number (mg/dL)
- bun: number (mg/dL) — blood urea nitrogen
- creatinine: number (mg/dL)
- sodium: number (mmol/L or mEq/L)
- potassium: number (mmol/L or mEq/L)
- chloride: number (mmol/L or mEq/L)
- co2: number (mmol/L) — bicarbonate/CO2
- calcium: number (mg/dL)
- total_protein: number (g/dL)
- albumin: number (g/dL)
- bilirubin: number (mg/dL) — total bilirubin
- alt: number (U/L)
- ast: number (U/L)
- alkphos: number (U/L) — alkaline phosphatase
- ldh: number (U/L)

Iron studies & inflammation:
- ferritin: number (ng/mL)
- iron: number (ug/dL)
- tibc: number (ug/dL)
- crp: number (mg/L) — C-reactive protein

Only include fields where you found a clear numeric value in the report. Omit fields that are missing, unclear, or non-numeric. Do not guess or infer values.

Lab report text:
---
{pdf_text_truncated}
---

JSON output:"""

    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI extraction failed: {exc}",
        )

    # Parse JSON — strip markdown fences if model added them
    raw_clean = raw.strip()
    if raw_clean.startswith("```"):
        lines = raw_clean.split("\n")
        raw_clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        extracted = json.loads(raw_clean)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"AI returned invalid JSON: {exc}. Raw: {raw_clean[:200]}",
        )

    # Sanitize: only allow expected keys, numeric values where expected
    NUMERIC_KEYS = {
        "age", "wbc", "rbc", "hgb", "hct", "plt", "mcv", "mch", "mchc", "rdw",
        "neut_pct", "lymph_pct", "mono_pct", "eos_pct", "baso_pct",
        "glucose", "bun", "creatinine", "sodium", "potassium", "chloride",
        "co2", "calcium", "total_protein", "albumin", "bilirubin",
        "alt", "ast", "alkphos", "ldh", "ferritin", "iron", "tibc", "crp",
    }
    STRING_KEYS = {"sex", "patient_id"}

    clean: dict[str, Any] = {}
    for key, val in extracted.items():
        if key in NUMERIC_KEYS:
            try:
                clean[key] = float(val)
            except (TypeError, ValueError):
                pass
        elif key in STRING_KEYS:
            if isinstance(val, str) and val.strip():
                clean[key] = val.strip()

    return {"extracted": clean, "fields_found": len(clean)}


@app.get("/model/feature-importance")
def feature_importance():
    """Top features by mean |SHAP| across the NHANES population."""
    try:
        from src.models.explain import load_population_importance
        imp = load_population_importance("any_cancer")
        top = imp.head(20)
        return {
            "features": [
                {
                    "name": name,
                    "display_name": FEATURE_DISPLAY_NAMES.get(name, name),
                    "importance": round(float(val), 5),
                }
                for name, val in top.items()
            ]
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/model/info")
def model_info():
    """Model metadata, CV results, and description."""
    cv_path = MODELS_DIR / "cv_results.json"
    cv_results = {}
    if cv_path.exists():
        with open(cv_path) as f:
            cv_results = json.load(f)

    version_info = {}
    version_path = MODELS_DIR / "model_version.json"
    if version_path.exists():
        try:
            with open(version_path) as f:
                version_info = json.load(f)
        except Exception:
            pass

    return {
        "version": "1.0.0",
        "model_type": "XGBoost (gradient boosted trees)",
        "training_data": "NHANES 1999-2018 (CDC National Health and Nutrition Examination Survey)",
        "n_cycles": 10,
        "approx_participants": "53,448",
        "outcome": "Cancer diagnosis (self-report) + cancer-specific mortality (NCHS linkage)",
        "features": "CBC + CMP + derived inflammatory indices (NLR, PLR, SII, GPS)",
        "explainability": "SHAP TreeExplainer — per-patient feature attribution",
        "cv_results": cv_results,
        "version_info": version_info,
        "limitations": [
            "NHANES is primarily cross-sectional — true pre-diagnostic window varies",
            "Self-reported cancer diagnoses may have recall bias",
            "Mortality linkage does not capture non-fatal cancers after 2019",
            "PSA, CA-125, CEA, and other tumor markers are not included",
            "Model validated on US adult population — may not generalize internationally",
        ],
        "references": [
            "Templeton et al. (2014). Prognostic role of NLR in solid tumors. J Clin Oncol.",
            "Proctor et al. (2012). Glasgow Prognostic Score. Cancer Treatment Reviews.",
            "NHANES: https://www.cdc.gov/nchs/nhanes/",
            "NCHS Mortality Linkage: https://www.cdc.gov/nchs/data-linkage/mortality.htm",
        ],
    }


@app.get("/reference-ranges")
def reference_ranges():
    """Normal reference ranges for all labs, by sex."""
    return {
        "male": {
            "wbc":       {"lo": 4.0,  "hi": 11.0,  "unit": "10⁹/L"},
            "rbc":       {"lo": 4.5,  "hi": 5.9,   "unit": "10¹²/L"},
            "hgb":       {"lo": 13.5, "hi": 17.5,  "unit": "g/dL"},
            "hct":       {"lo": 41.0, "hi": 53.0,  "unit": "%"},
            "plt":       {"lo": 150,  "hi": 400,   "unit": "10⁹/L"},
            "mcv":       {"lo": 80,   "hi": 100,   "unit": "fL"},
            "albumin":   {"lo": 3.5,  "hi": 5.0,   "unit": "g/dL"},
            "crp":       {"lo": 0,    "hi": 3.0,   "unit": "mg/L"},
            "ferritin":  {"lo": 30,   "hi": 400,   "unit": "ng/mL"},
            "nlr_normal": {"lo": 1.0, "hi": 3.0,   "unit": "ratio"},
        },
        "female": {
            "wbc":       {"lo": 4.0,  "hi": 11.0,  "unit": "10⁹/L"},
            "rbc":       {"lo": 4.0,  "hi": 5.2,   "unit": "10¹²/L"},
            "hgb":       {"lo": 12.0, "hi": 15.5,  "unit": "g/dL"},
            "hct":       {"lo": 36.0, "hi": 46.0,  "unit": "%"},
            "plt":       {"lo": 150,  "hi": 400,   "unit": "10⁹/L"},
            "mcv":       {"lo": 80,   "hi": 100,   "unit": "fL"},
            "albumin":   {"lo": 3.5,  "hi": 5.0,   "unit": "g/dL"},
            "crp":       {"lo": 0,    "hi": 3.0,   "unit": "mg/L"},
            "ferritin":  {"lo": 13,   "hi": 150,   "unit": "ng/mL"},
            "nlr_normal": {"lo": 1.0, "hi": 3.0,   "unit": "ratio"},
        },
    }


# ── Serve frontend ────────────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
