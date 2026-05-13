"""
Pydantic schemas for the Cancer Risk API.
"""

from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class LabPanel(BaseModel):
    """Input: patient demographics + lab values."""

    # Patient identifier (optional — returned in response for correlation)
    patient_id: Optional[str] = Field(None, description="Optional patient identifier — returned in response for correlation")

    # Demographics
    age: float = Field(..., ge=18, le=120, description="Patient age in years")
    sex: Literal["male", "female"] = Field(..., description="Biological sex")

    # CBC — Complete Blood Count
    wbc:       Optional[float] = Field(None, ge=0.1, le=100,   description="WBC (10⁹/L)")
    rbc:       Optional[float] = Field(None, ge=1.0, le=10.0,  description="RBC (10¹²/L)")
    hgb:       Optional[float] = Field(None, ge=3.0, le=22.0,  description="Hemoglobin (g/dL)")
    hct:       Optional[float] = Field(None, ge=10,  le=65,    description="Hematocrit (%)")
    plt:       Optional[float] = Field(None, ge=10,  le=1500,  description="Platelets (10⁹/L)")
    mcv:       Optional[float] = Field(None, ge=50,  le=130,   description="MCV (fL)")
    mch:       Optional[float] = Field(None, ge=10,  le=50,    description="MCH (pg)")
    mchc:      Optional[float] = Field(None, ge=20,  le=40,    description="MCHC (g/dL)")
    rdw:       Optional[float] = Field(None, ge=5,   le=30,    description="RDW (%)")
    neut_pct:  Optional[float] = Field(None, ge=0,   le=100,   description="Neutrophils (%)")
    lymph_pct: Optional[float] = Field(None, ge=0,   le=100,   description="Lymphocytes (%)")
    mono_pct:  Optional[float] = Field(None, ge=0,   le=50,    description="Monocytes (%)")
    eos_pct:   Optional[float] = Field(None, ge=0,   le=60,    description="Eosinophils (%)")
    baso_pct:  Optional[float] = Field(None, ge=0,   le=10,    description="Basophils (%)")

    # CMP — Comprehensive Metabolic Panel
    glucose:       Optional[float] = Field(None, ge=20,   le=600,   description="Glucose (mg/dL)")
    bun:           Optional[float] = Field(None, ge=1,    le=200,   description="BUN (mg/dL)")
    creatinine:    Optional[float] = Field(None, ge=0.1,  le=30,    description="Creatinine (mg/dL)")
    sodium:        Optional[float] = Field(None, ge=110,  le=170,   description="Sodium (mmol/L)")
    potassium:     Optional[float] = Field(None, ge=2,    le=8,     description="Potassium (mmol/L)")
    chloride:      Optional[float] = Field(None, ge=80,   le=130,   description="Chloride (mmol/L)")
    co2:           Optional[float] = Field(None, ge=10,   le=40,    description="CO2/Bicarb (mmol/L)")
    calcium:       Optional[float] = Field(None, ge=5,    le=15,    description="Calcium (mg/dL)")
    total_protein: Optional[float] = Field(None, ge=3,    le=12,    description="Total Protein (g/dL)")
    albumin:       Optional[float] = Field(None, ge=1,    le=6,     description="Albumin (g/dL)")
    bilirubin:     Optional[float] = Field(None, ge=0.1,  le=30,    description="Total Bilirubin (mg/dL)")
    alt:           Optional[float] = Field(None, ge=1,    le=3000,  description="ALT (U/L)")
    ast:           Optional[float] = Field(None, ge=1,    le=3000,  description="AST (U/L)")
    alkphos:       Optional[float] = Field(None, ge=1,    le=3000,  description="Alkaline Phosphatase (U/L)")
    uric_acid:     Optional[float] = Field(None, ge=0.5,  le=20,    description="Uric Acid (mg/dL)")
    ldh:           Optional[float] = Field(None, ge=50,   le=5000,  description="LDH (U/L)")
    phosphorus:    Optional[float] = Field(None, ge=0.5,  le=10,    description="Phosphorus (mg/dL)")

    # Iron studies
    ferritin: Optional[float] = Field(None, ge=1,    le=10000, description="Ferritin (ng/mL)")
    iron:     Optional[float] = Field(None, ge=10,   le=500,   description="Serum Iron (μg/dL)")
    tibc:     Optional[float] = Field(None, ge=100,  le=600,   description="TIBC (μg/dL)")

    # Inflammation
    crp: Optional[float] = Field(None, ge=0.01, le=300, description="C-Reactive Protein (mg/L)")

    def to_model_dict(self) -> dict:
        """Convert to canonical dict used by the model."""
        d = self.model_dump()
        d["is_male"] = 1.0 if self.sex == "male" else 0.0
        return d

    @field_validator("neut_pct", "lymph_pct", "mono_pct", "eos_pct", "baso_pct", mode="before")
    @classmethod
    def check_differential_sum(cls, v):
        return v  # Individual validation; cross-field sum check done in API


class DriverFeature(BaseModel):
    """A single SHAP feature contribution."""
    name:           str
    display_name:   str
    value:          float
    shap_value:     float
    direction:      Literal["risk", "protective"]
    interpretation: str


class RiskReport(BaseModel):
    """Output: full cancer risk report."""
    risk_score:              float = Field(..., description="Raw probability (0-1)")
    risk_percent:            float = Field(..., description="Risk as percentage")
    risk_tier:               Literal["low", "moderate", "elevated", "high"]
    top_drivers:             list[DriverFeature]
    protective_factors:      list[DriverFeature]
    missing_labs:            list[str]
    expected_value:          float
    subtype_risks:           dict[str, float] = Field(default_factory=dict)
    clinical_summary:        str = Field(default="")
    # Threshold calibration fields (Fix 1)
    calibrated_threshold:    float = Field(default=0.5)
    operating_sensitivity:   float = Field(default=0.0)
    operating_specificity:   float = Field(default=0.0)
    # Patient identifier (Fix 2)
    patient_id:              Optional[str] = Field(None)
    disclaimer:              str = Field(
        default=(
            "This tool is for research and clinical decision support only. "
            "It does not constitute a diagnosis. All findings should be interpreted "
            "by a qualified clinician in the context of the full patient history."
        )
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    target: str
