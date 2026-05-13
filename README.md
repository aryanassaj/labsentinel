# LabSentinel — Early Cancer Signal Detector

**[Launch LabSentinel](https://aryanassaj.github.io/labsentinel/labsentinel.html)**

> **RESEARCH TOOL — NOT FOR CLINICAL DIAGNOSIS.**  
> LabSentinel is a clinical decision-support aid that flags statistical patterns in routine blood work associated with cancer. It does not diagnose cancer. All findings must be interpreted by a qualified clinician.

---

## What Is LabSentinel?

LabSentinel is an AI-powered early cancer signal detector that reads a patient's standard blood panel — the same Complete Blood Count (CBC) and Comprehensive Metabolic Panel (CMP) ordered at routine physicals — and outputs a cancer risk score with a detailed explanation of which lab values drove that score.

**The core insight:** Cancer causes measurable changes in blood chemistry months or years before symptoms appear. Inflammatory markers rise, red blood cell indices shift, liver enzymes change, protein ratios alter. These patterns are subtle and difficult for a clinician to spot while reviewing a routine lab printout — but machine learning can detect them reliably across a population.

LabSentinel translates those patterns into an actionable signal: a risk score from 0–100%, a risk tier (Low / Moderate / Elevated / High), a plain-English clinical summary, and a ranked list of the specific lab values driving the result — all in under one second.

---

## The Data Behind the Model

### Source: NHANES 1999–2018 (CDC National Survey)

The model was trained on the **National Health and Nutrition Examination Survey (NHANES)**, a CDC program that has collected comprehensive health and lab data on tens of thousands of Americans since 1999. NHANES is the gold standard for US population health research.

| Statistic | Value |
|---|---|
| Total participants (after adult filtering) | **53,448** |
| Survey cycles covered | 1999–2000 through 2017–2018 (10 cycles) |
| Cancer-positive labels | **5,723 (10.7%)** |
| Mortality follow-up through | **2019 (NCHS Linked Mortality Files)** |

### How Outcomes Are Defined

Cancer labels come from two complementary sources:

**1. NCHS Mortality Linkage (Gold Standard — Pre-Diagnostic)**  
The National Center for Health Statistics links NHANES participants to the National Death Index. We know who died of cancer (ICD-10 C00–C96) within 3, 5, and 10 years of their blood draw. This is the most clinically meaningful label: the participant had cancer-related pathology *at the time of their exam*, but didn't know it yet. We are detecting the disease before it was diagnosed.

- Cancer deaths confirmed: **1,978**
- Cancer deaths within 5 years of exam: **740**

**2. MCQ Self-Report (Incidence Coverage)**  
NHANES asks every participant about cancer diagnoses, type of cancer, and age at diagnosis. This adds diagnosed-but-surviving cancer patients to the training set and expands coverage across all 10 survey cycles.

- Self-reported cancer diagnoses: **5,166**

**Combined outcome:** 5,723 participants with either confirmed cancer death or self-reported diagnosis = the ground truth for training.

### Lab Data Collected

From each participant's exam:

- **CBC (Complete Blood Count):** WBC, RBC, Hemoglobin, Hematocrit, Platelets, MCV, MCH, MCHC, RDW, Neutrophil%, Lymphocyte%, Monocyte%, Eosinophil%, Basophil%
- **CMP (Comprehensive Metabolic Panel):** Glucose, BUN, Creatinine, Sodium, Potassium, Chloride, CO2/Bicarb, Calcium, Total Protein, Albumin, Bilirubin, ALT, AST, Alkaline Phosphatase, Uric Acid, LDH, Phosphorus
- **Iron Studies:** Ferritin, Serum Iron, TIBC
- **Inflammation:** C-Reactive Protein (CRP / hsCRP)

---

## How the Model Works

### Feature Engineering (86 Features)

Raw lab values are transformed into 86 clinically-meaningful features:

**Inflammatory Ratios**
- **NLR** (Neutrophil-to-Lymphocyte Ratio) — elevated in cancer, systemic inflammation
- **PLR** (Platelet-to-Lymphocyte Ratio) — marker of inflammatory response
- **MLR** (Monocyte-to-Lymphocyte Ratio) — innate immune dysregulation
- **SII** (Systemic Immune-Inflammation Index) — composite immune stress score
- **SIRI** (Systemic Inflammation Response Index) — multi-lineage inflammatory marker

**Composite Clinical Scores**
- **Glasgow Prognostic Score (GPS/mGPS)** — validated cancer prognosis score based on CRP + albumin
- **eGFR** (CKD-EPI formula) — estimated kidney function
- **De Ritis Ratio** (AST/ALT) — liver function pattern
- **BUN/Creatinine Ratio** — renal/metabolic stress
- **A/G Ratio** (Albumin/Globulin) — protein dysregulation

**Log Transformations** — Right-skewed markers (ferritin, LDH, CRP, liver enzymes) are log-transformed for better linearity

**Age Interactions** — `age × NLR`, `age²`, `age_50plus × is_male` — age modifies cancer risk non-linearly

**Binary Clinical Flags (30+)**
Each flag encodes a clinically recognized pattern:
- `flag_anemia` — Hemoglobin below sex-adjusted threshold
- `flag_thrombocytosis` — Platelets > 400 × 10⁹/L (cancer-associated)
- `flag_lymphopenia` — Lymphocytes < 20% (immune suppression)
- `flag_elevated_crp` — CRP > 10 mg/L (systemic inflammation)
- `flag_hypoalbuminemia` — Albumin < 3.5 g/dL (cancer-related malnutrition)
- `flag_elevated_ldh` — LDH > 280 U/L (cell turnover marker)
- `flag_macrocytosis` — MCV > 100 fL (B12/folate/marrow stress)
- ...and 23+ more

### Machine Learning Algorithm

**XGBoost** (Extreme Gradient Boosted Trees) — the best-performing algorithm for tabular clinical data. Chosen over neural networks for:
- Superior performance with missing data (common in real labs)
- Built-in feature importance
- No normalization required
- Fast inference (<1ms per patient)

**Class Imbalance Handling:** SMOTE (Synthetic Minority Oversampling Technique) creates synthetic minority-class examples to prevent the model from ignoring the 10.7% cancer-positive class.

**Training:** 5-fold stratified cross-validation on 53,448 participants, then a final model trained on all data.

### Model Performance

| Metric | Value |
|---|---|
| AUC-ROC | **0.810** |
| AUC-PR | **0.319** |
| Operating Sensitivity | **83.2%** |
| Operating Specificity | **66.9%** |
| Calibrated Threshold | **0.106** |

The threshold was calibrated using **Youden's J statistic** (sensitivity + specificity − 1, maximized) on a held-out validation set. This finds the optimal trade-off point: 83% of true cancer cases are flagged, while 67% of non-cancer patients correctly screen negative.

**AUC-ROC of 0.810** means the model ranks a randomly selected cancer patient above a randomly selected non-cancer patient 81% of the time — from routine blood work alone, before any symptoms.

---

## Explainability: SHAP Values

Every prediction includes a full SHAP (SHapley Additive exPlanations) breakdown — one value per feature showing exactly how much that lab value raised or lowered this patient's risk score.

This means:
- **Doctors see why** the model flagged the patient — not just a number
- **Each driver has a clinical note** explaining its significance
- **Protective factors** are also shown (labs that pushed the score down)
- The explanation is rendered as a ranked list in the UI and included in the PDF report

SHAP values are computed using TreeExplainer, which is exact (not approximate) for tree models.

---

## Cancer Subtype Models

In addition to the main "any cancer" risk score, LabSentinel trains 6 cancer-specific subtype models:

| Cancer Type | Notes |
|---|---|
| Colorectal | Leading GI malignancy |
| Lung | Most common cancer death |
| Breast | CBC patterns (anemia, PLR) |
| Prostate | Age + metabolic markers |
| Leukemia | WBC lineage ratios critical |
| Lymphoma | Lymphopenia, LDH patterns |

Each subtype score appears as a secondary "Cancer Subtype Signals" section in the report, giving clinicians an additional layer of specificity.

---

## The Interface

### Single Patient Analysis

The primary use case. A clinician or medical assistant enters lab values from a patient's routine blood draw:

1. **Demographics** — Age, Sex, optional Patient ID
2. **CBC** — 14 fields (all optional except age/sex)
3. **CMP** — 17 fields
4. **Additional Labs** — Iron studies, CRP, ferritin

The form validates values in real-time against reference ranges, highlighting abnormals in orange/red.

Results are returned in under 1 second:

- **Risk Score** — large percentage display with color-coded gauge (green → amber → red)
- **Risk Tier** — Low / Moderate / Elevated / High (calibrated at 10.6% threshold)
- **Clinical Summary** — 2–3 sentence plain-English paragraph stating which specific labs were most concerning and what it means clinically
- **Top Risk Drivers** — ranked list with SHAP impact bars and clinical notes per driver
- **Protective Factors** — labs that lowered the score
- **Cancer Subtype Signals** — 6 cancer-specific risk percentages
- **Missing Labs** — which optional tests, if added, would most improve accuracy
- **Operating Point** — displayed sensitivity/specificity so clinicians know the model's operating characteristics

### PDF Report Download

Every result can be downloaded as a formatted clinical PDF containing:
- Header with date, time, patient ID
- Full risk score and tier (large, color-coded)
- Clinical summary paragraph
- Risk factors table (Factor | Value | Direction | Clinical Note)
- Protective factors table
- Cancer subtype signals table
- Missing labs section
- Legal disclaimer footer

### Batch Worklist (API)

For labs and EHR integrations: `POST /predict/worklist` accepts up to 500 patients in a single call and returns only the `elevated`/`high` tier patients sorted by risk, ready to populate a morning worklist.

---

## API Reference

The backend is a **FastAPI** REST API (Python) running on port 8000.

### Authentication

Set `LABSENTINEL_API_KEY` environment variable to enable API key authentication. When set, all prediction endpoints require `X-API-Key: <key>` header. Without the env var set, auth is disabled (local dev mode).

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI (HTML) |
| `GET` | `/health` | Model status check |
| `GET` | `/model/info` | Version, training stats, feature count |
| `GET` | `/model/feature-importance` | Top 20 features by SHAP importance |
| `GET` | `/reference-ranges` | Lab reference ranges for UI |
| `POST` | `/predict` | Single patient prediction |
| `POST` | `/predict/batch` | Up to 500 patients (returns array) |
| `POST` | `/predict/report` | Returns downloadable PDF report |
| `POST` | `/predict/worklist` | Batch → elevated/high patients only, sorted |

### Example Request

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 58,
    "sex": "male",
    "wbc": 11.2,
    "hgb": 11.8,
    "plt": 412,
    "rdw": 15.1,
    "albumin": 3.2,
    "crp": 18.4,
    "ldh": 310,
    "lymph_pct": 14.2
  }'
```

### Example Response

```json
{
  "risk_score": 0.423,
  "risk_percent": 42.3,
  "risk_tier": "elevated",
  "calibrated_threshold": 0.106,
  "operating_sensitivity": 0.832,
  "operating_specificity": 0.669,
  "clinical_summary": "Routine lab panel flagged several patterns associated with cancer. CRP is markedly elevated (18.4 mg/L), suggesting systemic inflammation; albumin is low (3.2 g/dL), consistent with cancer-related malnutrition; lymphopenia at 14.2% indicates immune suppression. Clinical recommendation: Elevated signal — consider oncology referral and further workup.",
  "top_drivers": [
    {
      "name": "crp",
      "display_name": "C-Reactive Protein",
      "value": 18.4,
      "shap_value": 0.218,
      "direction": "risk",
      "interpretation": "CRP is markedly elevated. Systemic inflammation is a hallmark of occult malignancy."
    }
  ],
  "subtype_risks": {
    "colorectal": 31.2,
    "lung": 18.7,
    "lymphoma": 22.4
  }
}
```

---

## Clinical Interpretation Guide

### Risk Tiers

| Tier | Meaning | Suggested Action |
|---|---|---|
| **Low** | Pattern consistent with healthy population | No additional workup based on labs alone |
| **Moderate** | Some statistical elevation; may reflect benign inflammation | Clinical context review; repeat labs if appropriate |
| **Elevated** | Pattern statistically elevated above threshold | Consider oncology referral; additional targeted testing |
| **High** | Strong multi-marker cancer signal | Urgent oncology referral recommended |

*Tier boundaries are calibrated proportionally to the Youden's J threshold (0.106). "Elevated" begins at the model's operating threshold.*

### What This Tool Is Not

- **Not a diagnostic test.** A high score does not mean the patient has cancer.
- **Not a screening protocol.** It is a decision-support signal, not a replacement for established screening (colonoscopy, mammography, PSA, LDCT).
- **Not validated for all populations.** Trained on a US population survey. Performance may differ in pediatric, acute-care, or non-US populations.
- **Not real-time EHR integration.** Manual lab entry is required (API batch mode available for integration work).

### What This Tool Is

- **A pre-screen triage aid.** Flag patients whose routine labs warrant a second look.
- **An explainability layer on routine blood work.** Every lab result now has context — how much did it matter for this patient?
- **A mortality-validated detector.** Because we trained on deaths confirmed by the National Death Index, we are detecting real pre-diagnostic cancer signal, not just correlates of diagnosis.

---

## Setup & Installation

### Requirements

- Python 3.11 (not 3.12+ — some ML libraries require 3.11)
- ~4 GB disk space (NHANES data + model files)
- No GPU required

### Quick Start

```bash
# 1. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NHANES data (~1.5 GB)
python -m src.data.download_nhanes

# 4. Process data
python -m src.data.process_labs
python -m src.data.process_outcomes
python -m src.data.merge_datasets

# 5. Train model (~2–5 min)
python -m src.models.train

# 6. Start server
uvicorn api.main:app --reload --port 8000

# 7. Open browser
open http://localhost:8000
```

### One-Command Pipeline

```bash
python run_pipeline.py                  # Full pipeline + server
python run_pipeline.py --skip-download  # Skip re-downloading (data already present)
python run_pipeline.py --only-train     # Only retrain (processed data already present)
python run_pipeline.py --only-server    # Only start server (model already trained)
python run_pipeline.py --tune           # Full pipeline with Optuna hyperparameter tuning
```

### API Key (Optional, for Production)

```bash
export LABSENTINEL_API_KEY="your-secret-key-here"
uvicorn api.main:app --port 8000
```

Requests then require: `X-API-Key: your-secret-key-here`

---

## Project Structure

```
labsentinel/
├── api/
│   ├── main.py          # FastAPI app, endpoints, PDF generation, audit logging
│   └── schemas.py       # Pydantic request/response models
├── frontend/
│   ├── index.html       # EHR-style single-page UI
│   ├── app.js           # Form handling, rendering, PDF download
│   └── styles.css       # Clinical UI design
├── src/
│   ├── data/
│   │   ├── download_nhanes.py    # CDC data downloader
│   │   ├── process_labs.py       # CBC + CMP processing
│   │   ├── process_outcomes.py   # Cancer label construction
│   │   └── merge_datasets.py     # Final dataset assembly
│   ├── features/
│   │   └── lab_features.py       # 86 feature engineering transformations
│   └── models/
│       ├── train.py              # XGBoost training, CV, threshold calibration
│       └── explain.py            # SHAP explainer, risk tier assignment
├── data/
│   ├── raw/             # Downloaded NHANES .XPT files
│   ├── external/        # NCHS mortality linkage .dat files
│   └── processed/       # Parquet files (labs, outcomes, merged)
├── models_saved/
│   ├── model_any_cancer.json     # Trained XGBoost model
│   ├── threshold.json            # Calibrated operating threshold
│   ├── model_version.json        # Training metadata
│   ├── feature_cols.json         # Feature column names
│   ├── shap_expected_value.json  # SHAP baseline
│   ├── feature_importance_*.json # Feature importances
│   └── model_*.json              # Subtype models (colorectal, lung, breast, etc.)
├── logs/
│   └── predictions.log   # Rotating audit log (10 MB × 5 files)
├── requirements.txt
└── run_pipeline.py
```

---

## Audit Logging

Every prediction is logged to `logs/predictions.log` (rotating, 10 MB max, 5 backups):

```
2026-04-11 09:23:14 | patient=PT-001 | score=0.423 | tier=elevated | drivers=crp,albumin,lymph_pct
```

This provides a complete audit trail of all queries for compliance and review.

---

## Key Findings from Training Data

The top predictive features by mean SHAP importance:

| Rank | Feature | Clinical Meaning |
|---|---|---|
| 1 | **Age** | Cancer risk increases exponentially with age |
| 2 | **Age²** | Non-linear age effect captured |
| 3 | **Sex (male)** | Male sex associated with higher overall cancer risk |
| 4 | **LDH** | Cell turnover / tumor burden marker |
| 5 | **CRP** | Systemic inflammation — cancer hallmark |
| 6 | **Total Protein** | Protein dysregulation in malignancy |
| 7 | **MCV** | Macrocytosis from marrow stress or B12/folate depletion |
| 8 | **MLR** | Monocyte-lymphocyte ratio — innate immune shift |
| 9 | **MCH** | Red cell hemoglobin content |
| 10 | **eGFR** | Kidney function — renal cancer, paraneoplastic |

---

## Limitations & Ethical Considerations

1. **Population bias:** NHANES is a US civilian non-institutionalized population. Performance on hospital inpatients, pediatric, or international populations is untested.

2. **Cross-sectional design:** MCQ self-report labels are cross-sectional. The mortality linkage gives true pre-diagnostic signal, but only covers deaths through 2019.

3. **Spectrum bias:** Participants who completed blood draws may be healthier than non-participants. This could affect sensitivity in sicker populations.

4. **Label definition:** "Any cancer" includes non-melanoma skin cancer, which has low mortality and may dilute the signal for more aggressive cancers.

5. **No biopsy ground truth:** Cancer labels come from self-report and death certificates, not pathology reports.

6. **Intended use:** Pre-screening decision support only. This tool is not FDA-cleared and should not be used as the sole basis for any clinical decision.

---

## Technology Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost 1.7+ |
| Explainability | SHAP (TreeExplainer) |
| Class Balancing | imbalanced-learn (SMOTE) |
| Hyperparameter Tuning | Optuna |
| Backend API | FastAPI + Uvicorn |
| Data Schemas | Pydantic v2 |
| PDF Generation | ReportLab |
| Data Processing | pandas + pyreadstat |
| Frontend | Vanilla HTML/CSS/JavaScript |
| Data Source | NHANES (CDC), NCHS Mortality Linkage |

---

## License & Data Attribution

Training data: **NHANES (National Health and Nutrition Examination Survey)**, CDC/NCHS, 1999–2018. Public domain.
Mortality data: **NCHS Linked Mortality Files**, 2019 public use. Public domain.

This software is provided for research and educational use. It is not FDA-cleared for clinical use. The developers make no warranties regarding fitness for any particular purpose.

---

*LabSentinel v1.0.0 — Trained on 53,448 NHANES participants, 1999–2018 — AUC-ROC 0.810 — 83.2% sensitivity at calibrated threshold*
