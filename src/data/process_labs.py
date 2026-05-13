"""
Process and harmonize NHANES lab data across cycles 1999-2018.

NHANES changed variable names, units, and file names across cycles. This
module normalizes everything into a single consistent DataFrame with one
row per participant (SEQN) and standardized column names.

Output: data/processed/labs_harmonized.parquet

Run: python -m src.data.process_labs
"""

import warnings
import pandas as pd
import numpy as np
import pyreadstat
from pathlib import Path

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

# ---------------------------------------------------------------------------
# Canonical column names we want in the output
# ---------------------------------------------------------------------------
LAB_COLUMNS = {
    # CBC
    "wbc":      "WBC (10^9/L)",
    "rbc":      "RBC (10^12/L)",
    "hgb":      "Hemoglobin (g/dL)",
    "hct":      "Hematocrit (%)",
    "plt":      "Platelets (10^9/L)",
    "mcv":      "MCV (fL)",
    "mch":      "MCH (pg)",
    "mchc":     "MCHC (g/dL)",
    "rdw":      "RDW (%)",
    "neut_pct": "Neutrophils (%)",
    "lymph_pct":"Lymphocytes (%)",
    "mono_pct": "Monocytes (%)",
    "eos_pct":  "Eosinophils (%)",
    "baso_pct": "Basophils (%)",
    # CMP / Biochemistry
    "glucose":  "Glucose (mg/dL)",
    "bun":      "BUN (mg/dL)",
    "creatinine":"Creatinine (mg/dL)",
    "bilirubin":"Total Bilirubin (mg/dL)",
    "alkphos":  "Alkaline Phosphatase (U/L)",
    "alt":      "ALT (U/L)",
    "ast":      "AST (U/L)",
    "albumin":  "Albumin (g/dL)",
    "total_protein":"Total Protein (g/dL)",
    "sodium":   "Sodium (mmol/L)",
    "potassium":"Potassium (mmol/L)",
    "chloride": "Chloride (mmol/L)",
    "co2":      "CO2 (mmol/L)",
    "calcium":  "Calcium (mg/dL)",
    "phosphorus":"Phosphorus (mg/dL)",
    "uric_acid":"Uric Acid (mg/dL)",
    "ldh":      "LDH (U/L)",
    # Iron
    "ferritin": "Ferritin (ng/mL)",
    "iron":     "Iron (ug/dL)",
    "tibc":     "TIBC (ug/dL)",
    # Inflammation
    "crp":      "CRP (mg/L)",
}

# ---------------------------------------------------------------------------
# Per-cycle variable mapping
# Maps (canonical_name) -> (nhanes_variable_name, optional_unit_scale_factor)
# scale_factor: multiply raw value by this to get canonical unit
# ---------------------------------------------------------------------------

# Most cycles from 2005+ use consistent variable names; earlier ones differ.
CBC_VARS = {
    # canonical_name: nhanes_var
    "wbc":       "LBXWBCSI",
    "rbc":       "LBXRBCSI",
    "hgb":       "LBXHGB",
    "hct":       "LBXHCT",
    "plt":       "LBXPLTSI",
    "mcv":       "LBXMCVSI",
    "mch":       "LBXMCHSI",
    "mchc":      "LBXMC",
    "rdw":       "LBXRDW",
    "neut_pct":  "LBXNEPCT",
    "lymph_pct": "LBXLYPCT",
    "mono_pct":  "LBXMOPCT",
    "eos_pct":   "LBXEOPCT",
    "baso_pct":  "LBXBAPCT",
}

BIOPRO_VARS = {
    "glucose":       "LBXSGL",
    "bun":           "LBXSBU",
    "creatinine":    "LBXSCR",
    "bilirubin":     "LBXSTB",
    "alkphos":       "LBXSAPSI",
    "alt":           "LBXSATSI",
    "ast":           "LBXSASSI",
    "albumin":       "LBXSAL",
    "total_protein": "LBXSTP",
    "sodium":        "LBXSNASI",
    "potassium":     "LBXSKSI",
    "chloride":      "LBXSCLSI",
    "co2":           "LBXSC3SI",
    "calcium":       "LBXSCASI",
    "phosphorus":    "LBXSPSI",
    "uric_acid":     "LBXSUA",
    "ldh":           "LBXSLDSI",
}

# Early cycle CBC variable names differ from 2005+
CBC_EARLY_VARS = {
    # 1999-2004 used these names (mostly same, some differences)
    "wbc":       "LBXWBCSI",
    "rbc":       "LBXRBCSI",
    "hgb":       "LBXHGB",
    "hct":       "LBXHCT",
    "plt":       "LBXPLTSI",
    "mcv":       "LBXMCVSI",
    "mch":       "LBXMCHSI",
    "mchc":      "LBXMC",
    "rdw":       "LBXRDW",
    "neut_pct":  "LBXNEPCT",
    "lymph_pct": "LBXLYPCT",
    "mono_pct":  "LBXMOPCT",
    "eos_pct":   "LBXEOPCT",
    "baso_pct":  "LBXBAPCT",
}

BIOPRO_EARLY_VARS = {
    # 1999-2004 biochemistry used slightly different names in some cycles
    "glucose":       "LBXSGL",
    "bun":           "LBXSBU",
    "creatinine":    "LBXSCR",
    "bilirubin":     "LBXSTB",
    "alkphos":       "LBXSAPSI",
    "alt":           "LBXSATSI",
    "ast":           "LBXSASSI",
    "albumin":       "LBXSAL",
    "total_protein": "LBXSTP",
    "sodium":        "LBXSNASI",
    "potassium":     "LBXSKSI",
    "chloride":      "LBXSCLSI",
    "co2":           "LBXSC3SI",
    "calcium":       "LBXSCASI",
    "phosphorus":    "LBXSPSI",
    "uric_acid":     "LBXSUA",
}

FERRITIN_VARS = {
    "ferritin": "LBXFER",
    "iron":     "LBXFE",
    "tibc":     "LBXTIB",
}

CRP_VARS = {
    "crp": "LBXCRP",
}

DEMO_VARS = {
    "age":    "RIDAGEYR",
    "sex":    "RIAGENDR",   # 1=male, 2=female
    "race":   "RIDRETH1",
    "exam_year_start": "SDDSRVYR",  # survey year cycle code
}

# Cycle metadata
CYCLES = [
    {"label": "1999-2000", "suffix": "",    "year_mid": 2000, "cbc": "CBC",     "biopro": "BIOPRO",   "mcq": "MCQ",   "demo": "DEMO",   "ferritin": "FETIB",   "crp": "CRP"},
    {"label": "2001-2002", "suffix": "_B",  "year_mid": 2002, "cbc": "CBC_B",   "biopro": "BIOPRO_B", "mcq": "MCQ_B", "demo": "DEMO_B", "ferritin": "FETIB_B", "crp": "CRP_B"},
    {"label": "2003-2004", "suffix": "_C",  "year_mid": 2004, "cbc": "CBC_C",   "biopro": "BIOPRO_C", "mcq": "MCQ_C", "demo": "DEMO_C", "ferritin": "FETIB_C", "crp": "CRP_C"},
    {"label": "2005-2006", "suffix": "_D",  "year_mid": 2006, "cbc": "CBC_D",   "biopro": "BIOPRO_D", "mcq": "MCQ_D", "demo": "DEMO_D", "ferritin": "FETIB_D", "crp": "CRP_D"},
    {"label": "2007-2008", "suffix": "_E",  "year_mid": 2008, "cbc": "CBC_E",   "biopro": "BIOPRO_E", "mcq": "MCQ_E", "demo": "DEMO_E", "ferritin": "FETIB_E", "crp": "CRP_E"},
    {"label": "2009-2010", "suffix": "_F",  "year_mid": 2010, "cbc": "CBC_F",   "biopro": "BIOPRO_F", "mcq": "MCQ_F", "demo": "DEMO_F", "ferritin": "FETIB_F", "crp": "CRP_F"},
    {"label": "2011-2012", "suffix": "_G",  "year_mid": 2012, "cbc": "CBC_G",   "biopro": "BIOPRO_G", "mcq": "MCQ_G", "demo": "DEMO_G", "ferritin": "FETIB_G", "crp": "CRP_G"},
    {"label": "2013-2014", "suffix": "_H",  "year_mid": 2014, "cbc": "CBC_H",   "biopro": "BIOPRO_H", "mcq": "MCQ_H", "demo": "DEMO_H", "ferritin": "FETIB_H", "crp": "CRP_H"},
    {"label": "2015-2016", "suffix": "_I",  "year_mid": 2016, "cbc": "CBC_I",   "biopro": "BIOPRO_I", "mcq": "MCQ_I", "demo": "DEMO_I", "ferritin": "FETIB_I", "crp": "CRP_I"},
    {"label": "2017-2018", "suffix": "_J",  "year_mid": 2018, "cbc": "CBC_J",   "biopro": "BIOPRO_J", "mcq": "MCQ_J", "demo": "DEMO_J", "ferritin": "FETIB_J", "crp": None},
]


def safe_read_xpt(path: Path) -> pd.DataFrame | None:
    """Read an XPT file, return None if missing or corrupted."""
    if not path.exists():
        print(f"    [missing] {path.name}")
        return None
    try:
        df, _ = pyreadstat.read_xport(str(path))
        df.columns = df.columns.str.upper()
        return df
    except Exception as exc:
        print(f"    [error reading {path.name}] {exc}")
        return None


def extract_vars(df: pd.DataFrame, var_map: dict[str, str]) -> pd.DataFrame:
    """Extract canonical columns from a raw NHANES DataFrame."""
    out = pd.DataFrame(index=df.index)
    out["SEQN"] = df["SEQN"].astype(int)
    for canonical, nhanes_var in var_map.items():
        if nhanes_var in df.columns:
            out[canonical] = pd.to_numeric(df[nhanes_var], errors="coerce")
        else:
            out[canonical] = np.nan
    return out


def process_cycle(cycle: dict) -> pd.DataFrame:
    """Load and harmonize one NHANES cycle into a single row-per-SEQN DataFrame."""
    print(f"\n  Processing {cycle['label']}...")

    # Demographics
    demo_path = RAW_DIR / f"{cycle['demo']}.XPT"
    demo_raw = safe_read_xpt(demo_path)
    if demo_raw is None:
        print(f"    [SKIP] No demographics for {cycle['label']}")
        return pd.DataFrame()

    demo = extract_vars(demo_raw, DEMO_VARS)
    demo["cycle"] = cycle["label"]
    demo["exam_year"] = cycle["year_mid"]

    # CBC
    cbc_path = RAW_DIR / f"{cycle['cbc']}.XPT"
    cbc_raw = safe_read_xpt(cbc_path)
    var_map = CBC_EARLY_VARS if cycle["year_mid"] <= 2004 else CBC_VARS
    cbc = extract_vars(cbc_raw, var_map) if cbc_raw is not None else pd.DataFrame({"SEQN": demo["SEQN"]})

    # Biochemistry
    biopro_path = RAW_DIR / f"{cycle['biopro']}.XPT"
    biopro_raw = safe_read_xpt(biopro_path)
    var_map_bio = BIOPRO_EARLY_VARS if cycle["year_mid"] <= 2004 else BIOPRO_VARS
    biopro = extract_vars(biopro_raw, var_map_bio) if biopro_raw is not None else pd.DataFrame({"SEQN": demo["SEQN"]})

    # Ferritin / Iron
    ferritin_path = RAW_DIR / f"{cycle['ferritin']}.XPT"
    ferritin_raw = safe_read_xpt(ferritin_path)
    ferritin = extract_vars(ferritin_raw, FERRITIN_VARS) if ferritin_raw is not None else pd.DataFrame({"SEQN": demo["SEQN"]})

    # CRP
    crp = pd.DataFrame({"SEQN": demo["SEQN"], "crp": np.nan})
    if cycle["crp"] is not None:
        crp_path = RAW_DIR / f"{cycle['crp']}.XPT"
        crp_raw = safe_read_xpt(crp_path)
        if crp_raw is not None:
            crp = extract_vars(crp_raw, CRP_VARS)

    # Merge all on SEQN
    merged = demo.copy()
    for sub_df in [cbc, biopro, ferritin, crp]:
        if "SEQN" in sub_df.columns and len(sub_df) > 0:
            merged = merged.merge(sub_df, on="SEQN", how="left")

    print(f"    {len(merged):,} participants, {merged.shape[1]} columns")
    return merged


def run() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_cycles = []
    for cycle in CYCLES:
        df = process_cycle(cycle)
        if not df.empty:
            all_cycles.append(df)

    if not all_cycles:
        raise RuntimeError("No data loaded — run download_nhanes.py first.")

    labs = pd.concat(all_cycles, ignore_index=True)

    # ── Enforce plausible value ranges (hard clip — not impute) ──────────────
    ranges = {
        "wbc":       (0.1, 100.0),
        "rbc":       (1.0, 10.0),
        "hgb":       (3.0, 22.0),
        "hct":       (10.0, 65.0),
        "plt":       (10.0, 1500.0),
        "mcv":       (50.0, 130.0),
        "mch":       (10.0, 50.0),
        "mchc":      (20.0, 40.0),
        "rdw":       (5.0, 30.0),
        "neut_pct":  (0.0, 100.0),
        "lymph_pct": (0.0, 100.0),
        "mono_pct":  (0.0, 50.0),
        "eos_pct":   (0.0, 60.0),
        "baso_pct":  (0.0, 10.0),
        "glucose":   (20.0, 600.0),
        "bun":       (1.0, 200.0),
        "creatinine":(0.1, 30.0),
        "albumin":   (1.0, 6.0),
        "alt":       (1.0, 3000.0),
        "ast":       (1.0, 3000.0),
        "alkphos":   (1.0, 3000.0),
        "bilirubin": (0.1, 30.0),
        "ferritin":  (1.0, 10000.0),
        "crp":       (0.01, 300.0),
    }
    for col, (lo, hi) in ranges.items():
        if col in labs.columns:
            labs[col] = labs[col].clip(lower=lo, upper=hi)

    # ── Convert sex to binary (1=male, 0=female) ────────────────────────────
    if "sex" in labs.columns:
        labs["is_male"] = (labs["sex"] == 1).astype(float)

    out_path = PROCESSED_DIR / "labs_harmonized.parquet"
    labs.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(labs):,} rows, {labs.shape[1]} cols)")
    return labs


if __name__ == "__main__":
    df = run()
    print("\nColumn coverage (% non-null):")
    coverage = (df.notna().mean() * 100).sort_values(ascending=False)
    for col, pct in coverage.items():
        if pct > 0:
            print(f"  {col:<20} {pct:.1f}%")
