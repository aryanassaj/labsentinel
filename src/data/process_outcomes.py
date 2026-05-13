"""
Build the outcome labels for the cancer signal detection model.

Two complementary outcome sources are combined:

1. MORTALITY LINKAGE (gold standard for pre-diagnostic labeling)
   - NCHS linked mortality files provide follow-up through 2019
   - ICD-10 codes C00-C96 = malignant neoplasms (cancer death)
   - We label: cancer_death_5yr = did this person die of cancer within 5 years
     of their NHANES exam? This is the TRUE pre-diagnostic signal.

2. MCQ SELF-REPORT (cross-sectional, but adds incidence data)
   - MCQ220: ever diagnosed with cancer?
   - MCQ230A-D: type of cancer
   - MCQ240A-D: age at first diagnosis
   - We compute: approx_years_before_exam = age_at_exam - age_at_diagnosis
   - Label: had_cancer_at_exam (prevalent), had_recent_cancer (within 5yr)

Output: data/processed/outcomes.parquet

Run: python -m src.data.process_outcomes
"""

import re
import struct
import pandas as pd
import numpy as np
import pyreadstat
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
EXTERNAL_DIR = Path(__file__).parents[2] / "data" / "external"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

# ICD-10 cancer codes (C00-C96) in mortality linkage
CANCER_ICD10_PATTERN = re.compile(r"^C[0-9]")

# ICD-9 cancer codes (140-208) in older mortality linkage
CANCER_ICD9_RANGE = (140, 209)

# Cancer type mapping from MCQ230 codes
# NHANES codes: 10=bladder, 11=blood, 12=bone, 14=breast, 15=cervix,
# 16=colon, 17=esophagus, 18=gallbladder, 19=kidney, 20=larynx/throat,
# 21=leukemia, 22=liver, 23=lung, 24=lymphoma, 25=melanoma, 26=mouth/throat,
# 27=nervous system, 28=ovary, 29=pancreas, 30=prostate, 31=rectum,
# 32=skin (non-melanoma), 33=soft tissue, 34=stomach, 35=testis,
# 36=thyroid, 37=uterus, 38=other
CANCER_TYPE_MAP = {
    10: "bladder", 11: "blood/other_heme", 12: "bone", 14: "breast",
    15: "cervical", 16: "colorectal", 17: "esophageal", 18: "gallbladder",
    19: "kidney", 20: "laryngeal", 21: "leukemia", 22: "liver",
    23: "lung", 24: "lymphoma", 25: "melanoma", 26: "oral",
    27: "cns", 28: "ovarian", 29: "pancreatic", 30: "prostate",
    31: "colorectal", 32: "skin_nonmelanoma", 33: "soft_tissue",
    34: "gastric", 35: "testicular", 36: "thyroid", 37: "uterine", 38: "other",
}

# High-priority cancers for separate outcome columns
HIGH_PRIORITY_CANCERS = {
    "colorectal": [16, 31],
    "lung": [23],
    "breast": [14],
    "prostate": [30],
    "leukemia": [21],
    "lymphoma": [24],
    "pancreatic": [29],
}

# Cycle metadata with survey year midpoints
CYCLE_YEAR_MAP = {
    "1999-2000": 2000,
    "2001-2002": 2002,
    "2003-2004": 2004,
    "2005-2006": 2006,
    "2007-2008": 2008,
    "2009-2010": 2010,
    "2011-2012": 2012,
    "2013-2014": 2014,
    "2015-2016": 2016,
    "2017-2018": 2018,
}

MCQ_CYCLES = [
    ("1999-2000", "MCQ"),
    ("2001-2002", "MCQ_B"),
    ("2003-2004", "MCQ_C"),
    ("2005-2006", "MCQ_D"),
    ("2007-2008", "MCQ_E"),
    ("2009-2010", "MCQ_F"),
    ("2011-2012", "MCQ_G"),
    ("2013-2014", "MCQ_H"),
    ("2015-2016", "MCQ_I"),
    ("2017-2018", "MCQ_J"),
]

DEMO_CYCLES = [
    ("1999-2000", "DEMO"),
    ("2001-2002", "DEMO_B"),
    ("2003-2004", "DEMO_C"),
    ("2005-2006", "DEMO_D"),
    ("2007-2008", "DEMO_E"),
    ("2009-2010", "DEMO_F"),
    ("2011-2012", "DEMO_G"),
    ("2013-2014", "DEMO_H"),
    ("2015-2016", "DEMO_I"),
    ("2017-2018", "DEMO_J"),
]


# ── Fixed-width mortality linkage parser ─────────────────────────────────────
# Actual layout verified by byte-level inspection of the 2019 public use files.
# The 2019 release differs from older NCHS codebooks: SEQN is 5 chars (not 6),
# followed by 9 blank chars, then the mortality variables start at col 14.
# MAJOR_COD at [18] encodes broad cause-of-death category (2 = malignant neoplasm).
# PERMTH fields are right-justified 3-char fields at the end of each record.
MORT_COLSPECS = [
    (0,   5,  "SEQN"),
    (14,  15, "eligstat"),     # 1=eligible for mortality follow-up, 2=not eligible
    (15,  16, "mortstat"),     # 0=assumed alive, 1=confirmed dead
    (16,  17, "diabetes"),     # diabetes as contributing cause (0/1)
    (17,  18, "hyperten"),     # hypertension as contributing cause (0/1)
    (18,  19, "major_cod"),    # major cause-of-death: 1=heart, 2=cancer, 3=resp, etc.
    (19,  20, "diabetes_cod"), # diabetes indicator from multiple cause
    (42,  45, "permth_int"),   # person-months from interview to death/censoring
    (45,  48, "permth_exm"),   # person-months from exam to death/censoring
]


def parse_mortality_file(path: Path) -> pd.DataFrame:
    """Parse a fixed-width NCHS mortality linkage .dat file."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if len(line) < 19:
                continue
            row = {}
            for start, end, name in MORT_COLSPECS:
                val = line[start:end].strip() if len(line) >= end else line[start:].strip()
                row[name] = val if val not in ("", ".") else None
            rows.append(row)

    df = pd.DataFrame(rows)
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    df["mortstat"] = pd.to_numeric(df["mortstat"], errors="coerce")
    df["permth_int"] = pd.to_numeric(df["permth_int"], errors="coerce")
    df["permth_exm"] = pd.to_numeric(df["permth_exm"], errors="coerce")
    return df


def is_cancer_death(major_cod: str | None) -> bool:
    """Return True if MAJOR_COD == '2' (malignant neoplasm)."""
    return major_cod == "2"


def load_all_mortality() -> pd.DataFrame:
    """Load and concatenate all mortality linkage files."""
    mort_files = sorted(EXTERNAL_DIR.glob("NHANES_*_MORT_*.dat"))
    if not mort_files:
        print("  [WARNING] No mortality linkage files found in data/external/")
        print("  Run download_nhanes.py to fetch them.")
        return pd.DataFrame(columns=["SEQN", "mortstat", "permth_exm", "ucod_leading"])

    all_mort = []
    for path in mort_files:
        print(f"  Parsing mortality: {path.name}")
        df = parse_mortality_file(path)
        all_mort.append(df)

    mort = pd.concat(all_mort, ignore_index=True)
    mort = mort.drop_duplicates(subset="SEQN", keep="last")
    n_dead = (mort["mortstat"] == 1).sum()
    n_cancer = mort[mort["mortstat"] == 1]["major_cod"].eq("2").sum()
    print(f"  Mortality linkage: {len(mort):,} participants ({n_dead:,} deaths, {n_cancer:,} cancer deaths)")
    return mort


def build_mortality_labels(mort: pd.DataFrame) -> pd.DataFrame:
    """Create binary cancer outcome labels from mortality data."""
    labels = mort[["SEQN"]].copy()

    labels["cancer_death"] = mort.apply(
        lambda r: 1 if (r["mortstat"] == 1 and is_cancer_death(r.get("major_cod"))) else 0,
        axis=1,
    )

    # Months from exam to death/censoring
    labels["months_to_event"] = mort["permth_exm"]

    # Time-windowed outcomes
    for years in [3, 5, 10]:
        months = years * 12
        labels[f"cancer_death_{years}yr"] = (
            (labels["cancer_death"] == 1) &
            (labels["months_to_event"] <= months)
        ).astype(int)

    return labels


def load_mcq_cycle(cycle_label: str, mcq_file: str) -> pd.DataFrame:
    """Load one cycle of MCQ data, extract cancer questions."""
    path = RAW_DIR / f"{mcq_file}.XPT"
    if not path.exists():
        return pd.DataFrame()

    try:
        df, _ = pyreadstat.read_xport(str(path))
        df.columns = df.columns.str.upper()
    except Exception as exc:
        print(f"  [error] {path.name}: {exc}")
        return pd.DataFrame()

    out = pd.DataFrame()
    out["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")

    # MCQ220: ever told you had cancer? (1=yes, 2=no, 7=refused, 9=don't know)
    if "MCQ220" in df.columns:
        out["ever_cancer"] = (df["MCQ220"] == 1).astype(float)
    else:
        out["ever_cancer"] = np.nan

    # MCQ230A-D: type of cancer (first four mentions)
    for slot in ["A", "B", "C", "D"]:
        col = f"MCQ230{slot}"
        if col in df.columns:
            out[f"cancer_type_{slot.lower()}"] = pd.to_numeric(df[col], errors="coerce")

    # MCQ240A-D: age at first diagnosis for each cancer
    for slot in ["A", "B", "C", "D"]:
        col = f"MCQ240{slot}"
        if col in df.columns:
            out[f"cancer_age_{slot.lower()}"] = pd.to_numeric(df[col], errors="coerce")

    out["cycle"] = cycle_label
    return out


def load_demo_ages() -> pd.DataFrame:
    """Load ages at exam for all participants across cycles."""
    all_demo = []
    for cycle_label, demo_file in DEMO_CYCLES:
        path = RAW_DIR / f"{demo_file}.XPT"
        if not path.exists():
            continue
        try:
            df, _ = pyreadstat.read_xport(str(path))
            df.columns = df.columns.str.upper()
            demo = pd.DataFrame({
                "SEQN": pd.to_numeric(df["SEQN"], errors="coerce"),
                "age_at_exam": pd.to_numeric(df.get("RIDAGEYR", np.nan), errors="coerce"),
                "cycle": cycle_label,
            })
            all_demo.append(demo)
        except Exception as exc:
            print(f"  [demo error] {demo_file}: {exc}")

    return pd.concat(all_demo, ignore_index=True) if all_demo else pd.DataFrame()


def build_mcq_labels() -> pd.DataFrame:
    """Build cancer outcome labels from MCQ self-report across all cycles."""
    all_mcq = []
    for cycle_label, mcq_file in MCQ_CYCLES:
        df = load_mcq_cycle(cycle_label, mcq_file)
        if not df.empty:
            all_mcq.append(df)

    if not all_mcq:
        print("  [WARNING] No MCQ data loaded")
        return pd.DataFrame()

    mcq = pd.concat(all_mcq, ignore_index=True)

    # Load ages to compute years before diagnosis
    demo_ages = load_demo_ages()
    if not demo_ages.empty:
        mcq = mcq.merge(demo_ages[["SEQN", "age_at_exam"]], on="SEQN", how="left")
    else:
        mcq["age_at_exam"] = np.nan

    # Primary cancer age = min of all reported ages
    age_cols = [c for c in mcq.columns if c.startswith("cancer_age_")]
    if age_cols:
        mcq["cancer_age_first"] = mcq[age_cols].min(axis=1)
    else:
        mcq["cancer_age_first"] = np.nan

    # Years before exam = positive means diagnosed BEFORE exam
    mcq["years_since_diagnosis"] = mcq["age_at_exam"] - mcq["cancer_age_first"]

    # Cancer types across all slots
    type_cols = [c for c in mcq.columns if c.startswith("cancer_type_")]
    if type_cols:
        all_types = mcq[type_cols].values.flatten()
        all_types = pd.Series(all_types[~np.isnan(all_types.astype(float))]).astype(int)

    # Per-cancer-type flags
    for cancer_name, codes in HIGH_PRIORITY_CANCERS.items():
        mcq[f"cancer_{cancer_name}"] = mcq[type_cols].apply(
            lambda row: int(any(v in codes for v in row.dropna().astype(int))),
            axis=1,
        )

    # Map first cancer type to string label
    if "cancer_type_a" in mcq.columns:
        mcq["primary_cancer_type"] = mcq["cancer_type_a"].map(
            lambda x: CANCER_TYPE_MAP.get(int(x), "other") if not pd.isna(x) else None
        )

    keep_cols = (
        ["SEQN", "ever_cancer", "cancer_age_first", "age_at_exam",
         "years_since_diagnosis", "primary_cancer_type", "cycle"]
        + [f"cancer_{c}" for c in HIGH_PRIORITY_CANCERS]
    )
    keep_cols = [c for c in keep_cols if c in mcq.columns]
    return mcq[keep_cols].copy()


def run() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== Processing Outcomes ===\n")

    # 1. Mortality-based labels
    print("-- Mortality linkage --")
    mort = load_all_mortality()
    mort_labels = build_mortality_labels(mort) if not mort.empty else pd.DataFrame()

    # 2. MCQ-based labels
    print("\n-- MCQ self-report --")
    mcq_labels = build_mcq_labels()

    # 3. Merge
    if mort_labels.empty and mcq_labels.empty:
        raise RuntimeError("No outcome data loaded. Run download_nhanes.py first.")

    if not mort_labels.empty and not mcq_labels.empty:
        outcomes = mcq_labels.merge(
            mort_labels[["SEQN", "cancer_death", "cancer_death_3yr",
                          "cancer_death_5yr", "cancer_death_10yr",
                          "months_to_event"]],
            on="SEQN", how="outer"
        )
    elif not mort_labels.empty:
        outcomes = mort_labels
    else:
        outcomes = mcq_labels

    # ── Primary model target ─────────────────────────────────────────────────
    # Combine mortality + MCQ for a composite "any cancer outcome" flag:
    # 1 = confirmed cancer death within 5yr OR self-report cancer diagnosis
    outcomes["any_cancer"] = np.where(
        outcomes.get("ever_cancer", pd.Series(0, index=outcomes.index)).fillna(0).astype(bool) |
        outcomes.get("cancer_death", pd.Series(0, index=outcomes.index)).fillna(0).astype(bool),
        1, 0
    )

    out_path = PROCESSED_DIR / "outcomes.parquet"
    outcomes.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(outcomes):,} rows)")
    print(f"  Cancer deaths 5yr: {outcomes.get('cancer_death_5yr', pd.Series()).sum():.0f}")
    print(f"  Ever cancer (MCQ):  {outcomes.get('ever_cancer', pd.Series()).sum():.0f}")
    print(f"  Any cancer label:   {outcomes.get('any_cancer', pd.Series()).sum():.0f}")
    return outcomes


if __name__ == "__main__":
    run()
