"""
Feature engineering for cancer signal detection from lab values.

Produces derived biomarkers that literature links to pre-diagnostic cancer:
  - NLR  (Neutrophil-to-Lymphocyte Ratio)    — systemic inflammation
  - PLR  (Platelet-to-Lymphocyte Ratio)       — immune dysfunction
  - MLR  (Monocyte-to-Lymphocyte Ratio)       — immune suppression
  - SII  (Systemic Immune-Inflammation Index) — combined index
  - SIRI (Systemic Immune-Response Index)
  - eGFR (CKD-EPI estimated GFR)             — renal function
  - De Ritis ratio (AST/ALT)                 — liver parenchyma damage
  - Ferritin/iron saturation
  - Anemia flags (WHO criteria)
  - Thrombocytosis flag
  - Eosinophilia flag (paraneoplastic)
  - Hypoalbuminemia flag (malnutrition/inflammation)
  - Hypercalcemia flag (paraneoplastic)
  - Elevated LDH flag (cell turnover)
  - Glasgow Prognostic Score (CRP + albumin)

References:
  - Templeton et al., J Clin Oncol 2014 (NLR)
  - Takenaka et al., BMC Cancer 2019 (SII)
  - Oken et al., Cancer 1982 (performance status proxy)
"""

import numpy as np
import pandas as pd


# ── Normal reference ranges (used for flag features) ────────────────────────
# Source: NIH/NHLBI + standard clinical references
REF_RANGES = {
    # CBC
    "wbc":       {"lo": 4.0,  "hi": 11.0},
    "rbc_m":     {"lo": 4.5,  "hi": 5.9},   # male
    "rbc_f":     {"lo": 4.0,  "hi": 5.2},   # female
    "hgb_m":     {"lo": 13.5, "hi": 17.5},  # male
    "hgb_f":     {"lo": 12.0, "hi": 15.5},  # female
    "hct_m":     {"lo": 41.0, "hi": 53.0},
    "hct_f":     {"lo": 36.0, "hi": 46.0},
    "plt":       {"lo": 150.0,"hi": 400.0},
    "mcv":       {"lo": 80.0, "hi": 100.0},
    "mch":       {"lo": 27.0, "hi": 33.0},
    "neut_pct":  {"lo": 40.0, "hi": 75.0},
    "lymph_pct": {"lo": 20.0, "hi": 45.0},
    # CMP
    "glucose":   {"lo": 70.0, "hi": 99.0},
    "bun":       {"lo": 7.0,  "hi": 20.0},
    "creatinine_m": {"lo": 0.74, "hi": 1.35},
    "creatinine_f": {"lo": 0.59, "hi": 1.04},
    "albumin":   {"lo": 3.5,  "hi": 5.0},
    "alt":       {"lo": 7.0,  "hi": 56.0},
    "ast":       {"lo": 10.0, "hi": 40.0},
    "alkphos":   {"lo": 44.0, "hi": 147.0},
    "bilirubin": {"lo": 0.1,  "hi": 1.2},
    "calcium":   {"lo": 8.5,  "hi": 10.2},
    "sodium":    {"lo": 136.0,"hi": 145.0},
    "potassium": {"lo": 3.5,  "hi": 5.0},
    # CRP
    "crp":       {"lo": 0.0,  "hi": 3.0},   # <3 = low inflammation
    # Ferritin (sex-specific)
    "ferritin_m":{"lo": 30.0, "hi": 400.0},
    "ferritin_f":{"lo": 13.0, "hi": 150.0},
}


def add_absolute_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Compute absolute cell counts from % differentials × WBC."""
    eps = 1e-6
    for cell, pct_col in [("neut", "neut_pct"), ("lymph", "lymph_pct"),
                           ("mono", "mono_pct"), ("eos", "eos_pct"),
                           ("baso", "baso_pct")]:
        if pct_col in df.columns and "wbc" in df.columns:
            df[f"{cell}_abs"] = df["wbc"] * df[pct_col] / 100.0
    return df


def add_inflammatory_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """NLR, PLR, MLR, SII, SIRI — systemic inflammation/immune indices."""
    eps = 1e-6
    lymph = df.get("lymph_abs", df.get("lymph_pct", pd.Series(np.nan, index=df.index)))
    neut  = df.get("neut_abs",  df.get("neut_pct",  pd.Series(np.nan, index=df.index)))
    mono  = df.get("mono_abs",  df.get("mono_pct",  pd.Series(np.nan, index=df.index)))
    plt   = df.get("plt",       pd.Series(np.nan, index=df.index))

    lymph = lymph.replace(0, np.nan)

    # NLR: Neutrophil-to-Lymphocyte Ratio
    df["nlr"] = neut / lymph

    # PLR: Platelet-to-Lymphocyte Ratio
    df["plr"] = plt / lymph

    # MLR: Monocyte-to-Lymphocyte Ratio
    df["mlr"] = mono / lymph

    # SII: Systemic Immune-Inflammation Index = PLT × Neutrophil / Lymphocyte
    df["sii"] = (plt * neut) / lymph

    # SIRI: Systemic Immune-Response Index = Monocyte × Neutrophil / Lymphocyte
    df["siri"] = (mono * neut) / lymph

    # Log-transform skewed ratios
    for col in ["nlr", "plr", "mlr", "sii", "siri"]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0).to_numpy(dtype=float, na_value=np.nan))

    return df


def add_liver_markers(df: pd.DataFrame) -> pd.DataFrame:
    """AST/ALT (De Ritis ratio), bilirubin/albumin, etc."""
    eps = 1e-6
    alt = df.get("alt", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    ast = df.get("ast", pd.Series(np.nan, index=df.index))
    alb = df.get("albumin", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    bili = df.get("bilirubin", pd.Series(np.nan, index=df.index))
    prot = df.get("total_protein", pd.Series(np.nan, index=df.index))
    alkphos = df.get("alkphos", pd.Series(np.nan, index=df.index))

    df["de_ritis_ratio"] = ast / alt                # >2 = alcoholic/cirrhotic
    df["bilirubin_albumin_ratio"] = bili / alb
    df["globulin"] = prot - alb.fillna(0)           # globulin = total protein - albumin
    df["ag_ratio"] = alb / df["globulin"].replace(0, np.nan)
    df["log_alkphos"] = np.log1p(alkphos.clip(lower=0).to_numpy(dtype=float, na_value=np.nan))
    df["log_alt"] = np.log1p(alt.clip(lower=0).to_numpy(dtype=float, na_value=np.nan))
    df["log_ast"] = np.log1p(ast.clip(lower=0).to_numpy(dtype=float, na_value=np.nan))
    return df


def add_renal_markers(df: pd.DataFrame) -> pd.DataFrame:
    """eGFR (CKD-EPI 2021), BUN/Creatinine ratio."""
    is_male = df.get("is_male", pd.Series(0.5, index=df.index))
    age = df.get("age", pd.Series(50.0, index=df.index))
    cr = df.get("creatinine", pd.Series(np.nan, index=df.index))

    # CKD-EPI 2021 (race-free)
    kappa = np.where(is_male == 1, 0.9, 0.7)
    alpha = np.where(is_male == 1, -0.302, -0.241)
    sex_coef = np.where(is_male == 1, 1.0, 1.012)

    cr_kappa = cr / kappa
    egfr = (142 *
            np.minimum(cr_kappa, 1.0) ** alpha *
            np.maximum(cr_kappa, 1.0) ** (-1.200) *
            (0.9938 ** age) *
            sex_coef)

    df["egfr"] = egfr.clip(1, 200)
    df["bun_cr_ratio"] = df.get("bun", pd.Series(np.nan, index=df.index)) / cr.replace(0, np.nan)
    return df


def add_iron_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Iron saturation, ferritin log-transform."""
    iron = df.get("iron", pd.Series(np.nan, index=df.index))
    tibc = df.get("tibc", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    ferritin = df.get("ferritin", pd.Series(np.nan, index=df.index))

    df["iron_saturation"] = (iron / tibc) * 100.0
    df["log_ferritin"] = np.log1p(ferritin.clip(lower=0).to_numpy(dtype=float, na_value=np.nan))
    return df


def add_clinical_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags for clinically significant abnormalities."""
    is_male = df.get("is_male", pd.Series(0.5, index=df.index))

    # Anemia (WHO: Hgb < 13 male, < 12 female)
    hgb = df.get("hgb", pd.Series(np.nan, index=df.index))
    hgb_threshold = np.where(is_male == 1, 13.0, 12.0)
    df["flag_anemia"] = (hgb < hgb_threshold).astype(float)

    # Microcytic anemia (MCV < 80 + anemia)
    mcv = df.get("mcv", pd.Series(np.nan, index=df.index))
    df["flag_microcytic_anemia"] = ((hgb < hgb_threshold) & (mcv < 80)).astype(float)

    # Thrombocytosis (PLT > 400) — paraneoplastic signal
    plt = df.get("plt", pd.Series(np.nan, index=df.index))
    df["flag_thrombocytosis"] = (plt > 400).astype(float)

    # Thrombocytopenia (PLT < 150)
    df["flag_thrombocytopenia"] = (plt < 150).astype(float)

    # Leukocytosis (WBC > 11)
    wbc = df.get("wbc", pd.Series(np.nan, index=df.index))
    df["flag_leukocytosis"] = (wbc > 11.0).astype(float)

    # Leukopenia (WBC < 4)
    df["flag_leukopenia"] = (wbc < 4.0).astype(float)

    # Neutrophilia (>75%)
    neut_pct = df.get("neut_pct", pd.Series(np.nan, index=df.index))
    df["flag_neutrophilia"] = (neut_pct > 75.0).astype(float)

    # Lymphopenia (<20%)
    lymph_pct = df.get("lymph_pct", pd.Series(np.nan, index=df.index))
    df["flag_lymphopenia"] = (lymph_pct < 20.0).astype(float)

    # Eosinophilia (>5%) — paraneoplastic
    eos_pct = df.get("eos_pct", pd.Series(np.nan, index=df.index))
    df["flag_eosinophilia"] = (eos_pct > 5.0).astype(float)

    # Hypoalbuminemia (<3.5) — malnutrition/chronic inflammation
    albumin = df.get("albumin", pd.Series(np.nan, index=df.index))
    df["flag_hypoalbuminemia"] = (albumin < 3.5).astype(float)

    # Hypercalcemia (Ca > 10.5) — paraneoplastic (PTHrP)
    calcium = df.get("calcium", pd.Series(np.nan, index=df.index))
    df["flag_hypercalcemia"] = (calcium > 10.5).astype(float)

    # Elevated LDH (>250 U/L) — cell turnover marker
    ldh = df.get("ldh", pd.Series(np.nan, index=df.index))
    df["flag_elevated_ldh"] = (ldh > 250.0).astype(float)

    # Elevated CRP (>10 mg/L) — systemic inflammation
    crp = df.get("crp", pd.Series(np.nan, index=df.index))
    df["flag_elevated_crp"] = (crp > 10.0).astype(float)

    # Elevated ALT (>56)
    alt = df.get("alt", pd.Series(np.nan, index=df.index))
    df["flag_elevated_alt"] = (alt > 56.0).astype(float)

    # Elevated bilirubin (>1.2)
    bili = df.get("bilirubin", pd.Series(np.nan, index=df.index))
    df["flag_elevated_bili"] = (bili > 1.2).astype(float)

    # NLR >= 3 (high systemic inflammation)
    nlr = df.get("nlr", pd.Series(np.nan, index=df.index))
    df["flag_high_nlr"] = (nlr >= 3.0).astype(float)

    # NLR >= 5 (very high — strong cancer risk predictor)
    df["flag_very_high_nlr"] = (nlr >= 5.0).astype(float)

    return df


def add_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Composite prognostic scores used in clinical oncology."""
    # Glasgow Prognostic Score (GPS)
    # 0: CRP ≤10 AND albumin ≥35
    # 1: CRP >10 OR albumin <35
    # 2: CRP >10 AND albumin <35
    crp = df.get("crp", pd.Series(np.nan, index=df.index))
    albumin = df.get("albumin", pd.Series(np.nan, index=df.index))
    crp_high = (crp > 10.0).astype(int)
    alb_low = (albumin < 3.5).astype(int)
    df["gps"] = (crp_high + alb_low).clip(0, 2)

    # Modified GPS (mGPS) — same but 0 if only low albumin without high CRP
    mgps = crp_high.copy()
    mgps[crp_high == 1] = 1 + alb_low[crp_high == 1]
    df["mgps"] = mgps

    # Pretreatment Prognostic Score (PPS) — simplified
    # High NLR + low albumin
    nlr = df.get("nlr", pd.Series(np.nan, index=df.index))
    df["pps"] = ((nlr >= 3.0).astype(int) + (albumin < 3.5).astype(int))

    return df


def add_age_sex_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Age × lab interactions — cancer risk is age-dependent."""
    age = df.get("age", pd.Series(50.0, index=df.index))
    is_male = df.get("is_male", pd.Series(0.5, index=df.index))

    # Age buckets
    df["age_50plus"] = (age >= 50).astype(float)
    df["age_60plus"] = (age >= 60).astype(float)
    df["age_70plus"] = (age >= 70).astype(float)
    df["age_sq"] = age ** 2 / 10000.0  # normalized

    # Key interactions: age × inflammation markers
    nlr = df.get("nlr", pd.Series(np.nan, index=df.index))
    df["age_x_nlr"] = age * nlr / 100.0

    hgb = df.get("hgb", pd.Series(np.nan, index=df.index))
    df["age_x_hgb_low"] = age * df.get("flag_anemia", pd.Series(0, index=df.index))

    plt = df.get("plt", pd.Series(np.nan, index=df.index))
    df["age_x_plt"] = age * plt / 10000.0

    # Male × PSA-age proxy (no PSA in NHANES, but prostate risk by age in males)
    df["male_age_50plus"] = is_male * (age >= 50).astype(float)

    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Full feature engineering pipeline. Returns (enriched_df, feature_columns).

    Call this on your model_dataset.parquet after loading.
    """
    df = df.copy()
    df = add_absolute_counts(df)
    df = add_inflammatory_ratios(df)
    df = add_liver_markers(df)
    df = add_renal_markers(df)
    df = add_iron_markers(df)
    df = add_clinical_flags(df)
    df = add_composite_scores(df)
    df = add_age_sex_interactions(df)

    # Clip extreme outliers on derived features
    ratio_cols = ["nlr", "plr", "mlr", "sii", "siri"]
    for col in ratio_cols:
        if col in df.columns:
            df[col] = df[col].clip(0, df[col].quantile(0.999))

    # Define the feature set for the model
    raw_lab_features = [
        "age", "is_male",
        "wbc", "rbc", "hgb", "hct", "plt", "mcv", "mch", "mchc", "rdw",
        "neut_pct", "lymph_pct", "mono_pct", "eos_pct", "baso_pct",
        "neut_abs", "lymph_abs", "mono_abs", "eos_abs",
        "glucose", "bun", "creatinine", "bilirubin", "alkphos",
        "alt", "ast", "albumin", "total_protein",
        "sodium", "potassium", "chloride", "co2",
        "calcium", "phosphorus", "uric_acid", "ldh",
        "ferritin", "iron", "tibc", "crp",
    ]

    derived_features = [
        "nlr", "plr", "mlr", "sii", "siri",
        "log_nlr", "log_plr", "log_mlr", "log_sii", "log_siri",
        "de_ritis_ratio", "bilirubin_albumin_ratio", "globulin", "ag_ratio",
        "log_alkphos", "log_alt", "log_ast",
        "egfr", "bun_cr_ratio",
        "iron_saturation", "log_ferritin",
        "gps", "mgps", "pps",
        "age_sq", "age_50plus", "age_60plus", "age_70plus",
        "age_x_nlr", "age_x_hgb_low", "age_x_plt", "male_age_50plus",
    ]

    flag_features = [c for c in df.columns if c.startswith("flag_")]

    all_features = raw_lab_features + derived_features + flag_features
    # Keep only features that exist and have >50% non-null
    feature_cols = [
        f for f in all_features
        if f in df.columns and df[f].notna().mean() > 0.05
    ]

    return df, feature_cols


# ── Human-readable feature names for the report UI ──────────────────────────
FEATURE_DISPLAY_NAMES = {
    "wbc":                    "White Blood Cell Count",
    "rbc":                    "Red Blood Cell Count",
    "hgb":                    "Hemoglobin",
    "hct":                    "Hematocrit",
    "plt":                    "Platelet Count",
    "mcv":                    "Mean Corpuscular Volume (MCV)",
    "mch":                    "Mean Corpuscular Hemoglobin (MCH)",
    "mchc":                   "Mean Corpuscular Hemoglobin Concentration (MCHC)",
    "rdw":                    "Red Cell Distribution Width (RDW)",
    "neut_pct":               "Neutrophils (%)",
    "lymph_pct":              "Lymphocytes (%)",
    "mono_pct":               "Monocytes (%)",
    "eos_pct":                "Eosinophils (%)",
    "baso_pct":               "Basophils (%)",
    # Derived inflammatory ratios
    "nlr":                    "Neutrophil-to-Lymphocyte Ratio (NLR)",
    "plr":                    "Platelet-to-Lymphocyte Ratio (PLR)",
    "mlr":                    "Monocyte-to-Lymphocyte Ratio (MLR)",
    "sii":                    "Systemic Immune-Inflammation Index (SII)",
    "siri":                   "Systemic Immune-Response Index (SIRI)",
    "log_nlr":                "Log Neutrophil-to-Lymphocyte Ratio",
    "log_plr":                "Log Platelet-to-Lymphocyte Ratio",
    "log_mlr":                "Log Monocyte-to-Lymphocyte Ratio",
    "log_sii":                "Log Systemic Immune-Inflammation Index",
    "log_siri":               "Log Systemic Immune-Response Index",
    # CMP
    "glucose":                "Blood Glucose",
    "bun":                    "Blood Urea Nitrogen (BUN)",
    "creatinine":             "Creatinine",
    "albumin":                "Albumin",
    "total_protein":          "Total Protein",
    "alt":                    "ALT (Liver Enzyme)",
    "ast":                    "AST (Liver Enzyme)",
    "alkphos":                "Alkaline Phosphatase",
    "bilirubin":              "Total Bilirubin",
    "calcium":                "Calcium",
    "sodium":                 "Sodium",
    "potassium":              "Potassium",
    "chloride":               "Chloride",
    "co2":                    "CO2 / Bicarbonate",
    "phosphorus":             "Phosphorus",
    "uric_acid":              "Uric Acid",
    # Iron studies
    "ferritin":               "Ferritin",
    "iron":                   "Serum Iron",
    "tibc":                   "Total Iron-Binding Capacity (TIBC)",
    "iron_saturation":        "Iron Saturation %",
    "log_ferritin":           "Log Ferritin",
    # Other labs
    "crp":                    "C-Reactive Protein (CRP)",
    "ldh":                    "Lactate Dehydrogenase (LDH)",
    # Derived liver markers
    "log_alt":                "Log ALT",
    "log_ast":                "Log AST",
    "log_alkphos":            "Log Alkaline Phosphatase",
    "de_ritis_ratio":         "De Ritis Ratio (AST/ALT)",
    "bilirubin_albumin_ratio":"Bilirubin/Albumin Ratio",
    "globulin":               "Globulin",
    "ag_ratio":               "Albumin/Globulin Ratio",
    # Renal
    "egfr":                   "eGFR (Kidney Function)",
    "bun_cr_ratio":           "BUN/Creatinine Ratio",
    # Absolute cell counts
    "neut_abs":               "Absolute Neutrophil Count",
    "lymph_abs":              "Absolute Lymphocyte Count",
    "mono_abs":               "Absolute Monocyte Count",
    "eos_abs":                "Absolute Eosinophil Count",
    "baso_abs":               "Absolute Basophil Count",
    # Composite scores
    "gps":                    "Glasgow Prognostic Score",
    "mgps":                   "Modified Glasgow Prognostic Score",
    "pps":                    "Pretreatment Prognostic Score",
    # Age / sex interactions
    "age":                    "Age",
    "is_male":                "Sex (Male)",
    "age_sq":                 "Age² (non-linear age effect)",
    "age_50plus":             "Age >50 flag",
    "age_60plus":             "Age >60 flag",
    "age_70plus":             "Age >70 flag",
    "age_x_nlr":              "Age × NLR interaction",
    "age_x_hgb_low":          "Age × Anemia interaction",
    "age_x_plt":              "Age × Platelet Count interaction",
    "male_age_50plus":        "Male age >50 interaction",
    # Clinical flags
    "flag_anemia":            "Anemia Flag",
    "flag_microcytic_anemia": "Microcytic Anemia Flag",
    "flag_thrombocytosis":    "Thrombocytosis Flag",
    "flag_thrombocytopenia":  "Thrombocytopenia Flag",
    "flag_leukocytosis":      "Leukocytosis Flag",
    "flag_leukopenia":        "Leukopenia Flag",
    "flag_neutrophilia":      "Neutrophilia Flag",
    "flag_lymphopenia":       "Lymphopenia Flag",
    "flag_eosinophilia":      "Eosinophilia Flag",
    "flag_hypoalbuminemia":   "Hypoalbuminemia Flag",
    "flag_hypercalcemia":     "Hypercalcemia Flag",
    "flag_elevated_ldh":      "Elevated LDH Flag",
    "flag_elevated_crp":      "Elevated CRP Flag",
    "flag_elevated_alt":      "Elevated ALT Flag",
    "flag_elevated_bili":     "Elevated Bilirubin Flag",
    "flag_high_nlr":          "High NLR (≥3)",
    "flag_very_high_nlr":     "Very High NLR (≥5)",
}


if __name__ == "__main__":
    from pathlib import Path
    PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
    df = pd.read_parquet(PROCESSED_DIR / "model_dataset.parquet")
    df_feat, features = build_feature_matrix(df)
    print(f"Feature matrix: {df_feat.shape}")
    print(f"Features ({len(features)}): {features[:20]} ...")
    missing = df_feat[features].isna().mean().sort_values(ascending=False)
    print("\nMissing rate (top 10):")
    print(missing.head(10))
