"""
Merge harmonized labs + outcomes into a single model-ready dataset.

Steps:
  1. Load labs_harmonized.parquet
  2. Load outcomes.parquet
  3. Join on SEQN
  4. Apply final exclusion criteria (age < 18, missing key labs)
  5. Save as model_dataset.parquet

Run: python -m src.data.merge_datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


def run() -> pd.DataFrame:
    print("\n=== Merging Labs + Outcomes ===\n")

    labs_path = PROCESSED_DIR / "labs_harmonized.parquet"
    outcomes_path = PROCESSED_DIR / "outcomes.parquet"

    if not labs_path.exists():
        raise FileNotFoundError("labs_harmonized.parquet not found. Run process_labs.py first.")
    if not outcomes_path.exists():
        raise FileNotFoundError("outcomes.parquet not found. Run process_outcomes.py first.")

    labs = pd.read_parquet(labs_path)
    outcomes = pd.read_parquet(outcomes_path)

    print(f"Labs:     {len(labs):,} rows")
    print(f"Outcomes: {len(outcomes):,} rows")

    df = labs.merge(outcomes.drop(columns=["cycle"], errors="ignore"), on="SEQN", how="inner")
    print(f"After merge: {len(df):,} rows")

    # ── Exclusion criteria ───────────────────────────────────────────────────
    n_before = len(df)

    # Adults only (18+)
    df = df[df["age"].fillna(0) >= 18]
    print(f"  After age >= 18: {len(df):,} (-{n_before - len(df):,})")
    n_before = len(df)

    # Must have at least WBC and hemoglobin (minimum viable CBC)
    df = df[df["wbc"].notna() & df["hgb"].notna()]
    print(f"  After WBC+HGB not null: {len(df):,} (-{n_before - len(df):,})")
    n_before = len(df)

    # Must have outcome label
    df = df[df["any_cancer"].notna()]
    print(f"  After outcome not null: {len(df):,} (-{n_before - len(df):,})")

    # ── Fill target columns ──────────────────────────────────────────────────
    for col in ["cancer_death_5yr", "cancer_death_3yr", "ever_cancer",
                "cancer_colorectal", "cancer_lung", "cancer_breast",
                "cancer_prostate", "cancer_leukemia", "cancer_lymphoma",
                "cancer_pancreatic"]:
        if col not in df.columns:
            df[col] = np.nan

    # ── Report class balance ─────────────────────────────────────────────────
    print(f"\nFinal dataset: {len(df):,} participants")
    print(f"  any_cancer = 1:  {df['any_cancer'].sum():.0f} ({df['any_cancer'].mean()*100:.1f}%)")

    out_path = PROCESSED_DIR / "model_dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return df


if __name__ == "__main__":
    run()
