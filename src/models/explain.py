"""
Generate SHAP-based explanations for individual patient predictions.

This module is used by the API to produce patient-level explanations:
  - Which labs are driving the risk score up or down
  - How each value compares to the population distribution
  - Natural-language interpretation of key findings

Also generates population-level importance plots for the frontend.
"""

import json
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from pathlib import Path

from src.features.lab_features import FEATURE_DISPLAY_NAMES

MODELS_DIR = Path(__file__).parents[2] / "models_saved"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


class CancerRiskExplainer:
    """Wraps model + SHAP explainer for patient-level risk explanations."""

    def __init__(self, target: str = "any_cancer"):
        self.target = target
        self.model = self._load_model(target)
        self.feature_cols = self._load_feature_cols()
        self.explainer = shap.TreeExplainer(self.model)
        self.expected_value = float(self.explainer.expected_value)
        self._load_threshold()

    def _load_model(self, target: str) -> xgb.XGBClassifier:
        name = target.replace("cancer_", "").replace("_5yr", "_5yr")
        candidates = [
            MODELS_DIR / f"model_{name}.json",
            MODELS_DIR / f"model_{target}.json",
            MODELS_DIR / "model_any_cancer.json",
        ]
        for path in candidates:
            if path.exists():
                model = xgb.XGBClassifier()
                model.load_model(str(path))
                return model
        raise FileNotFoundError(
            f"No model found for target '{target}'. Run: python -m src.models.train"
        )

    def _load_feature_cols(self) -> list[str]:
        path = MODELS_DIR / "feature_cols.json"
        if not path.exists():
            raise FileNotFoundError("feature_cols.json not found. Run training first.")
        with open(path) as f:
            return json.load(f)

    def _load_threshold(self) -> None:
        """Load calibrated operating threshold from threshold.json (default 0.5)."""
        path = MODELS_DIR / "threshold.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self.threshold = float(data.get("threshold", 0.5))
                self.operating_sensitivity = float(data.get("sensitivity", 0.0))
                self.operating_specificity = float(data.get("specificity", 0.0))
                return
            except Exception:
                pass
        # Defaults when file is absent
        self.threshold = 0.5
        self.operating_sensitivity = 0.0
        self.operating_specificity = 0.0

    def predict(self, lab_values: dict) -> dict:
        """
        Given a dictionary of lab values, return risk score + explanation.

        Args:
            lab_values: dict mapping canonical lab names to values
                        (e.g. {"wbc": 7.2, "hgb": 11.5, "age": 58, ...})

        Returns:
            {
              "risk_score": float (0-1),
              "risk_tier": str ("low" | "moderate" | "elevated" | "high"),
              "risk_percent": float (0-100),
              "top_drivers": list of {name, display_name, value, shap_value, direction, interpretation},
              "protective_factors": list of ...,
              "expected_value": float,
              "missing_labs": list of str,
            }
        """
        from src.features.lab_features import (
            add_absolute_counts, add_inflammatory_ratios, add_liver_markers,
            add_renal_markers, add_iron_markers, add_clinical_flags,
            add_composite_scores, add_age_sex_interactions,
        )

        # Build a one-row DataFrame from input
        row = {col: np.nan for col in self.feature_cols}
        for k, v in lab_values.items():
            if k in row:
                row[k] = v

        df = pd.DataFrame([row])

        # Run feature engineering pipeline
        df = add_absolute_counts(df)
        df = add_inflammatory_ratios(df)
        df = add_liver_markers(df)
        df = add_renal_markers(df)
        df = add_iron_markers(df)
        df = add_clinical_flags(df)
        df = add_composite_scores(df)
        df = add_age_sex_interactions(df)

        # Align to expected feature order
        X = np.array([[df[f].values[0] if f in df.columns else np.nan
                        for f in self.feature_cols]], dtype=np.float32)

        # Predict
        risk_score = float(self.model.predict_proba(X)[0][1])
        # Use calibrated threshold to set tier boundaries
        t = self.threshold

        # SHAP
        shap_values = self.explainer.shap_values(X)[0]

        # Build explanation
        contributions = []
        for i, (feat, shap_val) in enumerate(zip(self.feature_cols, shap_values)):
            if np.isnan(X[0][i]):
                continue
            contributions.append({
                "name":         feat,
                "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "value":        float(X[0][i]),
                "shap_value":   float(shap_val),
                "direction":    "risk" if shap_val > 0 else "protective",
            })

        # Sort by |SHAP|
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        # Add natural language interpretation
        for c in contributions:
            c["interpretation"] = self._interpret_feature(c["name"], c["value"], c["shap_value"])

        top_drivers   = [c for c in contributions if c["shap_value"] > 0.001][:8]
        protective    = [c for c in contributions if c["shap_value"] < -0.001][:5]

        missing_labs = [
            FEATURE_DISPLAY_NAMES.get(f, f)
            for f in ["wbc", "hgb", "plt", "albumin", "crp"]
            if f in self.feature_cols and np.isnan(X[0][self.feature_cols.index(f)])
        ]

        return {
            "risk_score":              risk_score,
            "risk_tier":               self._tier(risk_score, t),
            "risk_percent":            round(risk_score * 100, 1),
            "top_drivers":             top_drivers,
            "protective_factors":      protective,
            "expected_value":          self.expected_value,
            "missing_labs":            missing_labs,
            "shap_sum":                float(sum(shap_values)),
            "threshold":               self.threshold,
            "operating_sensitivity":   self.operating_sensitivity,
            "operating_specificity":   self.operating_specificity,
        }

    @staticmethod
    def _tier(score: float, threshold: float = 0.5) -> str:
        """
        Assign a risk tier relative to the calibrated operating threshold.

        Tier boundaries are scaled proportionally around the threshold so that
        scores at or above it are always "elevated" or "high".
        """
        if score < threshold * 0.5:
            return "low"
        if score < threshold:
            return "moderate"
        if score < threshold * 1.5:
            return "elevated"
        return "high"

    @staticmethod
    def _interpret_feature(name: str, value: float, shap_val: float) -> str:
        """Generate a 1-sentence clinical interpretation."""
        direction = "increases" if shap_val > 0 else "reduces"

        interpretations = {
            "nlr": lambda v: (
                f"NLR of {v:.1f} (≥3 is elevated) {direction} cancer risk — "
                "high NLR reflects immune dysregulation and systemic inflammation."
            ),
            "hgb": lambda v: (
                f"Hemoglobin of {v:.1f} g/dL {'(below normal)' if v < 12 else '(normal range)'} "
                f"{direction} risk — unexplained anemia is an early cancer signal."
            ),
            "plt": lambda v: (
                f"Platelet count {v:.0f} ×10⁹/L {'(elevated — thrombocytosis)' if v > 400 else ''} "
                f"{direction} risk — reactive thrombocytosis is associated with GI and lung cancers."
            ),
            "albumin": lambda v: (
                f"Albumin {v:.1f} g/dL {'(low)' if v < 3.5 else ''} {direction} risk — "
                "hypoalbuminemia reflects chronic inflammation and poor nutritional reserve."
            ),
            "crp": lambda v: (
                f"CRP {v:.1f} mg/L {'(elevated)' if v > 10 else ''} {direction} risk — "
                "elevated CRP indicates systemic inflammation associated with occult malignancy."
            ),
            "ldh": lambda v: (
                f"LDH {v:.0f} U/L {'(elevated)' if v > 250 else ''} {direction} risk — "
                "elevated LDH signals increased cellular turnover."
            ),
            "ferritin": lambda v: (
                f"Ferritin {v:.0f} ng/mL {direction} risk — "
                "ferritin is an acute-phase reactant elevated in malignancy."
            ),
            "age": lambda v: (
                f"Age {v:.0f} {direction} risk — cancer incidence rises sharply after age 50."
            ),
            "sii": lambda v: (
                f"SII {v:.0f} {direction} risk — "
                "the Systemic Immune-Inflammation Index captures combined CBC-based inflammation."
            ),
            "egfr": lambda v: (
                f"eGFR {v:.0f} mL/min/1.73m² {direction} risk — "
                "reduced kidney function may reflect cancer-related nephropathy."
            ),
        }

        if name in interpretations:
            try:
                return interpretations[name](value)
            except Exception:
                pass

        display = FEATURE_DISPLAY_NAMES.get(name, name)
        return f"{display} = {value:.2f} {direction} estimated cancer risk."


# ── Population-level importance (for frontend dashboard) ────────────────────

def load_population_importance(target: str = "any_cancer") -> pd.Series:
    """Return pre-computed population-level SHAP importance."""
    path = MODELS_DIR / f"feature_importance_{target}.json"
    if path.exists():
        return pd.read_json(path, typ="series").sort_values(ascending=False)
    # Fallback: load from model
    explainer = CancerRiskExplainer(target)
    path2 = PROCESSED_DIR / "model_dataset.parquet"
    if path2.exists():
        from src.features.lab_features import build_feature_matrix
        df = pd.read_parquet(path2)
        df, _ = build_feature_matrix(df)
        X = df[explainer.feature_cols].values.astype(np.float32)
        n = min(500, len(X))
        idx = np.random.choice(len(X), n, replace=False)
        sv = explainer.explainer.shap_values(X[idx])
        importance = pd.Series(np.abs(sv).mean(axis=0),
                                index=explainer.feature_cols).sort_values(ascending=False)
        importance.to_json(path)
        return importance
    return pd.Series(dtype=float)


if __name__ == "__main__":
    print("Loading explainer...")
    exp = CancerRiskExplainer("any_cancer")
    test_patient = {
        "age": 62,
        "is_male": 1,
        "wbc": 9.5,
        "rbc": 4.2,
        "hgb": 11.8,   # low — anemia flag
        "hct": 36.0,
        "plt": 420.0,  # thrombocytosis
        "mcv": 78.0,   # microcytic
        "neut_pct": 72.0,
        "lymph_pct": 18.0,
        "albumin": 3.3,   # low
        "crp": 18.0,      # elevated
        "ferritin": 380.0,
        "alt": 42.0,
        "ast": 38.0,
        "glucose": 105.0,
        "creatinine": 1.1,
    }
    result = exp.predict(test_patient)
    print(f"\nRisk Score: {result['risk_percent']:.1f}% ({result['risk_tier']})")
    print("\nTop drivers:")
    for d in result["top_drivers"][:5]:
        print(f"  {d['display_name']}: {d['value']:.2f}  SHAP={d['shap_value']:+.3f}")
        print(f"    → {d['interpretation']}")
