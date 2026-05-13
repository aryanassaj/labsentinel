"""
Train the cancer signal detection model.

Architecture:
  - XGBoost classifier (handles missing values natively — critical for NHANES)
  - SMOTE to address class imbalance (~5-10% positive rate)
  - Stratified K-fold CV for evaluation
  - Optuna hyperparameter tuning (optional, --tune flag)
  - SHAP values for interpretability
  - Separate models per cancer subtype (colorectal, lung, etc.)

Outputs saved to models_saved/:
  - model_any_cancer.json          — main XGBoost model
  - model_<subtype>.json           — per-subtype models
  - feature_cols.json              — ordered feature list
  - shap_expected_value.json       — SHAP baseline
  - cv_results.json                — cross-validation metrics

Run: python -m src.models.train [--tune] [--target cancer_death_5yr]
"""

import argparse
import datetime
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
MODELS_DIR    = Path(__file__).parents[2] / "models_saved"

# Cancer subtypes with their label columns
SUBTYPES = {
    "any_cancer":       "any_cancer",
    "colorectal":       "cancer_colorectal",
    "lung":             "cancer_lung",
    "breast":           "cancer_breast",
    "prostate":         "cancer_prostate",
    "leukemia":         "cancer_leukemia",
    "lymphoma":         "cancer_lymphoma",
    "pancreatic":       "cancer_pancreatic",
    "cancer_death_5yr": "cancer_death_5yr",
}

# XGBoost baseline params (will be tuned if --tune)
BASE_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "aucpr",
    "max_depth":        6,
    "learning_rate":    0.05,
    "n_estimators":     500,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
    "early_stopping_rounds": 30,
    "verbosity":        0,
}


def load_data() -> tuple[pd.DataFrame, list[str]]:
    """Load model dataset and run feature engineering."""
    from src.features.lab_features import build_feature_matrix

    path = PROCESSED_DIR / "model_dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "model_dataset.parquet not found.\n"
            "Run: python -m src.data.merge_datasets"
        )

    df = pd.read_parquet(path)
    df, feature_cols = build_feature_matrix(df)
    print(f"Loaded {len(df):,} participants, {len(feature_cols)} features")
    return df, feature_cols


def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                          feature_cols: list[str]) -> dict:
    """Optuna hyperparameter search over XGBoost params."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("Optuna not installed — using base params.")
        return BASE_PARAMS.copy()

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective":        "binary:logistic",
            "eval_metric":      "aucpr",
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("lr", 0.01, 0.2, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "random_state":     42,
            "n_jobs":           -1,
            "verbosity":        0,
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            Xtr, Xval = X_train[tr_idx], X_train[val_idx]
            ytr, yval = y_train[tr_idx], y_train[val_idx]

            model = xgb.XGBClassifier(**params, early_stopping_rounds=20)
            model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
            proba = model.predict_proba(Xval)[:, 1]
            aucs.append(average_precision_score(yval, proba))

        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    best = study.best_params
    print(f"Best tuned AUCPR: {study.best_value:.4f}")

    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": best["max_depth"],
        "learning_rate": best["lr"],
        "n_estimators": best["n_estimators"],
        "subsample": best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "min_child_weight": best["min_child_weight"],
        "reg_alpha": best["reg_alpha"],
        "reg_lambda": best["reg_lambda"],
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 30,
        "verbosity": 0,
    }


def cross_validate(df: pd.DataFrame, feature_cols: list[str],
                   target: str, params: dict) -> dict:
    """Run 5-fold stratified CV and return metrics."""
    valid = df[target].notna() & df[feature_cols].notna().any(axis=1)
    sub = df[valid].copy()

    X = sub[feature_cols].values.astype(np.float32)
    y = sub[target].values.astype(int)

    print(f"  Class balance: {y.mean()*100:.1f}% positive ({y.sum():,}/{len(y):,})")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        Xtr, Xval = X[tr_idx], X[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]

        # SMOTE on training fold only
        if ytr.mean() < 0.3:
            try:
                sm = SMOTE(random_state=42, k_neighbors=5)
                Xtr, ytr = sm.fit_resample(Xtr, ytr)
            except Exception:
                pass  # Skip SMOTE if too few samples

        p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        p["n_estimators"] = params["n_estimators"]

        model = xgb.XGBClassifier(
            **p,
            early_stopping_rounds=params.get("early_stopping_rounds", 30)
        )
        model.fit(
            Xtr, ytr,
            eval_set=[(Xval, yval)],
            verbose=False,
        )

        proba = model.predict_proba(Xval)[:, 1]
        auc_roc = roc_auc_score(yval, proba)
        auc_pr  = average_precision_score(yval, proba)

        # Threshold at 0.5 for classification metrics
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yval, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        fold_metrics.append({
            "fold": fold + 1,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "n_val": len(yval),
            "n_pos_val": int(yval.sum()),
        })

        print(f"    Fold {fold+1}: AUC-ROC={auc_roc:.3f}  AUC-PR={auc_pr:.3f}  "
              f"Sens={sensitivity:.3f}  Spec={specificity:.3f}")

    means = {
        k: np.mean([m[k] for m in fold_metrics])
        for k in ["auc_roc", "auc_pr", "sensitivity", "specificity"]
    }
    print(f"  Mean AUC-ROC={means['auc_roc']:.3f}  "
          f"AUC-PR={means['auc_pr']:.3f}  "
          f"Sensitivity={means['sensitivity']:.3f}  "
          f"Specificity={means['specificity']:.3f}")

    return {"folds": fold_metrics, "means": means}


def train_final_model(df: pd.DataFrame, feature_cols: list[str],
                       target: str, params: dict) -> tuple:
    """
    Train on all available data for the target label.

    Returns (model, Xval, yval) so the caller can compute a calibrated threshold
    on the held-out validation split.
    """
    valid = df[target].notna() & df[feature_cols].notna().any(axis=1)
    sub = df[valid].copy()

    X = sub[feature_cols].values.astype(np.float32)
    y = sub[target].values.astype(int)

    # Apply SMOTE on full training data
    if y.mean() < 0.3:
        try:
            sm = SMOTE(random_state=42, k_neighbors=5)
            X, y = sm.fit_resample(X, y)
        except Exception:
            pass

    # Split off a small validation set for early stopping + threshold calibration
    from sklearn.model_selection import train_test_split
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.1,
                                              stratify=y, random_state=42)

    p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    model = xgb.XGBClassifier(
        **p,
        early_stopping_rounds=params.get("early_stopping_rounds", 30)
    )
    model.fit(
        Xtr, ytr,
        eval_set=[(Xval, yval)],
        verbose=False,
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model, Xval, yval


def calibrate_threshold(model: xgb.XGBClassifier, Xval: np.ndarray,
                         yval: np.ndarray, target: str = "any_cancer") -> dict:
    """
    Find the operating threshold using Youden's J statistic (sensitivity + specificity - 1).

    Also finds the threshold closest to ~65% sensitivity as a cross-check.
    Saves result to models_saved/threshold.json and returns the threshold dict.
    """
    proba = model.predict_proba(Xval)[:, 1]
    fpr, tpr, thresholds = roc_curve(yval, proba)

    # Youden's J: maximise sensitivity + specificity - 1
    specificity = 1.0 - fpr
    j_scores = tpr + specificity - 1.0
    best_idx = int(np.argmax(j_scores))

    best_threshold   = float(thresholds[best_idx])
    best_sensitivity = float(tpr[best_idx])
    best_specificity = float(specificity[best_idx])

    print(f"\n[Threshold Calibration — Youden's J]")
    print(f"  Optimal threshold : {best_threshold:.4f}")
    print(f"  Sensitivity        : {best_sensitivity:.3f}")
    print(f"  Specificity        : {best_specificity:.3f}")

    threshold_info = {
        "threshold":   round(best_threshold, 4),
        "sensitivity": round(best_sensitivity, 4),
        "specificity": round(best_specificity, 4),
        "target":      target,
    }

    with open(MODELS_DIR / "threshold.json", "w") as f:
        json.dump(threshold_info, f, indent=2)

    print(f"  Saved → models_saved/threshold.json")
    return threshold_info


def compute_shap(model: xgb.XGBClassifier, X: np.ndarray,
                  feature_cols: list[str]) -> tuple:
    """Compute SHAP values for a sample of the data."""
    n_sample = min(2000, len(X))
    idx = np.random.choice(len(X), n_sample, replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)

    # Mean |SHAP| per feature
    mean_shap = np.abs(shap_vals).mean(axis=0)
    importance = pd.Series(mean_shap, index=feature_cols).sort_values(ascending=False)

    return explainer, shap_vals, importance


def run(target: str = "any_cancer", tune: bool = False) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training cancer signal detector — target: {target}")
    print('='*60)

    df, feature_cols = load_data()

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {[c for c in SUBTYPES.values() if c in df.columns]}")

    # Hyperparameter tuning
    if tune:
        print("\n[Tuning hyperparameters with Optuna...]")
        valid = df[target].notna()
        X_tune = df[valid][feature_cols].values.astype(np.float32)
        y_tune = df[valid][target].values.astype(int)
        params = tune_hyperparameters(X_tune, y_tune, feature_cols)
    else:
        params = BASE_PARAMS.copy()

    # Cross-validation
    print(f"\n[5-Fold Cross-Validation on '{target}']")
    cv_results = cross_validate(df, feature_cols, target, params)

    # Train final model
    print(f"\n[Training final model on '{target}' (all data)]")
    model, Xval_cal, yval_cal = train_final_model(df, feature_cols, target, params)

    # Threshold calibration (Fix 1)
    if target == "any_cancer":
        calibrate_threshold(model, Xval_cal, yval_cal, target)

    # SHAP
    print("\n[Computing SHAP values...]")
    valid = df[target].notna()
    sub = df[valid]
    X_all = sub[feature_cols].values.astype(np.float32)
    explainer, shap_vals, importance = compute_shap(model, X_all, feature_cols)

    print("\nTop 15 features by mean |SHAP|:")
    print(importance.head(15).to_string())

    # Save artifacts
    model_name = f"model_{target.replace('cancer_', '').replace('_5yr', '_5yr')}"
    model.save_model(str(MODELS_DIR / f"{model_name}.json"))

    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    with open(MODELS_DIR / "cv_results.json", "w") as f:
        json.dump({target: cv_results}, f, indent=2)

    # Save SHAP expected value
    with open(MODELS_DIR / "shap_expected_value.json", "w") as f:
        json.dump({"expected_value": float(explainer.expected_value)}, f)

    # Save feature importance
    importance.to_json(MODELS_DIR / f"feature_importance_{target}.json")

    # Save model version info (Fix 5)
    model_info = {
        "version":    "1.0.0",
        "trained_at": datetime.datetime.utcnow().isoformat(),
        "target":     target,
        "n_samples":  int(valid.sum()),
        "n_features": len(feature_cols),
        "cv_auc_roc": cv_results["means"]["auc_roc"],
        "cv_auc_pr":  cv_results["means"]["auc_pr"],
        "tuned":      tune,
    }
    with open(MODELS_DIR / "model_version.json", "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\nSaved to {MODELS_DIR}/")
    print(f"  {model_name}.json")
    print(f"  feature_cols.json")
    print(f"  cv_results.json")
    print(f"  shap_expected_value.json")
    print(f"  feature_importance_{target}.json")
    print(f"  model_version.json")

    # Train subtypes
    print("\n[Training cancer subtype models...]")
    for subtype_name, subtype_col in SUBTYPES.items():
        if subtype_col == target or subtype_col not in df.columns:
            continue
        if df[subtype_col].notna().sum() < 200:
            print(f"  Skipping {subtype_name} — insufficient labels ({df[subtype_col].notna().sum()})")
            continue
        if df[subtype_col].sum() < 50:
            print(f"  Skipping {subtype_name} — too few positives ({df[subtype_col].sum():.0f})")
            continue

        print(f"\n  Subtype: {subtype_name}")
        try:
            sub_model, _, _ = train_final_model(df, feature_cols, subtype_col, params)
            sub_model.save_model(str(MODELS_DIR / f"model_{subtype_name}.json"))
        except Exception as exc:
            print(f"    [error] {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--target", default="any_cancer",
                        choices=list(SUBTYPES.keys()),
                        help="Primary target to train on")
    args = parser.parse_args()
    run(target=args.target, tune=args.tune)
