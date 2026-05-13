"""
End-to-end pipeline runner.

Steps:
  1. Download NHANES data
  2. Process labs
  3. Process outcomes
  4. Merge datasets
  5. Train model
  6. Start API server

Usage:
  python run_pipeline.py                 # Full pipeline + server
  python run_pipeline.py --skip-download # Skip re-downloading files
  python run_pipeline.py --only-train    # Only train (data already processed)
  python run_pipeline.py --only-server   # Only start server (model already trained)
  python run_pipeline.py --tune          # Full pipeline with Optuna tuning
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROCESSED = Path("data/processed")
MODELS    = Path("models_saved")


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {desc}")
    print('─'*60)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed: {desc}")
        print("Check the output above. Continuing to next step...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--only-train",    action="store_true")
    parser.add_argument("--only-server",   action="store_true")
    parser.add_argument("--tune",          action="store_true")
    parser.add_argument("--port",          default="8000")
    args = parser.parse_args()

    py = sys.executable

    if args.only_server:
        run(["uvicorn", "api.main:app", "--reload", "--port", args.port],
            "Starting API server")
        return

    if args.only_train:
        cmd = [py, "-m", "src.models.train"]
        if args.tune:
            cmd.append("--tune")
        run(cmd, "Training model")
        run(["uvicorn", "api.main:app", "--reload", "--port", args.port],
            "Starting API server")
        return

    # Full pipeline
    if not args.skip_download:
        run([py, "-m", "src.data.download_nhanes"], "Downloading NHANES data")

    run([py, "-m", "src.data.process_labs"],    "Processing lab data")
    run([py, "-m", "src.data.process_outcomes"],"Processing cancer outcomes")
    run([py, "-m", "src.data.merge_datasets"],  "Merging datasets")

    cmd = [py, "-m", "src.models.train"]
    if args.tune:
        cmd.append("--tune")
    run(cmd, "Training XGBoost model")

    print(f"\n{'='*60}")
    print(f"  Pipeline complete! Starting server on port {args.port}")
    print(f"  Open http://localhost:{args.port}")
    print('='*60)
    run(["uvicorn", "api.main:app", "--reload", "--port", args.port],
        "Starting API server")


if __name__ == "__main__":
    main()
