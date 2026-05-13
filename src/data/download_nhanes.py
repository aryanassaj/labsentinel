"""
Download NHANES data files for cancer signal detection.

Covers cycles 1999-2018. Downloads:
- Demographics (DEMO)
- Complete Blood Count (CBC)
- Comprehensive Metabolic Panel (BIOPRO / biochemistry)
- Standard biochemistry (early cycles used different naming)
- Medical Conditions Questionnaire (MCQ) — contains cancer diagnosis
- NCHS Mortality Linkage files — the key to pre-diagnostic labeling

Run: python -m src.data.download_nhanes
"""

import os
import time
import requests
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
EXTERNAL_DIR = Path(__file__).parents[2] / "data" / "external"
# CDC changed their NHANES URL structure in 2024.
# New format: /Nchs/Data/Nhanes/Public/{start_year}/DataFiles/{filename}
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
MORTALITY_BASE = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality"

# ---------------------------------------------------------------------------
# NHANES file manifest
# Each entry: (cycle_label, local_name, start_year, server_filename)
# URL built as: BASE_URL/{start_year}/DataFiles/{server_filename}
# ---------------------------------------------------------------------------
NHANES_FILES = [
    # ── Demographics ────────────────────────────────────────────────────────
    ("1999-2000", "DEMO",    1999, "DEMO.XPT"),
    ("2001-2002", "DEMO_B",  2001, "DEMO_B.XPT"),
    ("2003-2004", "DEMO_C",  2003, "DEMO_C.XPT"),
    ("2005-2006", "DEMO_D",  2005, "DEMO_D.XPT"),
    ("2007-2008", "DEMO_E",  2007, "DEMO_E.XPT"),
    ("2009-2010", "DEMO_F",  2009, "DEMO_F.XPT"),
    ("2011-2012", "DEMO_G",  2011, "DEMO_G.XPT"),
    ("2013-2014", "DEMO_H",  2013, "DEMO_H.XPT"),
    ("2015-2016", "DEMO_I",  2015, "DEMO_I.XPT"),
    ("2017-2018", "DEMO_J",  2017, "DEMO_J.XPT"),

    # ── CBC (Complete Blood Count) ───────────────────────────────────────────
    ("1999-2000", "CBC",     1999, "LAB25.XPT"),
    ("2001-2002", "CBC_B",   2001, "L25_B.XPT"),
    ("2003-2004", "CBC_C",   2003, "L25_C.XPT"),
    ("2005-2006", "CBC_D",   2005, "CBC_D.XPT"),
    ("2007-2008", "CBC_E",   2007, "CBC_E.XPT"),
    ("2009-2010", "CBC_F",   2009, "CBC_F.XPT"),
    ("2011-2012", "CBC_G",   2011, "CBC_G.XPT"),
    ("2013-2014", "CBC_H",   2013, "CBC_H.XPT"),
    ("2015-2016", "CBC_I",   2015, "CBC_I.XPT"),
    ("2017-2018", "CBC_J",   2017, "CBC_J.XPT"),

    # ── Comprehensive Metabolic Panel ───────────────────────────────────────
    ("1999-2000", "BIOPRO",   1999, "LAB18.XPT"),
    ("2001-2002", "BIOPRO_B", 2001, "L40_B.XPT"),
    ("2003-2004", "BIOPRO_C", 2003, "L40_C.XPT"),
    ("2005-2006", "BIOPRO_D", 2005, "BIOPRO_D.XPT"),
    ("2007-2008", "BIOPRO_E", 2007, "BIOPRO_E.XPT"),
    ("2009-2010", "BIOPRO_F", 2009, "BIOPRO_F.XPT"),
    ("2011-2012", "BIOPRO_G", 2011, "BIOPRO_G.XPT"),
    ("2013-2014", "BIOPRO_H", 2013, "BIOPRO_H.XPT"),
    ("2015-2016", "BIOPRO_I", 2015, "BIOPRO_I.XPT"),
    ("2017-2018", "BIOPRO_J", 2017, "BIOPRO_J.XPT"),

    # ── Medical Conditions Questionnaire (cancer Qs live here) ──────────────
    ("1999-2000", "MCQ",    1999, "MCQ.XPT"),
    ("2001-2002", "MCQ_B",  2001, "MCQ_B.XPT"),
    ("2003-2004", "MCQ_C",  2003, "MCQ_C.XPT"),
    ("2005-2006", "MCQ_D",  2005, "MCQ_D.XPT"),
    ("2007-2008", "MCQ_E",  2007, "MCQ_E.XPT"),
    ("2009-2010", "MCQ_F",  2009, "MCQ_F.XPT"),
    ("2011-2012", "MCQ_G",  2011, "MCQ_G.XPT"),
    ("2013-2014", "MCQ_H",  2013, "MCQ_H.XPT"),
    ("2015-2016", "MCQ_I",  2015, "MCQ_I.XPT"),
    ("2017-2018", "MCQ_J",  2017, "MCQ_J.XPT"),

    # ── Ferritin / Iron ──────────────────────────────────────────────────────
    # Note: NHANES did not release standalone ferritin files for 2011-2012 and 2013-2014.
    ("1999-2000", "FETIB",   1999, "LAB06.XPT"),
    ("2001-2002", "FETIB_B", 2001, "L06_B.XPT"),
    ("2003-2004", "FETIB_C", 2003, "L06TFR_C.XPT"),
    ("2005-2006", "FETIB_D", 2005, "FERTIN_D.XPT"),
    ("2007-2008", "FETIB_E", 2007, "FERTIN_E.XPT"),
    ("2009-2010", "FETIB_F", 2009, "FERTIN_F.XPT"),
    # 2011-2012 and 2013-2014: ferritin not released as standalone NHANES file — skipped
    ("2015-2016", "FETIB_I", 2015, "FERTIN_I.XPT"),
    ("2017-2018", "FETIB_J", 2017, "FERTIN_J.XPT"),

    # ── C-Reactive Protein (inflammation marker) ────────────────────────────
    # Note: NHANES discontinued standalone CRP panels for 2011-2012 and 2013-2014.
    # From 2015 onward the file is named HSCRP (high-sensitivity CRP).
    ("1999-2000", "CRP",   1999, "LAB11.XPT"),
    ("2001-2002", "CRP_B", 2001, "L11_B.XPT"),
    ("2003-2004", "CRP_C", 2003, "L11_C.XPT"),
    ("2005-2006", "CRP_D", 2005, "CRP_D.XPT"),
    ("2007-2008", "CRP_E", 2007, "CRP_E.XPT"),
    ("2009-2010", "CRP_F", 2009, "CRP_F.XPT"),
    # 2011-2012 and 2013-2014: CRP not measured in NHANES — skipped
    ("2015-2016", "CRP_I", 2015, "HSCRP_I.XPT"),
    ("2017-2018", "CRP_J", 2017, "HSCRP_J.XPT"),
]

# NCHS Linked Mortality files — one per cycle, fixed-width text
MORTALITY_FILES = [
    "NHANES_1999_2000_MORT_2019_PUBLIC.dat",
    "NHANES_2001_2002_MORT_2019_PUBLIC.dat",
    "NHANES_2003_2004_MORT_2019_PUBLIC.dat",
    "NHANES_2005_2006_MORT_2019_PUBLIC.dat",
    "NHANES_2007_2008_MORT_2019_PUBLIC.dat",
    "NHANES_2009_2010_MORT_2019_PUBLIC.dat",
    "NHANES_2011_2012_MORT_2019_PUBLIC.dat",
    "NHANES_2013_2014_MORT_2019_PUBLIC.dat",
    "NHANES_2015_2016_MORT_2019_PUBLIC.dat",
]


def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Download a file with retry logic and progress bar.

    Guards against CDC's soft-404 behaviour: their server returns HTTP 200
    with an HTML "Page Not Found" body instead of a real 404 status code.
    We reject any response whose Content-Type is text/html.
    """
    if dest.exists():
        # Re-check: if the existing file is an HTML error page, delete it.
        if dest.stat().st_size < 50_000:
            try:
                with open(dest, "rb") as f:
                    snippet = f.read(20)
                if snippet.strip().lower().startswith(b"<!doctype") or snippet.strip().lower().startswith(b"<html"):
                    print(f"  [stale HTML] removing {dest.name} and re-downloading")
                    dest.unlink()
                else:
                    print(f"  [skip] {dest.name} already exists")
                    return True
            except Exception:
                print(f"  [skip] {dest.name} already exists")
                return True
        else:
            print(f"  [skip] {dest.name} already exists")
            return True

    for attempt in range(retries):
        try:
            resp = requests.get(url, stream=True, timeout=60)
            if resp.status_code == 404:
                print(f"  [404]  {url}")
                return False
            resp.raise_for_status()

            # Reject HTML responses — CDC returns 200 + HTML for missing files
            content_type = resp.headers.get("content-type", "")
            if "text/html" in content_type:
                print(f"  [404/html] {url}")
                return False

            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=dest.name,
                leave=False,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            return True

        except Exception as exc:
            print(f"  [err]  attempt {attempt+1}/{retries}: {exc}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return False


def download_nhanes_labs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Downloading NHANES lab + MCQ files to {RAW_DIR} ===\n")

    ok, fail = 0, 0
    for cycle, name, start_year, server_filename in NHANES_FILES:
        url = f"{BASE_URL}/{start_year}/DataFiles/{server_filename}"
        dest = RAW_DIR / f"{name}.XPT"
        print(f"[{cycle}] {name}")
        if download_file(url, dest):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} downloaded, {fail} failed/missing")


def download_mortality_linkage() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Downloading NCHS Mortality Linkage files to {EXTERNAL_DIR} ===\n")

    ok, fail = 0, 0
    for fname in MORTALITY_FILES:
        url = f"{MORTALITY_BASE}/{fname}"
        dest = EXTERNAL_DIR / fname
        print(f"  {fname}")
        if download_file(url, dest):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} downloaded, {fail} failed/missing")


if __name__ == "__main__":
    download_nhanes_labs()
    download_mortality_linkage()
    print("\nAll downloads complete. Run `python -m src.data.process_labs` next.")
