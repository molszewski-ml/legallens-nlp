"""
LegalLens - Dataset Downloader
====================================
Downloads benchmark legal contract datasets for testing and evaluation.

Datasets:
    1. CUAD (Contract Understanding Atticus Dataset)
       - 510 commercial contracts, 13,000+ expert annotations, 41 clause categories
       - Source: The Atticus Project (NeurIPS 2021), License: CC BY 4.0

    2. ContractNLI
       - 607 annotated contracts for document-level NLI
       - Source: Stanford NLP / Hitachi America (EMNLP 2021), License: CC BY 4.0

Usage:
    conda activate legallens
    python download_datasets.py
"""

import json
import zipfile
import requests
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data" if (Path(__file__).parent / "data").exists() else Path("data")
CUAD_DIR = DATA_DIR / "cuad"
CNLI_DIR = DATA_DIR / "contract-nli"


def download_file(url: str, dest: Path, description: str):
    """Download a file with progress indication."""
    print(f"  Downloading {description}...")
    response = requests.get(url, stream=True, timeout=300, allow_redirects=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                print(f"\r  {pct}% ({downloaded // 1024} KB / {total // 1024} KB)", end="", flush=True)
    print()


def download_cuad():
    """
    Download CUAD dataset.
    Primary: GitHub (SQuAD JSON). Fallback: Zenodo (full ZIP with PDFs+TXTs).
    """
    print("[1/2] CUAD - Contract Understanding Atticus Dataset")
    print("     510 contracts | 13,000+ annotations | 41 categories")

    CUAD_DIR.mkdir(parents=True, exist_ok=True)

    cuad_json = CUAD_DIR / "CUADv1.json"

    if cuad_json.exists():
        print("  CUADv1.json already exists. Skipping.")
    else:
        # Try multiple known URLs for the JSON file
        urls = [
            "https://raw.githubusercontent.com/TheAtticusProject/cuad/main/data/CUADv1.json",
            "https://github.com/TheAtticusProject/cuad/raw/main/data/CUADv1.json",
        ]

        downloaded = False
        for url in urls:
            try:
                download_file(url, cuad_json, f"CUADv1.json")
                downloaded = True
                break
            except Exception as e:
                print(f"  Failed ({url}): {e}")
                cuad_json.unlink(missing_ok=True)

        if not downloaded:
            # Fallback: Zenodo full archive
            print("  Trying Zenodo full archive...")
            zen_url = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
            zen_zip = CUAD_DIR / "CUAD_v1.zip"
            try:
                download_file(zen_url, zen_zip, "CUAD from Zenodo (full archive)")
                print("  Extracting...")
                with zipfile.ZipFile(zen_zip, "r") as z:
                    z.extractall(CUAD_DIR)
                zen_zip.unlink()
            except Exception as e2:
                print(f"  Zenodo also failed: {e2}")
                print("  Manual download: https://zenodo.org/records/4595826")
                return

    # Also grab master_clauses.csv
    master_csv = CUAD_DIR / "master_clauses.csv"
    if not master_csv.exists():
        csv_url = "https://raw.githubusercontent.com/TheAtticusProject/cuad/main/data/master_clauses.csv"
        try:
            download_file(csv_url, master_csv, "master_clauses.csv")
        except Exception:
            print("  Could not download master_clauses.csv (optional)")

    # Dataset info
    info = {
        "name": "CUAD",
        "version": "1.0",
        "description": "Contract Understanding Atticus Dataset",
        "num_contracts": 510,
        "num_annotations": 13000,
        "num_categories": 41,
        "license": "CC BY 4.0",
        "citation": "Hendrycks et al., CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review, NeurIPS 2021",
        "url": "https://www.atticusprojectai.org/cuad",
    }
    with open(CUAD_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("  CUAD download complete.\n")


def download_contract_nli():
    """
    Download ContractNLI dataset from Stanford NLP.
    """
    print("[2/2] ContractNLI - Document-level NLI for Contracts")
    print("     607 contracts | hypothesis classification | evidence spans")

    CNLI_DIR.mkdir(parents=True, exist_ok=True)

    zip_url = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
    zip_dest = CNLI_DIR / "contract-nli.zip"

    # Check if already extracted (files may be in subdirectory)
    def find_json(name):
        return (CNLI_DIR / name).exists() or (CNLI_DIR / "contract-nli" / name).exists()

    if find_json("train.json") and find_json("dev.json") and find_json("test.json"):
        print("  ContractNLI already downloaded. Skipping.")
    else:
        if not zip_dest.exists():
            download_file(zip_url, zip_dest, "ContractNLI ZIP (~50 MB)")

        print("  Extracting...")
        with zipfile.ZipFile(zip_dest, "r") as z:
            z.extractall(CNLI_DIR)
        zip_dest.unlink()
        print(f"  Extracted to {CNLI_DIR}")

    # Dataset info
    info = {
        "name": "ContractNLI",
        "version": "1.0",
        "description": "Document-level Natural Language Inference for Contracts",
        "num_contracts": 607,
        "task": "NLI (entailment / contradiction / neutral) + evidence span identification",
        "license": "CC BY 4.0",
        "citation": "Koreeda & Manning, ContractNLI, Findings of EMNLP 2021",
        "url": "https://stanfordnlp.github.io/contract-nli/",
    }
    with open(CNLI_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("  ContractNLI download complete.\n")


def print_summary():
    """Print summary of downloaded datasets."""
    print("=" * 55)
    print(" Dataset download complete.")
    print("=" * 55)

    for name, path in [("CUAD", CUAD_DIR), ("ContractNLI", CNLI_DIR)]:
        if path.exists():
            total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            file_count = sum(1 for f in path.rglob("*") if f.is_file())
            print(f"\n {name}:")
            print(f"   Path:  {path}")
            print(f"   Files: {file_count}")
            print(f"   Size:  {total_size / (1024*1024):.1f} MB")

    print()


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("LegalLens Dataset Downloader")
    print("=" * 55)
    print()

    try:
        download_cuad()
        download_contract_nli()
        print_summary()
    except requests.ConnectionError:
        print("\n[ERROR] No internet connection.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
