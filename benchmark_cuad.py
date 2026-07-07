"""
LegalLens - CUAD & ContractNLI Benchmark
==========================================
Runs LegalLens pipeline on sample contracts and compares detected
risk categories against expert annotations.

Usage:
    python benchmark_cuad.py
    python benchmark_cuad.py --sample 20
    python benchmark_cuad.py --no-llm
    python benchmark_cuad.py --dataset contractnli --sample 10

Output:
    - Console: per-category precision, recall, F1
    - File: output/benchmark_results.json
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

from src.pipeline import LegalLensPipeline


# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
CUAD_JSON = PROJECT_ROOT / "data" / "cuad" / "CUAD_v1" / "CUAD_v1.json"
CUAD_TXT_DIR = PROJECT_ROOT / "data" / "cuad" / "CUAD_v1" / "full_contract_txt"
CNLI_DIR = PROJECT_ROOT / "data" / "contract-nli" / "contract-nli"
OUTPUT_DIR = PROJECT_ROOT / "output"


# --- CUAD category -> lexicon snake_case category ---
# Lexicon uses snake_case internally. This mapping must match that.
CUAD_TO_LEGALLENS = {
    "Termination For Convenience": "termination",
    "Rofr/Rofo/Rofn": "termination",
    "Anti-Assignment": "anti_assignment",
    "Change Of Control": "change_of_control",
    "Non-Compete": "non_compete",
    "Exclusivity": "exclusivity",
    "No-Solicit Of Customers": "non_solicitation",
    "No-Solicit Of Employees": "non_solicitation",
    "Non-Disparagement": "non_disparagement",
    "Competitive Restriction Exception": "non_compete",
    "Covenant Not To Sue": "covenant_not_to_sue",
    "Third Party Beneficiary": "third_party_beneficiary",
    "Governing Law": "governing_law",
    "Most Favored Nation": "most_favored_nation",
    "Revenue/Profit Sharing": "revenue_profit_sharing",
    "Price Restrictions": "price_restrictions",
    "Minimum Commitment": "minimum_commitment",
    "Volume Restriction": "volume_restriction",
    "Ip Ownership Assignment": "ip_ownership_assignment",
    "Joint Ip Ownership": "ip_ownership_assignment",
    "License Grant": "license_grant",
    "Non-Transferable License": "license_grant",
    "Affiliate License-Loss Of Rights": "license_grant",
    "Unlimited/All-You-Can-Eat License": "license_grant",
    "Irrevocable Or Perpetual License": "irrevocable_perpetual_license",
    "Source Code Escrow": "source_code_escrow",
    "Post-Termination Services": "post_termination_services",
    "Audit Rights": "audit_rights",
    "Uncapped Liability": "uncapped_liability",
    "Cap On Liability": "uncapped_liability",
    "Liquidated Damages": "liquidated_damages",
    "Warranty Duration": "warranty",
    "Insurance": "insurance",
    "Expiration Date": "expiration",
    "Renewal Term": "renewal",
    "Notice Period To Terminate Renewal": "termination",
    "Effective Date": "effective_date",
}


# --- ContractNLI hypothesis -> lexicon category ---
CNLI_TO_LEGALLENS = {
    "Confidentiality of Agreement": "confidentiality",
    "Non-Compete": "non_compete",
    "No-Solicit": "non_solicitation",
    "Non-Disparagement": "non_disparagement",
    "Termination For Convenience": "termination",
    "Return of Confidential Information": "confidentiality",
    "Survival": "post_termination_services",
    "Governing Law": "governing_law",
}


def load_cuad_annotations() -> dict:
    """
    Load CUAD_v1.json and extract per-contract category annotations.

    Returns:
        dict mapping contract name to set of snake_case categories present.
    """
    with open(CUAD_JSON, "r", encoding="utf-8") as f:
        cuad = json.load(f)

    contracts = defaultdict(set)

    for item in cuad["data"]:
        contract_name = item["title"]

        for paragraph in item["paragraphs"]:
            for qa in paragraph["qas"]:
                # Check if category is present (has non-empty answers)
                answers = qa.get("answers", [])
                has_answer = any(a.get("text", "").strip() for a in answers)

                if has_answer:
                    category = _extract_cuad_category(qa)
                    if category:
                        contracts[contract_name].add(category)

    return dict(contracts)


def _extract_cuad_category(qa: dict) -> str:
    """Extract the mapped LegalLens category from a CUAD QA entry."""
    qid = qa.get("id", "")

    for cuad_cat, ll_cat in CUAD_TO_LEGALLENS.items():
        normalized = cuad_cat.lower().replace(" ", "").replace("-", "").replace("/", "")
        qid_normalized = qid.lower().replace(" ", "").replace("-", "").replace("_", "").replace("/", "")
        if normalized in qid_normalized:
            return ll_cat

    question = qa.get("question", "")
    for cuad_cat, ll_cat in CUAD_TO_LEGALLENS.items():
        if cuad_cat.lower() in question.lower():
            return ll_cat

    return ""


def load_contractnli_annotations() -> list[dict]:
    """
    Load ContractNLI test.json and extract per-document hypothesis labels.

    Returns:
        list of dicts with 'text', 'name', and 'expected_categories'.
    """
    test_path = CNLI_DIR / "test.json"
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for doc_id, doc_data in data.get("documents", {}).items():
        text = doc_data.get("text", "")
        if not text.strip():
            continue

        expected = set()
        for ann in doc_data.get("annotations", []):
            hypothesis = ann.get("hypothesis", "")
            label = ann.get("label", "")
            if label == "Entailment":
                mapped = CNLI_TO_LEGALLENS.get(hypothesis)
                if mapped:
                    expected.add(mapped)

        if expected:
            documents.append({
                "name": doc_id,
                "text": text,
                "expected": expected,
            })

    return documents


def find_txt_file(contract_name: str) -> Path | None:
    """Find the TXT file for a given CUAD contract name."""
    txt_path = CUAD_TXT_DIR / contract_name
    if txt_path.exists():
        return txt_path

    if not contract_name.endswith(".txt"):
        txt_path = CUAD_TXT_DIR / f"{contract_name}.txt"
        if txt_path.exists():
            return txt_path

    stem = Path(contract_name).stem.lower()
    for f in CUAD_TXT_DIR.iterdir():
        if f.stem.lower() == stem:
            return f

    return None


def _get_detected_categories(result) -> set:
    """Extract detected category names from pipeline result."""
    detected = set()
    for match in result.lexicon_matches:
        cat = getattr(match, "category", None)
        if cat is None and isinstance(match, dict):
            cat = match.get("category")
        if cat:
            detected.add(cat)
    return detected


def _compute_metrics(tp: dict, fp: dict, fn: dict) -> list[dict]:
    """Compute per-category precision, recall, F1."""
    all_categories = sorted(set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())))
    metrics = []

    for cat in all_categories:
        t = tp[cat]
        f_p = fp[cat]
        f_n = fn[cat]

        precision = t / (t + f_p) if (t + f_p) > 0 else 0
        recall = t / (t + f_n) if (t + f_n) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics.append({
            "category": cat,
            "tp": t, "fp": f_p, "fn": f_n,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        })

    return metrics


def _print_metrics(metrics: list[dict], tp: dict, fp: dict, fn: dict):
    """Print formatted metrics table."""
    for m in metrics:
        print(f"  {m['category']:<35} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  (TP={m['tp']} FP={m['fp']} FN={m['fn']})")

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    print(f"\n  {'MICRO-AVERAGE':<35} P={micro_p:.3f}  R={micro_r:.3f}  F1={micro_f1:.3f}")
    return micro_p, micro_r, micro_f1


def run_cuad_benchmark(sample_size: int = 10, use_llm: bool = True) -> dict:
    """Run LegalLens on CUAD contracts and compare with ground truth."""
    print("Loading CUAD annotations...")
    annotations = load_cuad_annotations()
    print(f"Loaded annotations for {len(annotations)} contracts.")

    available = []
    for contract_name, categories in annotations.items():
        txt_file = find_txt_file(contract_name)
        if txt_file and categories:
            available.append((contract_name, txt_file, categories))

    print(f"Contracts with TXT files and annotations: {len(available)}")

    if not available:
        print("No matching contracts found.")
        return {}

    sample = available[:sample_size]
    print(f"Running CUAD benchmark on {len(sample)} contracts...")
    print("=" * 60)

    pipeline = LegalLensPipeline(use_llm=use_llm)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    contract_results = []
    total_time = 0

    for idx, (name, txt_path, cuad_categories) in enumerate(sample, 1):
        print(f"\n[{idx}/{len(sample)}] {name}")

        expected = set()
        for cat in cuad_categories:
            mapped = CUAD_TO_LEGALLENS.get(cat)
            if mapped:
                expected.add(mapped)

        if not expected:
            print("  No mappable categories, skipping.")
            continue

        try:
            result = pipeline.analyze(str(txt_path))
        except Exception as e:
            print(f"  Pipeline error: {e}")
            continue

        total_time += result.processing_time
        detected = _get_detected_categories(result)

        true_pos = expected & detected
        false_pos = detected - expected
        false_neg = expected - detected

        for cat in true_pos:
            tp[cat] += 1
        for cat in false_pos:
            fp[cat] += 1
        for cat in false_neg:
            fn[cat] += 1

        print(f"  Expected:  {sorted(expected)}")
        print(f"  Detected:  {sorted(detected)}")
        print(f"  TP={len(true_pos)} FP={len(false_pos)} FN={len(false_neg)} | Time: {result.processing_time}s")

        contract_results.append({
            "contract": name,
            "expected": sorted(expected),
            "detected": sorted(detected),
            "tp": sorted(true_pos),
            "fp": sorted(false_pos),
            "fn": sorted(false_neg),
            "processing_time": result.processing_time,
        })

    print("\n" + "=" * 60)
    print("CUAD BENCHMARK RESULTS")
    print("=" * 60)

    metrics = _compute_metrics(tp, fp, fn)
    micro_p, micro_r, micro_f1 = _print_metrics(metrics, tp, fp, fn)

    print(f"  Contracts tested: {len(contract_results)}")
    print(f"  Total time: {total_time:.1f}s")

    results = {
        "dataset": "CUAD",
        "sample_size": len(contract_results),
        "use_llm": use_llm,
        "total_time": round(total_time, 2),
        "micro_precision": round(micro_p, 3),
        "micro_recall": round(micro_r, 3),
        "micro_f1": round(micro_f1, 3),
        "category_metrics": metrics,
        "contract_results": contract_results,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "benchmark_cuad.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved: {output_path}")
    return results


def run_contractnli_benchmark(sample_size: int = 10, use_llm: bool = True) -> dict:
    """Run LegalLens on ContractNLI documents and compare with ground truth."""
    print("Loading ContractNLI annotations...")
    documents = load_contractnli_annotations()
    print(f"Loaded {len(documents)} annotated documents.")

    if not documents:
        print("No annotated documents found.")
        return {}

    sample = documents[:sample_size]
    print(f"Running ContractNLI benchmark on {len(sample)} documents...")
    print("=" * 60)

    pipeline = LegalLensPipeline(use_llm=use_llm)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    doc_results = []
    total_time = 0

    for idx, doc in enumerate(sample, 1):
        print(f"\n[{idx}/{len(sample)}] {doc['name']}")

        expected = doc["expected"]

        try:
            result = pipeline.analyze_text(doc["text"], filename=doc["name"])
        except Exception as e:
            print(f"  Pipeline error: {e}")
            continue

        total_time += result.processing_time
        detected = _get_detected_categories(result)

        true_pos = expected & detected
        false_pos = detected - expected
        false_neg = expected - detected

        for cat in true_pos:
            tp[cat] += 1
        for cat in false_pos:
            fp[cat] += 1
        for cat in false_neg:
            fn[cat] += 1

        print(f"  Expected:  {sorted(expected)}")
        print(f"  Detected:  {sorted(detected)}")
        print(f"  TP={len(true_pos)} FP={len(false_pos)} FN={len(false_neg)} | Time: {result.processing_time}s")

        doc_results.append({
            "document": doc["name"],
            "expected": sorted(expected),
            "detected": sorted(detected),
            "tp": sorted(true_pos),
            "fp": sorted(false_pos),
            "fn": sorted(false_neg),
            "processing_time": result.processing_time,
        })

    print("\n" + "=" * 60)
    print("CONTRACTNLI BENCHMARK RESULTS")
    print("=" * 60)

    metrics = _compute_metrics(tp, fp, fn)
    micro_p, micro_r, micro_f1 = _print_metrics(metrics, tp, fp, fn)

    print(f"  Documents tested: {len(doc_results)}")
    print(f"  Total time: {total_time:.1f}s")

    results = {
        "dataset": "ContractNLI",
        "sample_size": len(doc_results),
        "use_llm": use_llm,
        "total_time": round(total_time, 2),
        "micro_precision": round(micro_p, 3),
        "micro_recall": round(micro_r, 3),
        "micro_f1": round(micro_f1, 3),
        "category_metrics": metrics,
        "document_results": doc_results,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "benchmark_contractnli.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved: {output_path}")
    return results


if __name__ == "__main__":
    sample = 10
    use_llm = True
    dataset = "cuad"

    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--sample" and i + 1 < len(sys.argv) - 1:
            sample = int(sys.argv[i + 2])
        if arg == "--no-llm":
            use_llm = False
        if arg == "--dataset" and i + 1 < len(sys.argv) - 1:
            dataset = sys.argv[i + 2].lower()

    if dataset == "contractnli":
        run_contractnli_benchmark(sample_size=sample, use_llm=use_llm)
    elif dataset == "both":
        run_cuad_benchmark(sample_size=sample, use_llm=use_llm)
        print("\n\n")
        run_contractnli_benchmark(sample_size=sample, use_llm=use_llm)
    else:
        run_cuad_benchmark(sample_size=sample, use_llm=use_llm)
