"""
LegalLens - Full System Test
===============================
Tests every module from bottom to top. Each test is independent -
if one fails, the rest still run. At the end you get a clear
pass/fail summary showing exactly where the problem is.

Usage:
    conda activate legallens
    python test_full.py
"""

import sys
import json
import time
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS = []


def test(name: str):
    """Decorator that catches exceptions and records pass/fail."""
    def decorator(func):
        def wrapper():
            try:
                func()
                RESULTS.append(("PASS", name, ""))
                print(f"  PASS  {name}")
            except Exception as e:
                tb = traceback.format_exc()
                RESULTS.append(("FAIL", name, str(e)))
                print(f"  FAIL  {name}")
                print(f"        {e}")
                print(f"        {tb.strip().splitlines()[-2].strip()}")
        return wrapper
    return decorator


# ================================================================
# 1. ENVIRONMENT
# ================================================================

@test("1.1  import stanza")
def test_import_stanza():
    import stanza

@test("1.2  import PyMuPDF (fitz)")
def test_import_fitz():
    import fitz

@test("1.3  import python-docx")
def test_import_docx():
    import docx

@test("1.4  import requests")
def test_import_requests():
    import requests

@test("1.5  import jinja2")
def test_import_jinja2():
    import jinja2

@test("1.6  import watchdog")
def test_import_watchdog():
    import watchdog

@test("1.7  Stanza English model downloaded")
def test_stanza_model():
    # Stanza 1.9+ stores models in AppData/Local/StanfordNLP or ~/stanza_resources
    home = Path.home()
    candidates = [
        home / "stanza_resources" / "en",
        home / "AppData" / "Local" / "StanfordNLP" / "stanza",
    ]
    found = any(p.exists() for p in candidates)
    assert found, (
        "Stanza model not found. Run: python -c \"import stanza; stanza.download('en')\""
    )


# ================================================================
# 2. PROJECT STRUCTURE
# ================================================================

@test("2.1  src/ directory exists")
def test_src_dir():
    assert (PROJECT_ROOT / "src").is_dir()

@test("2.2  src/__init__.py exists")
def test_src_init():
    assert (PROJECT_ROOT / "src" / "__init__.py").exists(), "Missing src/__init__.py"

@test("2.3  templates/report.html exists")
def test_template():
    assert (PROJECT_ROOT / "templates" / "report.html").exists()

@test("2.4  data/legal_lexicon.json exists")
def test_lexicon_file():
    path = PROJECT_ROOT / "data" / "legal_lexicon.json"
    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "terms" in data, "Lexicon missing 'terms' key"
    assert len(data["terms"]) >= 90, f"Expected 90+ terms, got {len(data['terms'])}"

@test("2.5  CUAD dataset present")
def test_cuad_data():
    cuad_json = PROJECT_ROOT / "data" / "cuad" / "CUAD_v1" / "CUAD_v1.json"
    assert cuad_json.exists(), f"CUAD_v1.json not found at {cuad_json}"
    cuad_txt = PROJECT_ROOT / "data" / "cuad" / "CUAD_v1" / "full_contract_txt"
    assert cuad_txt.is_dir(), f"full_contract_txt not found at {cuad_txt}"
    txt_files = list(cuad_txt.glob("*.txt"))
    assert len(txt_files) > 0, "No TXT files in full_contract_txt/"

@test("2.6  ContractNLI dataset present")
def test_contractnli_data():
    nli_dir = PROJECT_ROOT / "data" / "contract-nli" / "contract-nli"
    assert nli_dir.is_dir(), f"ContractNLI not found at {nli_dir}"
    assert (nli_dir / "train.json").exists(), "train.json not found"

@test("2.7  output/ directory exists (or can be created)")
def test_output_dir():
    out = PROJECT_ROOT / "output"
    out.mkdir(exist_ok=True)
    assert out.is_dir()

@test("2.8  input/ directory exists")
def test_input_dir():
    assert (PROJECT_ROOT / "input").is_dir()

@test("2.9  processing/ directory exists")
def test_processing_dir():
    proc = PROJECT_ROOT / "processing"
    proc.mkdir(exist_ok=True)
    assert proc.is_dir()


# ================================================================
# 3. MODULE IMPORTS
# ================================================================

@test("3.1  import document_extractor")
def test_import_extractor():
    from src.document_extractor import extract

@test("3.2  import clause_chunker")
def test_import_chunker():
    from src.clause_chunker import chunk_document

@test("3.3  import lexicon_scanner")
def test_import_scanner():
    from src.lexicon_scanner import LexiconScanner

@test("3.4  import ollama_client")
def test_import_ollama():
    from src.ollama_client import OllamaClient

@test("3.5  import pipeline")
def test_import_pipeline():
    from src.pipeline import LegalLensPipeline

@test("3.6  import report_generator")
def test_import_report():
    from src.report_generator import generate_report


# ================================================================
# 4. DOCUMENT EXTRACTOR
# ================================================================

@test("4.1  extract TXT file")
def test_extract_txt():
    from src.document_extractor import extract

    tmp = PROJECT_ROOT / "output" / "_test_input.txt"
    tmp.write_text(
        "This is a test contract.\nSection 1. Definitions.\n"
        "The Contractor shall provide services.",
        encoding="utf-8",
    )

    result = extract(str(tmp))
    assert result.text is not None
    assert len(result.text) > 0
    assert "Contractor" in result.text
    tmp.unlink()

@test("4.2  extract from CUAD TXT sample")
def test_extract_cuad():
    from src.document_extractor import extract

    cuad_txt = PROJECT_ROOT / "data" / "cuad" / "CUAD_v1" / "full_contract_txt"
    txt_files = list(cuad_txt.glob("*.txt"))
    assert len(txt_files) > 0, "No CUAD TXT files found"

    result = extract(str(txt_files[0]))
    assert result.text is not None
    assert len(result.text) > 100, f"Extracted text too short ({len(result.text)} chars)"


# ================================================================
# 5. CLAUSE CHUNKER
# ================================================================

@test("5.1  chunk simple text (no Stanza)")
def test_chunk_simple():
    from src.clause_chunker import chunk_document

    text = (
        "SECTION 1. DEFINITIONS\n"
        "The following terms shall have the meanings set forth herein.\n\n"
        "SECTION 2. TERM AND TERMINATION\n"
        "This Agreement shall commence on the Effective Date."
    )

    clauses = chunk_document(text, stanza_nlp=None)
    assert len(clauses) >= 1, f"Expected at least 1 clause, got {len(clauses)}"
    assert all(hasattr(c, "text") for c in clauses)

@test("5.2  chunk CUAD contract sample")
def test_chunk_cuad():
    from src.document_extractor import extract
    from src.clause_chunker import chunk_document

    cuad_txt = PROJECT_ROOT / "data" / "cuad" / "CUAD_v1" / "full_contract_txt"
    txt_files = list(cuad_txt.glob("*.txt"))
    doc = extract(str(txt_files[0]))
    clauses = chunk_document(doc.text, stanza_nlp=None)
    assert len(clauses) >= 2, f"Expected multiple clauses, got {len(clauses)}"


# ================================================================
# 6. LEXICON SCANNER
# ================================================================

@test("6.1  LexiconScanner loads lexicon")
def test_scanner_load():
    from src.lexicon_scanner import LexiconScanner
    scanner = LexiconScanner()
    assert scanner is not None

@test("6.2  scanner detects known term")
def test_scanner_detect():
    import stanza
    from src.lexicon_scanner import LexiconScanner

    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse", verbose=False)
    scanner = LexiconScanner()

    doc = nlp("The contractor shall be liable for all consequential damages arising from breach of this agreement.")
    matches = scanner.scan(doc)
    assert len(matches) > 0, "Scanner found no matches in sentence with legal terms"

@test("6.3  scanner detects negation")
def test_scanner_negation():
    import stanza
    from src.lexicon_scanner import LexiconScanner

    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse", verbose=False)
    scanner = LexiconScanner()

    doc = nlp("The contractor shall not be liable for any indirect damages.")
    matches = scanner.scan(doc)

    negated_matches = [m for m in matches if getattr(m, "negated", False)]
    if matches:
        assert len(negated_matches) > 0, "Scanner did not detect negation in 'shall not be liable'"


# ================================================================
# 7. OLLAMA
# ================================================================

@test("7.1  Ollama API reachable")
def test_ollama_reachable():
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        assert r.status_code == 200, f"Ollama responded with status {r.status_code}"
    except requests.ConnectionError:
        raise AssertionError("Ollama not running. Start it with: ollama serve")

@test("7.2  Ollama model available (qwen3.5:9b)")
def test_ollama_model():
    import requests
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    qwen_found = any("qwen3.5" in m for m in models)
    assert qwen_found, f"qwen3.5:9b not found. Available: {models}"

@test("7.3  OllamaClient sends and receives")
def test_ollama_client():
    from src.ollama_client import OllamaClient
    client = OllamaClient()
    result = client.analyze_clause(
        clause_text="The Contractor shall indemnify the Company against all claims.",
        lexicon_category="indemnification",
        lexicon_risk=4,
        legal_refs=["Art. 471 Civil Code"],
    )
    assert result is not None, "OllamaClient returned None"


# ================================================================
# 8. PIPELINE
# ================================================================

@test("8.1  Pipeline initializes (lexicon only)")
def test_pipeline_init_no_llm():
    from src.pipeline import LegalLensPipeline
    p = LegalLensPipeline(use_llm=False)
    assert p is not None

@test("8.2  Pipeline analyzes test text (lexicon only)")
def test_pipeline_analyze_text():
    from src.pipeline import LegalLensPipeline
    p = LegalLensPipeline(use_llm=False)

    result = p.analyze_text(
        text=(
            "SECTION 1. INDEMNIFICATION\n"
            "The Contractor shall indemnify and hold harmless the Company from any claims, "
            "damages, or liabilities arising out of the Contractor's negligence.\n\n"
            "SECTION 2. TERMINATION\n"
            "Either party may terminate this Agreement upon thirty days written notice. "
            "Upon termination, all confidential information shall be returned or destroyed.\n\n"
            "SECTION 3. LIMITATION OF LIABILITY\n"
            "In no event shall either party be liable for any indirect, incidental, "
            "or consequential damages, including lost profits."
        ),
        filename="test_contract.txt",
    )

    assert result is not None, "Pipeline returned None"
    assert result.clauses_total > 0, "No clauses detected"
    assert len(result.lexicon_matches) > 0, "No lexicon matches in text with legal terms"
    assert result.processing_time > 0, "Processing time is 0"

@test("8.3  Pipeline analyzes CUAD contract (lexicon only)")
def test_pipeline_cuad():
    from src.pipeline import LegalLensPipeline

    cuad_txt = PROJECT_ROOT / "data" / "cuad" / "CUAD_v1" / "full_contract_txt"
    txt_files = list(cuad_txt.glob("*.txt"))

    p = LegalLensPipeline(use_llm=False)
    result = p.analyze(str(txt_files[0]))

    assert result is not None
    assert result.clauses_total > 0, "No clauses in real CUAD contract"
    assert len(result.lexicon_matches) > 0, "No lexicon matches in real CUAD contract"

@test("8.4  Pipeline with LLM (full stack)")
def test_pipeline_llm():
    from src.pipeline import LegalLensPipeline
    p = LegalLensPipeline(use_llm=True)

    result = p.analyze_text(
        text=(
            "SECTION 1. UNLIMITED LIABILITY\n"
            "The Contractor shall assume unlimited liability for all direct and "
            "consequential damages arising from any breach of this Agreement, "
            "including but not limited to lost profits and business interruption."
        ),
        filename="test_llm.txt",
    )

    assert result is not None
    assert len(result.lexicon_matches) > 0, "No lexicon matches"
    assert len(result.llm_analyses) > 0, "LLM produced no analyses for high-risk text"

@test("8.5  AnalysisResult.to_json() works")
def test_result_json():
    from src.pipeline import LegalLensPipeline
    p = LegalLensPipeline(use_llm=False)

    result = p.analyze_text(
        text="The Contractor shall indemnify the Company against all claims.",
        filename="test_json.txt",
    )

    json_str = result.to_json()
    assert json_str is not None
    parsed = json.loads(json_str)
    assert "document" in parsed
    assert "lexicon_matches" in parsed


# ================================================================
# 9. REPORT GENERATOR
# ================================================================

@test("9.1  Jinja2 template loads")
def test_template_loads():
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(str(PROJECT_ROOT / "templates")))
    t = env.get_template("report.html")
    assert t is not None

@test("9.2  generate_report produces HTML file")
def test_generate_report():
    from src.pipeline import LegalLensPipeline
    from src.report_generator import generate_report

    p = LegalLensPipeline(use_llm=False)
    result = p.analyze_text(
        text=(
            "SECTION 1. NON-COMPETE\n"
            "The Employee shall not engage in any competing business within a radius "
            "of fifty miles for a period of two years following termination.\n\n"
            "SECTION 2. CONFIDENTIALITY\n"
            "All proprietary information shall remain strictly confidential and shall "
            "not be disclosed to any third party without prior written consent."
        ),
        filename="test_report.txt",
    )

    report_path = generate_report(result)
    assert report_path.exists(), f"HTML report not created at {report_path}"
    assert report_path.stat().st_size > 1000, "HTML report suspiciously small"

    html = report_path.read_text(encoding="utf-8")
    assert "LegalLens" in html, "Report missing LegalLens branding"
    assert "test_report" in html, "Report missing document filename"

    report_path.unlink()


# ================================================================
# 10. WATCHER
# ================================================================

@test("10.1  watcher.py imports successfully")
def test_watcher_imports():
    import importlib.util
    spec = importlib.util.spec_from_file_location("watcher", PROJECT_ROOT / "watcher.py")
    assert spec is not None


# ================================================================
# RUN ALL
# ================================================================

def run_all():
    print("=" * 60)
    print("LegalLens - Full System Test")
    print("=" * 60)

    all_tests = [
        ("ENVIRONMENT", [
            test_import_stanza, test_import_fitz, test_import_docx,
            test_import_requests, test_import_jinja2, test_import_watchdog,
            test_stanza_model,
        ]),
        ("PROJECT STRUCTURE", [
            test_src_dir, test_src_init, test_template, test_lexicon_file,
            test_cuad_data, test_contractnli_data, test_output_dir,
            test_input_dir, test_processing_dir,
        ]),
        ("MODULE IMPORTS", [
            test_import_extractor, test_import_chunker, test_import_scanner,
            test_import_ollama, test_import_pipeline, test_import_report,
        ]),
        ("DOCUMENT EXTRACTOR", [
            test_extract_txt, test_extract_cuad,
        ]),
        ("CLAUSE CHUNKER", [
            test_chunk_simple, test_chunk_cuad,
        ]),
        ("LEXICON SCANNER", [
            test_scanner_load, test_scanner_detect, test_scanner_negation,
        ]),
        ("OLLAMA", [
            test_ollama_reachable, test_ollama_model, test_ollama_client,
        ]),
        ("PIPELINE", [
            test_pipeline_init_no_llm, test_pipeline_analyze_text,
            test_pipeline_cuad, test_pipeline_llm, test_result_json,
        ]),
        ("REPORT GENERATOR", [
            test_template_loads, test_generate_report,
        ]),
        ("WATCHER", [
            test_watcher_imports,
        ]),
    ]

    for section_name, tests in all_tests:
        print(f"\n--- {section_name} ---")
        for t in tests:
            t()

    passed = sum(1 for r in RESULTS if r[0] == "PASS")
    failed = sum(1 for r in RESULTS if r[0] == "FAIL")
    total = len(RESULTS)

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\nFAILED TESTS:")
        for status, name, error in RESULTS:
            if status == "FAIL":
                print(f"  {name}")
                print(f"    -> {error}")

    print()
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
