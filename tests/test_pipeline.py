"""
Integration Test - Full LegalLens Pipeline
============================================
Run from project root:
    conda activate legallens
    python test_pipeline.py

Tests the complete flow:
    Text -> Chunking -> Stanza NLP -> Lexicon Scan -> LLM Analysis -> Report
"""

from pathlib import Path
from src.pipeline import LegalLensPipeline


# Sample contract text with various clause types
SAMPLE_CONTRACT = """
MASTER SERVICES AGREEMENT

This Agreement is entered into as of January 1, 2025, by and between 
Acme Corporation ("Company") and Tech Solutions Ltd ("Contractor").

1. SERVICES
The Contractor shall provide software development services as described 
in Exhibit A. All intellectual property created shall be assigned to 
the Company as work for hire.

2. CONFIDENTIALITY
The Contractor agrees to maintain strict confidentiality of all 
trade secrets and confidential information disclosed by the Company. 
This non-disclosure obligation shall survive termination of this Agreement.

3. INDEMNIFICATION
The Contractor shall indemnify, defend, and hold harmless the Company 
from and against any and all claims, damages, losses, costs, and expenses 
arising out of the Contractor's performance under this Agreement.

4. LIMITATION OF LIABILITY
In no event shall either party be liable for any consequential, incidental, 
or special damages. The aggregate liability of either party shall not exceed 
the total fees paid under this Agreement in the twelve months preceding the claim.

5. TERMINATION
Either party may terminate this Agreement for convenience upon thirty (30) 
days written notice. The Company may terminate for cause immediately upon 
written notice if Contractor breaches any material term.

6. NON-COMPETE
During the term and for two years thereafter, Contractor shall not engage 
in any competitive activity or solicit employees of the Company.

7. GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware. 
Any disputes shall be resolved through binding arbitration.

8. INSURANCE
Contractor shall maintain professional liability insurance with coverage 
of not less than $1,000,000 per occurrence.
"""


def test_lexicon_only():
    """Test pipeline without LLM (lexicon-only mode)."""
    print("=" * 60)
    print("TEST 1: Lexicon-only mode (no LLM)")
    print("=" * 60)

    pipeline = LegalLensPipeline(use_llm=False)
    result = pipeline.analyze_text(SAMPLE_CONTRACT, filename="test_contract.txt")

    print(f"\nDocument: {result.document['filename']}")
    print(f"Words: {result.document['words']}")
    print(f"Clauses: {result.clauses_total}")
    print(f"Flagged: {result.clauses_flagged}")
    print(f"Matches: {len(result.lexicon_matches)}")
    print(f"Time: {result.processing_time}s")

    print(f"\nRisk Level: {result.risk_summary['risk_level']}")
    print(f"Risk Score: {result.risk_summary['overall_risk']}/5")

    print("\nCategories detected:")
    for cat, data in result.risk_summary.get('categories', {}).items():
        print(f"  - {cat}: {data['count']} matches")

    print("\nCritical clauses:")
    for c in result.risk_summary.get('critical_clauses', []):
        print(f"  [{c['category']}] {c['term']} (risk {c['risk']})")

    # Assertions
    assert result.clauses_total > 0, "Should find clauses"
    assert len(result.lexicon_matches) > 0, "Should find lexicon matches"
    assert result.risk_summary['overall_risk'] > 0, "Should calculate risk"
    assert len(result.llm_analyses) == 0, "Should have no LLM analyses in lexicon-only mode"

    print("\nTEST 1 PASSED")
    return result


def test_with_llm(host: str = "http://localhost:11434"):
    """Test pipeline with LLM analysis."""
    print("\n" + "=" * 60)
    print("TEST 2: Full pipeline with LLM")
    print("=" * 60)

    try:
        pipeline = LegalLensPipeline(use_llm=True, ollama_host=host)
    except Exception as e:
        print(f"Could not initialize LLM: {e}")
        print("Skipping LLM test.")
        return None

    result = pipeline.analyze_text(SAMPLE_CONTRACT, filename="test_contract.txt")

    print(f"\nMatches: {len(result.lexicon_matches)}")
    print(f"LLM Analyses: {len(result.llm_analyses)}")
    print(f"Time: {result.processing_time}s")

    if result.llm_analyses:
        print("\nSample LLM analysis:")
        sample = result.llm_analyses[0]
        print(f"  Risk Level: {sample.get('risk_level')}/5")
        print(f"  Label: {sample.get('risk_label')}")
        print(f"  Key Concern: {sample.get('key_concern')}")
        print(f"  Source: {sample.get('source')}")

    # Assertions
    assert len(result.llm_analyses) > 0, "Should have LLM analyses"

    print("\nTEST 2 PASSED")
    return result


def test_category_coverage():
    """Test that lexicon covers expected CUAD categories."""
    print("\n" + "=" * 60)
    print("TEST 3: Category coverage check")
    print("=" * 60)

    pipeline = LegalLensPipeline(use_llm=False)
    result = pipeline.analyze_text(SAMPLE_CONTRACT, filename="test_contract.txt")

    detected = set(result.risk_summary.get('categories', {}).keys())

    expected = {
        'ip_ownership_assignment',  # "work for hire", "intellectual property"
        'confidentiality',          # "confidential", "trade secret"
        'indemnification',          # "indemnify", "hold harmless"
        'uncapped_liability',       # "consequential damages", "liable"
        'cap_on_liability',         # "limitation of liability"
        'termination_for_convenience',  # "terminate for convenience"
        'non_compete',              # "non-compete", "competitive activity"
        'governing_law',            # "governing law", "governed by"
        'insurance',                # "insurance"
    }

    found = detected & expected
    missing = expected - detected

    print(f"Expected categories: {len(expected)}")
    print(f"Found: {len(found)}")
    print(f"Missing: {len(missing)}")

    if found:
        print("\nFound:")
        for cat in sorted(found):
            print(f"  + {cat}")

    if missing:
        print("\nMissing:")
        for cat in sorted(missing):
            print(f"  - {cat}")

    coverage = len(found) / len(expected) * 100
    print(f"\nCoverage: {coverage:.0f}%")

    assert coverage >= 50, f"Coverage too low: {coverage}%"

    print("\nTEST 3 PASSED")


def test_json_output():
    """Test JSON serialization."""
    print("\n" + "=" * 60)
    print("TEST 4: JSON output")
    print("=" * 60)

    pipeline = LegalLensPipeline(use_llm=False)
    result = pipeline.analyze_text(SAMPLE_CONTRACT, filename="test_contract.txt")

    json_str = result.to_json()
    parsed = __import__('json').loads(json_str)

    assert 'document' in parsed
    assert 'lexicon_matches' in parsed
    assert 'risk_summary' in parsed

    print(f"JSON size: {len(json_str)} bytes")
    print(f"Keys: {list(parsed.keys())}")

    # Save test output
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    test_report = output_dir / "test_pipeline_report.json"

    with open(test_report, "w", encoding="utf-8") as f:
        f.write(json_str)

    print(f"Test report saved: {test_report}")
    print("\nTEST 4 PASSED")


def main():
    print("\nLegalLens Integration Tests")
    print("=" * 60)

    try:
        test_lexicon_only()
        test_category_coverage()
        test_json_output()

        # LLM test - optional, may fail if Ollama not running
        print("\n" + "-" * 60)
        print("Attempting LLM test (requires Ollama)...")
        test_with_llm()

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
