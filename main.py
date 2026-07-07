"""
LegalLens - Main Entry Point
==============================
Usage:
    python main.py <path_to_contract>
    python main.py input/contract.pdf
    python main.py --no-llm input/contract.txt
    python main.py --host http://localhost:11435 input/contract.pdf
    python main.py --model qwen3:8b input/contract.pdf
    python main.py --help

Output:
    - Console summary
    - JSON report in output/<filename>_report.json
"""

import sys
import json
from pathlib import Path

from src.pipeline import LegalLensPipeline


def parse_args(args: list) -> dict:
    """Parse command line arguments."""
    config = {
        "use_llm": True,
        "host": "http://localhost:11434",
        "model": "qwen3.5:9b",
        "filepath": None,
    }

    i = 0
    while i < len(args):
        arg = args[i]

        if arg == "--help":
            return {"help": True}
        elif arg == "--no-llm":
            config["use_llm"] = False
        elif arg == "--host" and i + 1 < len(args):
            config["host"] = args[i + 1]
            i += 1
        elif arg == "--model" and i + 1 < len(args):
            config["model"] = args[i + 1]
            i += 1
        elif not arg.startswith("--"):
            config["filepath"] = arg

        i += 1

    return config


def print_help():
    """Print usage instructions."""
    print("LegalLens - Legal Document Risk Analyzer")
    print()
    print("Usage: python main.py [options] <path_to_contract>")
    print()
    print("Options:")
    print("  --no-llm              Run lexicon-only analysis (no Ollama required)")
    print("  --host <url>          Ollama host (default: http://localhost:11434)")
    print("  --model <name>        Ollama model (default: qwen3.5:9b)")
    print("  --help                Show this help message")
    print()
    print("Examples:")
    print("  python main.py input/contract.pdf")
    print("  python main.py --no-llm input/contract.txt")
    print("  python main.py --host http://localhost:11435 input/contract.pdf")
    print("  python main.py --model llama3.1:8b input/contract.pdf")


def main():
    args = sys.argv[1:]

    if not args:
        print_help()
        return

    config = parse_args(args)

    if config.get("help"):
        print_help()
        return

    filepath = config.get("filepath")

    if not filepath:
        print("Error: No file path provided.")
        print("Run 'python main.py --help' for usage.")
        return

    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        return

    try:
        pipeline = LegalLensPipeline(
            use_llm=config["use_llm"],
            ollama_model=config["model"],
            ollama_host=config["host"],
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return

    print(f"\nAnalyzing: {filepath}")
    print("-" * 50)

    try:
        result = pipeline.analyze(filepath)
    except Exception as e:
        print(f"Error during analysis: {e}")
        return

    # Console output
    print(f"\nDocument: {result.document['filename']}")
    print(f"Pages: {result.document['pages']} | Words: {result.document['words']}")
    print(f"Clauses analyzed: {result.clauses_total}")
    print(f"Clauses flagged: {result.clauses_flagged}")
    print(f"Lexicon matches: {len(result.lexicon_matches)}")
    print(f"LLM analyses: {len(result.llm_analyses)}")
    print(f"Processing time: {result.processing_time}s")

    print(f"\nOverall Risk: {result.risk_summary.get('risk_level', 'N/A')}")
    print(f"Risk Score: {result.risk_summary.get('overall_risk', 0)}/5")

    # Critical clauses
    critical = result.risk_summary.get("critical_clauses", [])
    if critical:
        print(f"\nCritical clauses ({len(critical)}):")
        for c in critical:
            print(f"  [{c['category']}] Risk {c['risk']}/5")
            sentence = c.get('sentence', '')
            if len(sentence) > 120:
                sentence = sentence[:120] + "..."
            print(f"    {sentence}")
            if c.get("legal_refs"):
                print(f"    Refs: {', '.join(c['legal_refs'])}")

    # Save JSON report
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    stem = Path(filepath).stem
    report_path = output_dir / f"{stem}_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(result.to_json())

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
