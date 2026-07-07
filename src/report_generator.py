"""
LegalLens - HTML Report Generator
==================================
Takes AnalysisResult from pipeline and renders a styled HTML report
using the Jinja2 template in templates/report.html.

Usage:
    from src.report_generator import generate_report
    generate_report(result, output_dir="output")
"""

from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader


# Project root = one level up from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "templates"


def _format_category(name: str) -> str:
    """Convert snake_case category to readable form: uncapped_liability -> Uncapped Liability."""
    return name.replace("_", " ").title()


def _build_risk_summary(raw_summary: dict) -> list[dict]:
    """
    Convert pipeline's risk_summary dict into a sorted list for the template.

    Pipeline returns:
        {
            'total_matches': N,
            'overall_risk': float,
            'risk_level': str,
            'categories': {'cat_name': {'count': N, 'max_risk': M, 'terms': [...]}, ...},
            'critical_clauses': [...]
        }

    Template expects:
        [{'category': str, 'count': int, 'max_risk': int}, ...]
    """
    categories = raw_summary.get("categories", {})

    rows = []
    for category, stats in categories.items():
        rows.append({
            "category": _format_category(category),
            "count": stats.get("count", 0),
            "max_risk": stats.get("max_risk", 0),
        })

    rows.sort(key=lambda r: (r["max_risk"], r["count"]), reverse=True)
    return rows


def _build_match_data(match) -> dict:
    """
    Convert a ClauseMatch object into a flat dict for the template.
    """
    if isinstance(match, dict):
        raw = {
            "sentence_text": match.get("sentence_text", ""),
            "matched_term": match.get("matched_term", ""),
            "category": match.get("category", ""),
            "base_risk": match.get("base_risk", 0),
            "adjusted_risk": match.get("adjusted_risk", 0),
            "negated": match.get("negated", False),
            "intensified": match.get("intensified", False),
            "legal_refs": match.get("legal_refs", []),
        }
    else:
        raw = {
            "sentence_text": getattr(match, "sentence_text", ""),
            "matched_term": getattr(match, "matched_term", ""),
            "category": getattr(match, "category", ""),
            "base_risk": getattr(match, "base_risk", 0),
            "adjusted_risk": getattr(match, "adjusted_risk", 0),
            "negated": getattr(match, "negated", False),
            "intensified": getattr(match, "intensified", False),
            "legal_refs": getattr(match, "legal_refs", []),
        }

    raw["category"] = _format_category(raw["category"])
    return raw


def _build_llm_data(analysis) -> dict:
    """
    Convert an LLM analysis object into a flat dict for the template.
    """
    if isinstance(analysis, dict):
        raw = {
            "clause_text": analysis.get("clause_text", ""),
            "assessment": analysis.get("assessment", ""),
            "category": analysis.get("category", ""),
            "lexicon_risk": analysis.get("lexicon_risk", 0),
            "llm_risk": analysis.get("llm_risk", 0),
        }
    else:
        raw = {
            "clause_text": getattr(analysis, "clause_text", ""),
            "assessment": getattr(analysis, "assessment", ""),
            "category": getattr(analysis, "category", ""),
            "lexicon_risk": getattr(analysis, "lexicon_risk", 0),
            "llm_risk": getattr(analysis, "llm_risk", 0),
        }

    raw["category"] = _format_category(raw["category"])
    return raw


def generate_report(result, output_dir: str = "output") -> Path:
    """
    Generate an HTML report from an AnalysisResult.

    Args:
        result: AnalysisResult from pipeline.py (object or dict).
        output_dir: Directory to write the HTML file into.

    Returns:
        Path to the generated HTML file.
    """
    if isinstance(result, dict):
        document = result.get("document", {})
        clauses_total = result.get("clauses_total", 0)
        clauses_flagged = result.get("clauses_flagged", 0)
        lexicon_matches_raw = result.get("lexicon_matches", [])
        llm_analyses_raw = result.get("llm_analyses", [])
        risk_summary_raw = result.get("risk_summary", {})
        processing_time = result.get("processing_time", 0)
    else:
        document = getattr(result, "document", {})
        clauses_total = getattr(result, "clauses_total", 0)
        clauses_flagged = getattr(result, "clauses_flagged", 0)
        lexicon_matches_raw = getattr(result, "lexicon_matches", [])
        llm_analyses_raw = getattr(result, "llm_analyses", [])
        risk_summary_raw = getattr(result, "risk_summary", {})
        processing_time = getattr(result, "processing_time", 0)

    # Prepare data for template
    lexicon_matches = [_build_match_data(m) for m in lexicon_matches_raw]
    llm_analyses = [_build_llm_data(a) for a in llm_analyses_raw]
    risk_summary = _build_risk_summary(risk_summary_raw)

    categories_count = len(risk_summary)

    if clauses_total > 0:
        risk_percent = round((clauses_flagged / clauses_total) * 100, 1)
    else:
        risk_percent = 0

    lexicon_matches.sort(key=lambda m: m["adjusted_risk"], reverse=True)

    # Load and render Jinja2 template
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html")

    html = template.render(
        document=document,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        processing_time=processing_time,
        clauses_total=clauses_total,
        clauses_flagged=clauses_flagged,
        categories_count=categories_count,
        risk_percent=risk_percent,
        risk_summary=risk_summary,
        lexicon_matches=lexicon_matches,
        llm_analyses=llm_analyses,
    )

    # Write to output
    output_path = PROJECT_ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    filename = document.get("filename", "unknown")
    stem = Path(filename).stem
    report_file = output_path / f"{stem}_report.html"

    report_file.write_text(html, encoding="utf-8")

    return report_file
