"""
LegalLens - Analysis Pipeline
===============================
Orchestrates the full document analysis flow:

    Document (PDF/DOCX/TXT)
        -> Text Extraction (document_extractor)
        -> Clause Chunking (clause_chunker)
        -> Stanza NLP Processing (tokenize, POS, lemma, NER, depparse)
        -> Lexicon Scanning (lexicon_scanner)
        -> LLM Risk Analysis (ollama_client)
        -> Structured Report

Usage:
    from src.pipeline import LegalLensPipeline
    pipeline = LegalLensPipeline()
    result = pipeline.analyze("path/to/contract.pdf")
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict

import stanza

from src.document_extractor import extract, ExtractedDocument
from src.clause_chunker import chunk_document, Clause
from src.lexicon_scanner import LexiconScanner, ClauseMatch
from src.ollama_client import OllamaClient


@dataclass
class AnalysisResult:
    """Complete analysis result for a single document."""
    document: dict
    clauses_total: int
    clauses_flagged: int
    lexicon_matches: list
    llm_analyses: list
    risk_summary: dict
    processing_time: float

    def to_dict(self) -> dict:
        return {
            "document": self.document,
            "clauses_total": self.clauses_total,
            "clauses_flagged": self.clauses_flagged,
            "lexicon_matches": [
                {
                    "term": m.term,
                    "matched_text": m.matched_text,
                    "category": m.category,
                    "base_risk": m.base_risk,
                    "adjusted_risk": m.adjusted_risk,
                    "negated": m.negated,
                    "intensified": m.intensified,
                    "intensifier": m.intensifier,
                    "legal_refs": m.legal_refs,
                    "description": m.description,
                    "sentence": m.sentence_text,
                }
                for m in self.lexicon_matches
            ],
            "llm_analyses": self.llm_analyses,
            "risk_summary": self.risk_summary,
            "processing_time": self.processing_time,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class LegalLensPipeline:
    """
    Main analysis pipeline.

    Combines lexicon-based assessment (Stanza + custom legal lexicon)
    with LLM-based contextual analysis (Qwen 3.5 via Ollama).
    """

    def __init__(
        self,
        lexicon_path: str = None,
        ollama_model: str = "qwen3.5:9b",
        ollama_host: str = "http://localhost:11434",
        use_llm: bool = True,
    ):
        # Initialize Stanza pipeline
        print("[Pipeline] Loading Stanza NLP model...")
        self.nlp = stanza.Pipeline(
            "en",
            processors="tokenize,pos,lemma,ner,depparse",
            verbose=False,
        )

        # Initialize lexicon scanner
        print("[Pipeline] Loading legal lexicon...")
        self.scanner = LexiconScanner(lexicon_path)

        # Initialize LLM client
        self.use_llm = use_llm
        if use_llm:
            print("[Pipeline] Connecting to Ollama...")
            self.llm = OllamaClient(model=ollama_model, host=ollama_host)
        else:
            self.llm = None

        print("[Pipeline] Ready.")

    def analyze(self, filepath: str) -> AnalysisResult:
        """
        Run full analysis on a document.

        Args:
            filepath: Path to PDF, DOCX, or TXT file.

        Returns:
            AnalysisResult with lexicon matches, LLM analyses, and risk summary.
        """
        start_time = time.time()

        # Step 1: Extract text
        doc = extract(filepath)

        # Step 2: Chunk into clauses
        clauses = chunk_document(doc.text, stanza_nlp=self.nlp)

        # Step 3: Run Stanza NLP + lexicon scan on each clause
        all_matches = []
        for clause in clauses:
            stanza_doc = self.nlp(clause.text)
            matches = self.scanner.scan(stanza_doc)
            all_matches.extend(matches)

        # Step 4: LLM analysis on flagged clauses (risk >= 3)
        llm_analyses = []
        if self.use_llm and self.llm is not None:
            high_risk = [m for m in all_matches if m.adjusted_risk >= 3]

            # Deduplicate by sentence text
            seen_sentences = set()
            unique_high_risk = []
            for m in high_risk:
                if m.sentence_text not in seen_sentences:
                    seen_sentences.add(m.sentence_text)
                    unique_high_risk.append(m)

            for match in unique_high_risk:
                analysis = self.llm.analyze_clause(
                    clause_text=match.sentence_text,
                    lexicon_category=match.category,
                    lexicon_risk=match.adjusted_risk,
                    legal_refs=match.legal_refs,
                )
                llm_analyses.append(analysis)

        # Step 5: Generate risk summary
        risk_summary = self.scanner.get_summary(all_matches)

        elapsed = round(time.time() - start_time, 2)

        return AnalysisResult(
            document={
                "filename": doc.filename,
                "format": doc.format,
                "pages": doc.page_count,
                "words": doc.word_count,
            },
            clauses_total=len(clauses),
            clauses_flagged=len(set(m.sentence_text for m in all_matches)),
            lexicon_matches=all_matches,
            llm_analyses=llm_analyses,
            risk_summary=risk_summary,
            processing_time=elapsed,
        )

    def analyze_text(self, text: str, filename: str = "raw_text") -> AnalysisResult:
        """
        Run analysis directly on text string (for CUAD/ContractNLI testing).

        Args:
            text: Contract text as string.
            filename: Label for the document.

        Returns:
            AnalysisResult.
        """
        start_time = time.time()

        clauses = chunk_document(text, stanza_nlp=self.nlp)

        all_matches = []
        for clause in clauses:
            stanza_doc = self.nlp(clause.text)
            matches = self.scanner.scan(stanza_doc)
            all_matches.extend(matches)

        llm_analyses = []
        if self.use_llm and self.llm is not None:
            high_risk = [m for m in all_matches if m.adjusted_risk >= 3]
            seen = set()
            for m in high_risk:
                if m.sentence_text not in seen:
                    seen.add(m.sentence_text)
                    analysis = self.llm.analyze_clause(
                        clause_text=m.sentence_text,
                        lexicon_category=m.category,
                        lexicon_risk=m.adjusted_risk,
                        legal_refs=m.legal_refs,
                    )
                    llm_analyses.append(analysis)

        risk_summary = self.scanner.get_summary(all_matches)
        elapsed = round(time.time() - start_time, 2)

        return AnalysisResult(
            document={
                "filename": filename,
                "format": "text",
                "pages": 1,
                "words": len(text.split()),
            },
            clauses_total=len(clauses),
            clauses_flagged=len(set(m.sentence_text for m in all_matches)),
            lexicon_matches=all_matches,
            llm_analyses=llm_analyses,
            risk_summary=risk_summary,
            processing_time=elapsed,
        )
