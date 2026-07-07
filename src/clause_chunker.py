"""
LegalLens - Clause Chunker
============================
Splits legal document text into clause-level chunks suitable for
lexicon scanning and LLM analysis.

Strategy:
    1. Split by section headers (e.g., "1.", "1.1", "Section 1", "ARTICLE I")
    2. Within sections, split by sentence boundaries (via Stanza)
    3. Group sentences into clauses (consecutive sentences in same section)
    4. Assign metadata: section number, position, character offsets
"""

import re
from dataclasses import dataclass, field


@dataclass
class Clause:
    """Single clause extracted from a legal document."""
    id: int
    text: str
    section_header: str
    start_char: int
    end_char: int
    sentence_count: int


# Patterns matching common legal section headers
SECTION_PATTERNS = [
    # "ARTICLE I", "ARTICLE II", "ARTICLE 1"
    re.compile(r"^ARTICLE\s+[IVXLC\d]+\.?\s*[:\-—]?\s*", re.IGNORECASE),
    # "Section 1.1", "SECTION 2", "Section 12.3.4"
    re.compile(r"^SECTION\s+[\d]+(?:\.[\d]+)*\.?\s*[:\-—]?\s*", re.IGNORECASE),
    # "1.", "1.1", "1.1.1", "12.3" at start of line (numbered clauses)
    re.compile(r"^(\d{1,3}(?:\.\d{1,3}){0,3})\.?\s+"),
    # "A.", "B.", "C." — single uppercase letter + period (seen in Apollo contract)
    re.compile(r"^[A-Z]\.\s+"),
    # "(a)", "(b)", "(i)", "(ii)", "(1)", "(2)"
    re.compile(r"^\([a-z]{1,3}\)\s+"),
    re.compile(r"^\([ivx]{1,4}\)\s+"),
    re.compile(r"^\(\d{1,2}\)\s+"),
    # Legal preamble markers (seen in Adams Golf, standard in common law contracts)
    re.compile(r"^(RECITALS|WITNESSETH|WHEREAS|NOW\s*,?\s*THEREFORE)\s*[,:]?\s*", re.IGNORECASE),
    # ALL CAPS headers (e.g., "CONFIDENTIALITY", "LIMITATION OF LIABILITY")
    # Min 5 chars, max 60, must NOT match known SEC boilerplate
    re.compile(r"^[A-Z][A-Z\s,&/\-]{4,60}$"),
]

# SEC/EDGAR filing boilerplate that appears at the start of contracts.
# This is metadata, not contract content — strip before chunking.
SEC_BOILERPLATE_PATTERNS = [
    re.compile(r"^Exhibit\s+\d+[\.\d]*\b", re.IGNORECASE),
    re.compile(r"^CONFIDENTIAL\s+TREATMENT\s+REQUESTED", re.IGNORECASE),
    re.compile(r"^REDACTED\s+COPY", re.IGNORECASE),
    re.compile(r"^CONFIDENTIAL\s+PORTIONS\s+OF\s+THIS", re.IGNORECASE),
    re.compile(r"^DOCUMENT\s+HAVE\s+BEEN\s+REDACTED", re.IGNORECASE),
    re.compile(r"^AND\s+HAVE\s+BEEN\s+SEPARATELY", re.IGNORECASE),
    re.compile(r"^FILED\s+WITH\s+THE\s+COMMISSION", re.IGNORECASE),
    re.compile(r"^EX-\d+", re.IGNORECASE),
    re.compile(r"^\*+\s*$"),  # Lines of only asterisks
    re.compile(r"^_{3,}\s*$"),  # Lines of only underscores
]


def chunk_document(text: str, stanza_nlp=None, max_chunk_chars: int = 2000) -> list:
    """
    Split legal document text into clause-level chunks.

    Args:
        text: Full document text (from document_extractor).
        stanza_nlp: Optional Stanza pipeline for sentence segmentation.
                    If None, falls back to regex-based splitting.
        max_chunk_chars: Maximum characters per chunk. Longer sections
                        are split at sentence boundaries.

    Returns:
        List of Clause objects.
    """
    # Step 0: Strip SEC/EDGAR boilerplate from the beginning
    text = _strip_sec_boilerplate(text)

    # Step 1: Split into raw sections by headers
    raw_sections = _split_by_headers(text)

    # Step 2: Split long sections into sentence-based chunks
    clauses = []
    clause_id = 0

    for header, section_text, start_char in raw_sections:
        if not section_text.strip():
            continue

        if len(section_text) <= max_chunk_chars:
            # Section fits in one chunk
            sent_count = _count_sentences(section_text, stanza_nlp)
            clauses.append(Clause(
                id=clause_id,
                text=section_text.strip(),
                section_header=header,
                start_char=start_char,
                end_char=start_char + len(section_text),
                sentence_count=sent_count,
            ))
            clause_id += 1
        else:
            # Section too long — split at sentence boundaries
            sub_chunks = _split_long_section(section_text, stanza_nlp, max_chunk_chars)
            for chunk_text in sub_chunks:
                if not chunk_text.strip():
                    continue
                offset = text.find(chunk_text, start_char)
                if offset == -1:
                    offset = start_char
                sent_count = _count_sentences(chunk_text, stanza_nlp)
                clauses.append(Clause(
                    id=clause_id,
                    text=chunk_text.strip(),
                    section_header=header,
                    start_char=offset,
                    end_char=offset + len(chunk_text),
                    sentence_count=sent_count,
                ))
                clause_id += 1

    return clauses


def _strip_sec_boilerplate(text: str) -> str:
    """
    Remove SEC/EDGAR filing boilerplate from the beginning of a contract.

    Many CUAD contracts start with lines like:
        Exhibit 10.19 CONFIDENTIAL TREATMENT REQUESTED Certain portions...
        REDACTED COPY
        CONFIDENTIAL PORTIONS OF THIS
        DOCUMENT HAVE BEEN REDACTED
        AND HAVE BEEN SEPARATELY
        FILED WITH THE COMMISSION

    This is filing metadata, not contract content. We strip it so
    the chunker doesn't create false sections from it.

    Only strips from the beginning — stops as soon as it encounters
    a non-boilerplate line with actual content (>30 chars).
    """
    lines = text.split("\n")
    start_idx = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Check if line matches any boilerplate pattern
        is_boilerplate = False
        for pattern in SEC_BOILERPLATE_PATTERNS:
            if pattern.match(stripped):
                is_boilerplate = True
                break

        if is_boilerplate:
            start_idx = i + 1
            continue

        # Non-boilerplate line found — but short ALL CAPS lines
        # at the top (like "1") are page number artifacts, skip them
        if len(stripped) <= 3 and stripped.isdigit():
            start_idx = i + 1
            continue

        # Real content found — stop stripping
        break

    return "\n".join(lines[start_idx:])


def _split_by_headers(text: str) -> list:
    """
    Split text into sections based on legal section headers.

    Returns:
        List of (header, section_text, start_char) tuples.
    """
    lines = text.split("\n")
    sections = []
    current_header = "Preamble"
    current_lines = []
    current_start = 0
    char_pos = 0

    for line in lines:
        stripped = line.strip()
        is_header = False

        if stripped:
            # Skip lines that look like SEC boilerplate even mid-document
            is_sec = False
            for bp in SEC_BOILERPLATE_PATTERNS:
                if bp.match(stripped):
                    is_sec = True
                    break

            if not is_sec:
                for pattern in SECTION_PATTERNS:
                    if pattern.match(stripped):
                        # Save previous section
                        if current_lines:
                            section_text = "\n".join(current_lines)
                            sections.append((current_header, section_text, current_start))

                        current_header = stripped[:80]  # Truncate long headers
                        current_lines = []
                        current_start = char_pos
                        is_header = True
                        break

        if not is_header:
            current_lines.append(line)

        char_pos += len(line) + 1  # +1 for newline

    # Don't forget the last section
    if current_lines:
        section_text = "\n".join(current_lines)
        sections.append((current_header, section_text, current_start))

    return sections


def _split_long_section(text: str, stanza_nlp, max_chars: int) -> list:
    """Split a long section into chunks at sentence boundaries."""
    sentences = _get_sentences(text, stanza_nlp)

    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0

        current_chunk.append(sent)
        current_len += sent_len + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def _get_sentences(text: str, stanza_nlp) -> list:
    """Get sentence list using Stanza or regex fallback."""
    if stanza_nlp is not None:
        doc = stanza_nlp(text)
        return [sent.text for sent in doc.sentences]

    # Regex fallback: split on period/semicolon followed by space+uppercase
    parts = re.split(r'(?<=[.;])\s+(?=[A-Z"])', text)
    return [p.strip() for p in parts if p.strip()]


def _count_sentences(text: str, stanza_nlp) -> int:
    """Count sentences in text."""
    return len(_get_sentences(text, stanza_nlp))
