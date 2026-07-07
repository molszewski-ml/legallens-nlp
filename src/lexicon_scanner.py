"""
LegalLens - Lexicon-Based Legal Risk Scanner
=============================================
Self-developed lexicon-based assessment module that operates on
Stanza NLP pipeline output to identify, categorize, and score
legal risk in contract clauses.

Architecture:
    Raw text -> Stanza (tokenize, POS, lemma, depparse, NER)
              -> LexiconScanner (match terms, detect negation, score risk)
              -> List[ClauseMatch] with evidence chains
"""

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ClauseMatch:
    """Single matched legal term with full evidence chain."""
    term: str
    matched_text: str
    category: str
    base_risk: int
    adjusted_risk: int
    negated: bool
    intensified: bool
    intensifier: str
    legal_refs: list
    description: str
    sentence_text: str
    char_offset: int


class LexiconScanner:
    """
    Rule-based legal lexicon matcher built on Stanza NLP output.

    This is the 'lexicon-based assessment method' component of LegalLens.
    It replaces spaCy's PhraseMatcher with a custom implementation that:
    - Matches multi-word legal terms against lemmatized tokens from Stanza
    - Detects negation using dependency parse context
    - Detects intensifiers that escalate risk scores
    - Produces evidence chains (matched text + category + legal reference)
    """

    def __init__(self, lexicon_path: str = None):
        """
        Initialize scanner with lexicon file.

        Args:
            lexicon_path: Path to legal_lexicon.json. If None, uses default location.
        """
        if lexicon_path is None:
            lexicon_path = Path(__file__).parent.parent / "data" / "legal_lexicon.json"

        with open(lexicon_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.terms = data["terms"]
        self.negation_cues = set(data.get("negation_cues", []))
        self.intensifiers = set(data.get("intensifiers", []))
        self.metadata = data.get("metadata", {})

        # Build lookup structures for matching
        self._single_lemmas = {}   # single-word terms: lemma -> entry
        self._multi_lemmas = []    # multi-word terms: (lemma_list, entry)

        for entry in self.terms:
            lemma = entry["lemma"].lower()
            words = lemma.split()
            if len(words) == 1:
                self._single_lemmas[lemma] = entry
            else:
                self._multi_lemmas.append((words, entry))

        # Sort multi-word by length descending (longest match first)
        self._multi_lemmas.sort(key=lambda x: len(x[0]), reverse=True)

    def scan(self, stanza_doc) -> list:
        """
        Scan a Stanza-processed document for legal terms.

        Args:
            stanza_doc: Output of stanza.Pipeline(text)

        Returns:
            List of ClauseMatch objects with evidence chains.
        """
        matches = []

        for sentence in stanza_doc.sentences:
            sentence_text = sentence.text
            lemmas = [w.lemma.lower() for w in sentence.words]
            texts = [w.text for w in sentence.words]

            matched_indices = set()

            # --- Multi-word matching (longest first) ---
            for pattern_lemmas, entry in self._multi_lemmas:
                plen = len(pattern_lemmas)
                for i in range(len(lemmas) - plen + 1):
                    if i in matched_indices:
                        continue
                    window = lemmas[i:i + plen]
                    if window == pattern_lemmas:
                        span_indices = list(range(i, i + plen))
                        matched_text = " ".join(texts[i:i + plen])

                        negated, intensified, intensifier = self._check_context(
                            sentence, span_indices
                        )

                        adjusted = self._adjust_risk(
                            entry["base_risk"], negated, intensified
                        )

                        match = ClauseMatch(
                            term=entry["term"],
                            matched_text=matched_text,
                            category=entry["category"],
                            base_risk=entry["base_risk"],
                            adjusted_risk=adjusted,
                            negated=negated,
                            intensified=intensified,
                            intensifier=intensifier,
                            legal_refs=entry.get("legal_refs", []),
                            description=entry.get("description", ""),
                            sentence_text=sentence_text,
                            char_offset=sentence.words[i].start_char or 0,
                        )
                        matches.append(match)

                        for idx in span_indices:
                            matched_indices.add(idx)

            # --- Single-word matching ---
            for i, lemma in enumerate(lemmas):
                if i in matched_indices:
                    continue
                if lemma in self._single_lemmas:
                    entry = self._single_lemmas[lemma]
                    matched_text = texts[i]

                    negated, intensified, intensifier = self._check_context(
                        sentence, [i]
                    )

                    adjusted = self._adjust_risk(
                        entry["base_risk"], negated, intensified
                    )

                    match = ClauseMatch(
                        term=entry["term"],
                        matched_text=matched_text,
                        category=entry["category"],
                        base_risk=entry["base_risk"],
                        adjusted_risk=adjusted,
                        negated=negated,
                        intensified=intensified,
                        intensifier=intensifier,
                        legal_refs=entry.get("legal_refs", []),
                        description=entry.get("description", ""),
                        sentence_text=sentence_text,
                        char_offset=sentence.words[i].start_char or 0,
                    )
                    matches.append(match)
                    matched_indices.add(i)

        return matches

    def _check_context(self, sentence, span_indices: list) -> tuple:
        """
        Analyze dependency parse context around matched term.

        Uses Stanza's dependency parse to detect:
        - Negation: 'not', 'no', 'without' modifying the matched term
        - Intensifiers: 'unlimited', 'sole', 'exclusive' modifying the term

        Args:
            sentence: Stanza sentence object
            span_indices: List of word indices that make up the matched term

        Returns:
            (negated: bool, intensified: bool, intensifier_text: str)
        """
        negated = False
        intensified = False
        intensifier_text = ""

        words = sentence.words
        span_set = set(span_indices)

        for word in words:
            wid = word.id - 1  # Stanza uses 1-based indexing
            head_id = word.head - 1 if word.head > 0 else -1
            lemma_lower = word.lemma.lower()
            text_lower = word.text.lower()

            # Check if this word modifies any word in our matched span
            modifies_span = head_id in span_set

            # Also check siblings (words sharing the same head as span words)
            shares_head = False
            for si in span_indices:
                if word.head == words[si].head and wid != si:
                    shares_head = True
                    break

            is_nearby = modifies_span or shares_head or wid in span_set

            if not is_nearby:
                # Fallback: check within a 4-token window
                min_idx = min(span_indices)
                max_idx = max(span_indices)
                if not (min_idx - 4 <= wid <= max_idx + 4):
                    continue

            # Negation detection
            if lemma_lower in self.negation_cues or text_lower in self.negation_cues:
                if word.deprel in ("advmod", "det", "neg", "cc", "mark", "case"):
                    negated = True
                elif modifies_span:
                    negated = True

            # Intensifier detection
            if lemma_lower in self.intensifiers or text_lower in self.intensifiers:
                intensified = True
                intensifier_text = word.text

        return negated, intensified, intensifier_text

    def _adjust_risk(self, base: int, negated: bool, intensified: bool) -> int:
        """
        Adjust risk score based on context.

        Negation reduces risk by 2 (floor 1).
        Intensification increases risk by 1 (ceiling 5).
        Both can apply simultaneously.

        Args:
            base: Original risk score 1-5
            negated: Whether term was negated
            intensified: Whether term was intensified

        Returns:
            Adjusted risk score 1-5
        """
        adjusted = base
        if negated:
            adjusted = max(1, adjusted - 2)
        if intensified:
            adjusted = min(5, adjusted + 1)
        return adjusted

    def get_summary(self, matches: list) -> dict:
        """
        Aggregate scan results into a risk summary.

        Args:
            matches: List of ClauseMatch objects from scan()

        Returns:
            Dictionary with category breakdown, overall score, and statistics.
        """
        if not matches:
            return {
                "total_matches": 0,
                "overall_risk": 0.0,
                "risk_level": "LOW",
                "categories": {},
                "critical_clauses": [],
            }

        # Group by category
        categories = {}
        for m in matches:
            cat = m.category
            if cat not in categories:
                categories[cat] = {"count": 0, "max_risk": 0, "terms": []}
            categories[cat]["count"] += 1
            categories[cat]["max_risk"] = max(categories[cat]["max_risk"], m.adjusted_risk)
            categories[cat]["terms"].append(m.term)

        # Overall risk = weighted average (higher risks weigh more)
        total_weighted = sum(m.adjusted_risk ** 1.5 for m in matches)
        max_possible = len(matches) * (5 ** 1.5)
        overall_risk = round((total_weighted / max_possible) * 5, 2) if max_possible > 0 else 0

        # Risk level classification
        if overall_risk >= 4.0:
            risk_level = "CRITICAL"
        elif overall_risk >= 3.0:
            risk_level = "HIGH"
        elif overall_risk >= 2.0:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        # Critical clauses (adjusted_risk >= 4)
        critical = [
            {
                "term": m.term,
                "category": m.category,
                "risk": m.adjusted_risk,
                "sentence": m.sentence_text,
                "legal_refs": m.legal_refs,
            }
            for m in matches if m.adjusted_risk >= 4
        ]

        return {
            "total_matches": len(matches),
            "overall_risk": overall_risk,
            "risk_level": risk_level,
            "categories": categories,
            "critical_clauses": critical,
        }
