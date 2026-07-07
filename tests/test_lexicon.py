"""
Test script for LexiconScanner
==============================
Run from project root:
    conda activate legallens
    python test_lexicon.py
"""

import stanza
from src.lexicon_scanner import LexiconScanner

# Sample legal text with various risk terms
TEST_TEXT = """
The Contractor shall indemnify and hold harmless the Company from any claims.
Liability shall not exceed the total fees paid under this Agreement.
Either party may terminate this Agreement for convenience upon 30 days notice.
The license granted herein is non-transferable and non-exclusive.
This Agreement contains a non-compete clause for a period of two years.
All intellectual property created shall be assigned to the Company as work for hire.
In no event shall either party be liable for consequential damages.
"""

def main():
    print("Loading Stanza English model...")
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse", verbose=False)
    
    print("Loading LexiconScanner...")
    scanner = LexiconScanner()
    print(f"Loaded {len(scanner.terms)} terms from lexicon")
    
    print("\nProcessing test text...")
    doc = nlp(TEST_TEXT)
    
    print("Scanning for legal terms...")
    matches = scanner.scan(doc)
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(matches)} matches found")
    print('='*60)
    
    for i, m in enumerate(matches, 1):
        print(f"\n[{i}] {m.term}")
        print(f"    Category:      {m.category}")
        print(f"    Matched text:  \"{m.matched_text}\"")
        print(f"    Risk:          {m.base_risk} -> {m.adjusted_risk}", end="")
        if m.negated:
            print(" (NEGATED)", end="")
        if m.intensified:
            print(f" (INTENSIFIED: {m.intensifier})", end="")
        print()
        print(f"    Sentence:      \"{m.sentence_text[:80]}...\"" if len(m.sentence_text) > 80 else f"    Sentence:      \"{m.sentence_text}\"")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    summary = scanner.get_summary(matches)
    print(f"Total matches:  {summary['total_matches']}")
    print(f"Overall risk:   {summary['overall_risk']}/5")
    print(f"Risk level:     {summary['risk_level']}")
    
    print(f"\nCategories detected:")
    for cat, data in summary['categories'].items():
        print(f"  - {cat}: {data['count']} matches, max risk {data['max_risk']}")
    
    if summary['critical_clauses']:
        print(f"\nCritical clauses ({len(summary['critical_clauses'])}):")
        for c in summary['critical_clauses']:
            print(f"  - [{c['category']}] {c['term']} (risk {c['risk']})")


if __name__ == "__main__":
    main()
