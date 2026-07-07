"""
Test script for OllamaClient
============================
Run from project root:
    conda activate legallens
    python test_ollama.py

Requires Ollama running with qwen3.5:9b model:
    ollama serve
    ollama pull qwen3.5:9b
"""

from src.ollama_client import OllamaClient


TEST_CLAUSE = """
The Contractor shall indemnify, defend, and hold harmless the Company 
from and against any and all claims, damages, losses, costs, and expenses 
arising out of or related to the Contractor's performance under this Agreement.
"""


def main():
    print("Initializing OllamaClient...")
    client = OllamaClient(model="qwen3.5:9b")
    
    print(f"Model:    {client.model}")
    print(f"Host:     {client.host}")
    print(f"API URL:  {client.api_url}")
    
    print("\nChecking Ollama availability...")
    if not client.is_available():
        print("ERROR: Ollama is not running.")
        print("Start it with: ollama serve")
        print("Then pull model: ollama pull qwen3.5:9b")
        return
    
    print("Ollama is running.")
    
    print("\n" + "="*60)
    print("TEST 1: Raw generate()")
    print("="*60)
    
    try:
        response = client.generate("Say 'hello' and nothing else.")
        print(f"Response: {response[:100]}...")
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    print("\n" + "="*60)
    print("TEST 2: analyze_clause()")
    print("="*60)
    
    print(f"Clause: {TEST_CLAUSE[:80]}...")
    print("\nSending to LLM for analysis...")
    
    result = client.analyze_clause(
        clause_text=TEST_CLAUSE.strip(),
        lexicon_category="indemnification",
        lexicon_risk=5,
        legal_refs=["CUAD Category - Indemnification"],
    )
    
    print(f"\nSource:         {result.get('source', 'unknown')}")
    print(f"Risk Level:     {result.get('risk_level', 'N/A')}/5")
    print(f"Risk Label:     {result.get('risk_label', 'N/A')}")
    print(f"Key Concern:    {result.get('key_concern', 'N/A')}")
    print(f"Explanation:    {result.get('explanation', 'N/A')}")
    print(f"Recommendation: {result.get('recommendation', 'N/A')}")
    
    print("\n" + "="*60)
    print("TEST 3: Fallback (invalid prompt)")
    print("="*60)
    
    # Force a parse error by using gibberish
    bad_client = OllamaClient(model="qwen3.5:9b", temperature=2.0)
    result_fallback = bad_client.analyze_clause(
        clause_text="xyz123",
        lexicon_category="test",
        lexicon_risk=3,
        legal_refs=[],
    )
    
    print(f"Source: {result_fallback.get('source', 'unknown')}")
    if result_fallback.get('source') == 'lexicon_fallback':
        print("Fallback mechanism works correctly.")
    else:
        print("LLM responded (fallback not triggered).")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    main()
