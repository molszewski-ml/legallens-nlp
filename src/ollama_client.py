"""
LegalLens - Ollama LLM Client
================================
Handles communication with the Ollama API for contextual
risk analysis of flagged legal clauses.

The LLM receives:
    - The flagged clause text
    - The lexicon category and risk score
    - Relevant legal references

And returns:
    - Contextual risk assessment (1-5)
    - Plain-language explanation
    - Recommendation for the reviewer
"""

import json
import requests


ANALYSIS_PROMPT = """You are a legal risk analyst. Analyze the following contract clause.

CLAUSE:
"{clause_text}"

PRELIMINARY ASSESSMENT:
- Category: {category}
- Lexicon Risk Score: {risk}/5
- Related Legal References: {legal_refs}

Provide your analysis as a JSON object with exactly these fields:
{{
    "risk_level": <integer 1-5>,
    "risk_label": "<LOW|MODERATE|HIGH|CRITICAL>",
    "explanation": "<2-3 sentences explaining the risk in plain language>",
    "key_concern": "<single most important issue>",
    "recommendation": "<specific action the reviewer should take>"
}}

Respond ONLY with the JSON object, no other text."""


class OllamaClient:
    """Client for Ollama REST API."""

    def __init__(
        self,
        model: str = "qwen3.5:9b",
        host: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout: int = 120,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.api_url = f"{self.host}/api/generate"

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to Ollama and return the response text.

        Args:
            prompt: The full prompt string.

        Returns:
            Model response as string.

        Raises:
            ConnectionError: If Ollama is not running.
            RuntimeError: If the API returns an error.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 512,
            },
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
            )
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.host}. "
                "Start it with: ollama serve"
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama API error {response.status_code}: {response.text}"
            )

        data = response.json()
        return data.get("response", "")

    def analyze_clause(
        self,
        clause_text: str,
        lexicon_category: str,
        lexicon_risk: int,
        legal_refs: list,
    ) -> dict:
        """
        Analyze a flagged clause using the LLM.

        Args:
            clause_text: The sentence/clause text.
            lexicon_category: Category from lexicon scanner.
            lexicon_risk: Risk score from lexicon scanner (1-5).
            legal_refs: List of legal references from lexicon.

        Returns:
            Dictionary with risk_level, explanation, key_concern, recommendation.
            Falls back to lexicon-only result if LLM fails.
        """
        refs_str = ", ".join(legal_refs) if legal_refs else "None identified"

        prompt = ANALYSIS_PROMPT.format(
            clause_text=clause_text,
            category=lexicon_category,
            risk=lexicon_risk,
            legal_refs=refs_str,
        )

        try:
            raw = self.generate(prompt)
            result = self._parse_json_response(raw)
            result["clause_text"] = clause_text
            result["source"] = "llm"
            return result

        except Exception:
            # Fallback: return lexicon-only assessment
            return {
                "risk_level": lexicon_risk,
                "risk_label": self._risk_label(lexicon_risk),
                "explanation": f"Lexicon-based assessment: {lexicon_category} clause detected.",
                "key_concern": lexicon_category,
                "recommendation": "Manual review recommended.",
                "clause_text": clause_text,
                "source": "lexicon_fallback",
            }

    def _parse_json_response(self, raw: str) -> dict:
        """
        Extract JSON from LLM response.
        Handles cases where model wraps JSON in markdown code blocks.
        """
        text = raw.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Find JSON object boundaries
        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")

        json_str = text[start:end]
        result = json.loads(json_str)

        # Validate required fields
        required = ["risk_level", "explanation", "recommendation"]
        for field in required:
            if field not in result:
                raise ValueError(f"Missing field: {field}")

        # Ensure risk_level is int 1-5
        result["risk_level"] = max(1, min(5, int(result["risk_level"])))

        if "risk_label" not in result:
            result["risk_label"] = self._risk_label(result["risk_level"])

        return result

    @staticmethod
    def _risk_label(level: int) -> str:
        if level >= 5:
            return "CRITICAL"
        elif level >= 4:
            return "HIGH"
        elif level >= 3:
            return "MODERATE"
        elif level >= 2:
            return "LOW"
        return "INFORMATIONAL"

    def is_available(self) -> bool:
        """Check if Ollama server is running and model is loaded."""
        try:
            r = requests.get(f"{self.host}", timeout=5)
            return r.status_code == 200
        except Exception:
            return False
