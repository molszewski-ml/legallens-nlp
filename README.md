# LegalLens

Legal document risk analyzer that combines a self-developed lexicon-based assessment with a locally-run LLM for contextual analysis of contract clauses. Built as a first coursework project for the Natural Language Processing course at [Universidade Portucalense](https://www.upt.pt/), during my exchange semester in Porto (2025).

## What it does

You drop a contract (PDF, DOCX, or TXT) into the `input/` folder. LegalLens:

1. Extracts and cleans the text.
2. Runs a security scan on the file itself (magic bytes, embedded JavaScript in PDFs, VBA macros in DOCX, prompt injection patterns in the extracted text).
3. Splits the contract into clauses using section-header and sentence-boundary heuristics.
4. Scans each clause with a custom legal lexicon (indemnification, liability, non-compete, IP assignment, and so on), using Stanza for POS/lemma/dependency parsing to detect negation and intensifiers.
5. Sends flagged clauses (risk ≥ 3) to a local LLM through Ollama for a plain-language explanation and recommendation.
6. Generates a JSON report and an HTML report you can open in the browser.

The whole thing runs offline on your machine. No cloud calls, no data leaving your laptop.

## Architecture

```
Contract file (PDF / DOCX / TXT)
    │
    ▼
document_extractor.py  ─── security layer (magic bytes, JS, macros, injection)
    │
    ▼
clause_chunker.py       ─── section headers + sentence boundaries
    │
    ▼
Stanza NLP pipeline     ─── tokenize, POS, lemma, dependency parse, NER
    │
    ▼
lexicon_scanner.py      ─── custom legal lexicon + negation/intensifier logic
    │
    ▼
ollama_client.py        ─── local LLM analysis of flagged clauses (fallback: lexicon only)
    │
    ▼
report_generator.py     ─── JSON + HTML report
```

## Security layer

Because LLM applications are a common vector for prompt injection and because the input files come from users, the extractor validates every file before any text ever reaches the model:

- File size limit (50 MB).
- Magic byte validation catches file-type spoofing (e.g., an executable renamed to `.pdf`).
- Embedded JavaScript inside PDFs is blocked.
- VBA macros and OLE/ActiveX objects inside DOCX are blocked.
- Extracted text is scanned for common prompt injection patterns before being sent to the LLM. Matches produce warnings attached to the result, so the pipeline can decide how to handle them.

## Stack

- Python 3.10+
- [Stanza](https://stanfordnlp.github.io/stanza/) for tokenization, POS, lemmatization, and dependency parsing
- [PyMuPDF](https://pymupdf.readthedocs.io/) and [python-docx](https://python-docx.readthedocs.io/) for extraction
- [Ollama](https://ollama.com/) with [Qwen 3.5 9B](https://ollama.com/library/qwen3.5:9b) as the local LLM
- [Jinja2](https://jinja.palletsprojects.com/) for HTML reports
- [Watchdog](https://python-watchdog.readthedocs.io/) for the file-watcher mode

## Requirements

- Python 3.10 or newer
- ~4 GB RAM for Stanza
- Ollama running locally with `qwen3.5:9b` pulled (~6.6 GB)
- ~10 GB disk space if you download the benchmark datasets

## Running it

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama and pull the model
ollama serve
ollama pull qwen3.5:9b

# Option A: analyze a single file
python main.py input/contract.pdf

# Option B: file watcher (drop files into input/, reports appear in output/)
python watcher.py

# Option C: lexicon only, no LLM
python main.py --no-llm input/contract.pdf
```

## Evaluation

The `benchmark_cuad.py` script evaluates LegalLens against two standard legal NLP datasets:

- [CUAD](https://www.atticusprojectai.org/cuad) (510 commercial contracts, 13,000+ annotations, 41 categories)
- [ContractNLI](https://stanfordnlp.github.io/contract-nli/) (607 annotated contracts)

It computes per-category precision, recall, and F1 against expert annotations.

```bash
# Download datasets first
python download_datasets.py

# Run CUAD benchmark (default 10 contracts)
python benchmark_cuad.py --sample 20

# Run ContractNLI benchmark
python benchmark_cuad.py --dataset contractnli --sample 10

# Both datasets, lexicon only
python benchmark_cuad.py --dataset both --no-llm
```

## Status and honest limitations

This is coursework, not a production tool. Specifically:

- The lexicon-based risk scoring is heuristic. Adjustments for negation (`-2`) and intensification (`+1`) were chosen by hand, not learned from data.
- The lexicon covers common clause categories but is not exhaustive. Domain-specific contracts (medical, defense, real estate) would need extensions.
- The LLM analysis is only as good as the prompt and the local model. It explains what the lexicon already flagged, not make independent legal judgments.
- The fallback mechanism ensures the pipeline still works when Ollama is down or the model fails to produce valid JSON.

Things I'd change with more time: replacing the hand-tuned scoring with a supervised model trained on CUAD labels, and letting the LLM propose alternative wording, not just risk assessments.

## Course context

Built for the Natural Language Processing course, Master's in Data Science, Universidade Portucalense, spring 2025. The original assignment brief is included in [`docs/assignment-requirements.pdf`](docs/assignment-requirements.pdf) for full transparency about what was asked and what was delivered.

## License

MIT. See [`LICENSE`](LICENSE).

## Author

Michał Olszewski. Part of my Master's coursework in AI/ML.
Reach me on [LinkedIn](https://www.linkedin.com/in/mic-olszewski/).
