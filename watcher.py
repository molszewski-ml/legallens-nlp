"""
LegalLens - File Watcher
==========================
Monitors the input/ folder for new PDF, DOCX, or TXT files.
When a file appears, runs the full pipeline and generates reports
in output/. Processed files are moved to processing/.

Usage:
    python watcher.py
    python watcher.py --no-llm
    python watcher.py --host http://localhost:11435
    python watcher.py --model qwen3:8b
"""

import sys
import shutil
import time
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.pipeline import LegalLensPipeline
from src.report_generator import generate_report


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
PROCESSING_DIR = PROJECT_ROOT / "processing"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class ContractHandler(FileSystemEventHandler):
    """Handles new file events in the input directory."""

    def __init__(self, pipeline: LegalLensPipeline):
        super().__init__()
        self.pipeline = pipeline

    def on_created(self, event):
        if event.is_directory:
            return

        filepath = Path(event.src_path)

        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        # Brief delay — let the OS finish writing the file
        time.sleep(1)

        if not filepath.exists():
            return

        print(f"\nNew file detected: {filepath.name}")
        print("-" * 50)

        try:
            result = self.pipeline.analyze(str(filepath))

            # Console summary
            print(f"Clauses: {result.clauses_total} | Flagged: {result.clauses_flagged}")
            print(f"Lexicon matches: {len(result.lexicon_matches)}")
            print(f"LLM analyses: {len(result.llm_analyses)}")
            print(f"Time: {result.processing_time}s")

            # JSON report
            OUTPUT_DIR.mkdir(exist_ok=True)
            json_path = OUTPUT_DIR / f"{filepath.stem}_report.json"
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(result.to_json())
            print(f"JSON report: {json_path}")

            # HTML report
            html_path = generate_report(result)
            print(f"HTML report: {html_path}")

            # Move processed file
            PROCESSING_DIR.mkdir(exist_ok=True)
            dest = PROCESSING_DIR / filepath.name
            if dest.exists():
                dest = PROCESSING_DIR / f"{filepath.stem}_{int(time.time())}{filepath.suffix}"
            shutil.move(str(filepath), str(dest))
            print(f"File moved to: {dest}")

        except Exception as e:
            print(f"Error processing {filepath.name}: {e}")

        print("-" * 50)
        print("Watching for new files...")


def main():
    # Parse args
    use_llm = True
    host = "http://localhost:11434"
    model = "qwen3.5:9b"

    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--no-llm":
            use_llm = False
        elif arg == "--host" and i + 1 < len(sys.argv) - 1:
            host = sys.argv[i + 2]
        elif arg == "--model" and i + 1 < len(sys.argv) - 1:
            model = sys.argv[i + 2]

    # Ensure directories exist
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    PROCESSING_DIR.mkdir(exist_ok=True)

    # Initialize pipeline once (Stanza loads here)
    print("Initializing pipeline...")
    pipeline = LegalLensPipeline(
        use_llm=use_llm,
        ollama_model=model,
        ollama_host=host,
    )

    # Start watcher
    handler = ContractHandler(pipeline)
    observer = Observer()
    observer.schedule(handler, str(INPUT_DIR), recursive=False)
    observer.start()

    mode = "lexicon + LLM" if use_llm else "lexicon only"
    print(f"\nWatching {INPUT_DIR} for new contracts ({mode})")
    print("Drop a PDF, DOCX, or TXT file into input/ to analyze it.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nWatcher stopped.")

    observer.join()


if __name__ == "__main__":
    main()
