from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "docs" / "results"


def is_missing_results_page(path) -> bool:
    resolved = Path(path).resolve()
    return RESULTS_DIR in resolved.parents and not resolved.exists()


def docs_on_disk(docs) -> list[str]:
    return [doc for doc in docs if not is_missing_results_page(PROJECT_ROOT / doc)]
