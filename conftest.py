import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
EXAMPLES_NEED_PATTERN = re.compile(r"^Examples need: (?P<paths>[^(\n]+?)\s*(?:\((?P<hint>[^)\n]*)\))?\s*$", re.MULTILINE)


def declared_example_inputs(markdown_path: Path) -> tuple[list[str], str | None]:
    match = EXAMPLES_NEED_PATTERN.search(Path(markdown_path).read_text(encoding="utf-8"))
    if match is None:
        return [], None
    paths = [path.strip().strip("`") for path in match.group("paths").split(",") if path.strip()]
    return paths, match.group("hint")


def missing_example_inputs(markdown_path: Path) -> tuple[list[str], str | None]:
    paths, hint = declared_example_inputs(markdown_path)
    return [path for path in paths if not (PROJECT_ROOT / path).exists()], hint


def skip_reason(missing_paths: list[str], hint: str | None) -> str:
    reason = f"needs {', '.join(missing_paths)}"
    return f"{reason} ({hint})" if hint else reason


def pytest_collection_modifyitems(config, items):
    for item in items:
        item_path = getattr(item, "path", None)
        if item_path is None or Path(item_path).suffix != ".md":
            continue
        missing_paths, hint = missing_example_inputs(Path(item_path))
        if missing_paths:
            item.add_marker(pytest.mark.skip(reason=skip_reason(missing_paths, hint)))
