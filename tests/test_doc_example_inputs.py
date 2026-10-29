import re
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from analytics.build_shots import SCHEMA_PATH
from conftest import EXAMPLES_NEED_PATTERN, declared_example_inputs, missing_example_inputs, skip_reason
from shots.predict import PREDICTION_SUMMARY_SQL, UNKNOWN_SHOT_PREDICTIONS_SQL, prediction_summary

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHUTTLESET_SET_DIR = "data/CoachAI-Projects/ShuttleSet/set"
SHUTTLESET_DECLARATION = f"Examples need: {SHUTTLESET_SET_DIR} (run ./ingest/fetch_shuttleset.sh)"
DOCS_NEEDING_SHUTTLESET = ("build_shots.md", "fetch_shuttleset.md", "load_shuttleset.md", "spot_check.md")
TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b"


def markdown_with_example(declaration: str | None, expected: str) -> str:
    declaration_line = f"{declaration}\n\n" if declaration else ""
    return f"# Example\n\n{declaration_line}```python\n>>> 1 + 1\n{expected}\n\n```\n"


def test_declaration_parsing_handles_several_paths_back_quotes_and_an_optional_hint(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("# Title\n\nExamples need: `tests`, data/nope/one , data/nope/two.bin (fetch them)\n")
    assert declared_example_inputs(doc) == (["tests", "data/nope/one", "data/nope/two.bin"], "fetch them")
    assert missing_example_inputs(doc) == (["data/nope/one", "data/nope/two.bin"], "fetch them")
    doc.write_text("# Title\n\nExamples need: data/nope/three\n")
    assert declared_example_inputs(doc) == (["data/nope/three"], None)
    doc.write_text("# Title\n\nno declaration here\n")
    assert declared_example_inputs(doc) == ([], None) and missing_example_inputs(doc) == ([], None)
    doc.write_text("# Title\n\n    Examples need: data/nope/four (indented code, not a declaration)\n")
    assert declared_example_inputs(doc) == ([], None)
    assert skip_reason(["a", "b"], "get them") == "needs a, b (get them)" and skip_reason(["a"], None) == "needs a"


def test_existing_file_and_directory_both_count_as_present(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("Examples need: tests/test_doc_example_inputs.py, tests, docs/analytics/README.md\n")
    assert missing_example_inputs(doc) == ([], None)


def run_doctests(doc_dir: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(doc_dir), "-p", "conftest", "-p", "no:cacheprovider", "--doctest-glob=*.md", "-q", "-rs"],
        cwd=PROJECT_ROOT, capture_output=True, text=True)
    return result.stdout


def test_hook_skips_only_docs_whose_declared_input_is_missing_and_cannot_hide_wrong_examples(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "wrong_with_existing_file.md").write_text(markdown_with_example("Examples need: tests/test_doc_example_inputs.py (exists)", "3"))
    (docs / "wrong_with_existing_directory.md").write_text(markdown_with_example("Examples need: `tests`, docs", "3"))
    (docs / "right_with_existing_directory.md").write_text(markdown_with_example("Examples need: tests, docs/analytics", "2"))
    (docs / "wrong_without_declaration.md").write_text(markdown_with_example(None, "3"))
    (docs / "wrong_with_indented_line.md").write_text(markdown_with_example("    Examples need: data/nope/four", "3"))
    (docs / "missing_with_hint.md").write_text(markdown_with_example("Examples need: data/nope/one, tests, data/nope/two.bin (fetch them)", "3"))
    (docs / "missing_without_hint.md").write_text(markdown_with_example("Examples need: data/nope/three", "3"))
    output = run_doctests(docs)
    assert re.search(r"4 failed, 1 passed, 2 skipped", output), output
    for failing in ("wrong_with_existing_file.md", "wrong_with_existing_directory.md", "wrong_without_declaration.md", "wrong_with_indented_line.md"):
        assert f"FAILED docs/{failing}" in output or f"{failing}:" in output
    assert "missing_with_hint.md: needs data/nope/one, data/nope/two.bin (fetch them)" in output
    assert "missing_without_hint.md: needs data/nope/three" in output
    assert "right_with_existing_directory" not in output.split("short test summary info")[-1]


def test_the_four_shuttleset_docs_declare_the_download_and_the_others_declare_nothing_missing():
    for doc_name in DOCS_NEEDING_SHUTTLESET:
        text = (PROJECT_ROOT / "docs" / "analytics" / doc_name).read_text()
        assert SHUTTLESET_DECLARATION in text and len(EXAMPLES_NEED_PATTERN.findall(text)) == 1
    for doc_path in list((PROJECT_ROOT / "docs" / "analytics").glob("*.md")) + list((PROJECT_ROOT / "docs" / "vision").glob("*.md")):
        paths, _ = declared_example_inputs(doc_path)
        if doc_path.name in DOCS_NEEDING_SHUTTLESET:
            assert paths == [SHUTTLESET_SET_DIR]
        else:
            assert paths == [], doc_path


@pytest.mark.skipif(not (PROJECT_ROOT / SHUTTLESET_SET_DIR).is_dir(), reason="run ingest/fetch_shuttleset.sh first")
def test_with_the_download_present_no_doc_is_skipped():
    for doc_path in list((PROJECT_ROOT / "docs" / "analytics").glob("*.md")) + list((PROJECT_ROOT / "docs" / "vision").glob("*.md")):
        assert missing_example_inputs(doc_path)[0] == []


def test_mechanism_is_documented_in_both_readmes_without_task_ids():
    for area in ("analytics", "vision"):
        readme = (PROJECT_ROOT / "docs" / area / "README.md").read_text()
        assert "## Examples that need downloaded data" in readme and "conftest.py" in readme and "Examples need:" in readme
        assert "needs <path>" in readme and not re.search(TASK_ID_PATTERN, readme)
    for doc_name in DOCS_NEEDING_SHUTTLESET:
        assert not re.search(TASK_ID_PATTERN, (PROJECT_ROOT / "docs" / "analytics" / doc_name).read_text())


def test_conftest_has_no_comments_or_docstrings_and_one_hook():
    source = (PROJECT_ROOT / "conftest.py").read_text()
    assert '"""' not in source and "'''" not in source and not re.search(r"(?m)^\s*#", source)
    assert source.count("def pytest_") == 1 and "def pytest_collection_modifyitems" in source


def write_shots_database(database_path: Path, rows: list[dict]) -> Path:
    connection = duckdb.connect(str(database_path))
    connection.execute(SCHEMA_PATH.read_text())
    frame = pd.DataFrame(rows)
    connection.register("synthetic_rows", frame)
    columns = ", ".join(frame.columns)
    connection.execute(f"INSERT INTO shots ({columns}) SELECT {columns} FROM synthetic_rows")
    connection.close()
    return database_path


def shot(source: str, rally: int, type_group: str, type_pred: str | None, type_conf: float | None) -> dict:
    return {"source": source, "match_id": f"{source}_match", "set_no": 1, "rally": rally, "ball_round": 1, "frame_num": rally, "player": "A" if source == "shuttleset" else "me",
            "type_group": type_group, "type_pred": type_pred, "type_conf": type_conf}


def test_prediction_summary_counts_shuttleset_rows_only(tmp_path):
    rows = [shot("shuttleset", 1, "clear", "clear", 0.9), shot("shuttleset", 2, "unknown", "serve", 0.5), shot("shuttleset", 3, "drop", None, None),
            shot("own", 1, "lift", "smash", 0.1), shot("own", 2, "unknown", "net", 0.2), shot("own", 3, "drive", "clear", 0.3)]
    summary = prediction_summary(write_shots_database(tmp_path / "scout.duckdb", rows))
    assert (summary["shots"], summary["predicted_shots"]) == (3, 2)
    assert summary["lowest_conf"] == pytest.approx(0.5) and summary["mean_conf"] == pytest.approx(0.7)
    assert summary["in_sample_agreement"] == pytest.approx(1.0) and summary["unknown_shot_predictions"] == {"serve": 1}
    assert "WHERE shots.source = 'shuttleset'" in PREDICTION_SUMMARY_SQL and "shots.source = 'shuttleset'" in UNKNOWN_SHOT_PREDICTIONS_SQL
