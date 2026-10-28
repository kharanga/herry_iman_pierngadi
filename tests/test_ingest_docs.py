import ast
import re
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

from ingest.load_shuttleset import DATABASE_PATH, SHUTTLESET_SET_DIR, build_database, print_summary

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOAD_DOC_PATH = PROJECT_ROOT / "docs" / "analytics" / "load_shuttleset.md"
FETCH_DOC_PATH = PROJECT_ROOT / "docs" / "analytics" / "fetch_shuttleset.md"
LOADER_PATH = PROJECT_ROOT / "ingest" / "load_shuttleset.py"
INGEST_RESULTS_PATH = PROJECT_ROOT / "docs" / "results" / "ingest.md"
LOADER_MODULE_COMMAND = ".venv/bin/python -m ingest.load_shuttleset"
FETCH_SCRIPT_PATH = PROJECT_ROOT / "ingest" / "fetch_shuttleset.sh"

requires_shuttleset = pytest.mark.skipif(
    not (PROJECT_ROOT / SHUTTLESET_SET_DIR).exists(), reason="run ingest/fetch_shuttleset.sh first"
)

SYNTHETIC_SET_CSV = (
    "rally,ball_round,time,frame_num,player,type,hit_x,hit_y\n"
    "1,1,0:00:01,25,A,lob,,\n"
    "1,2,0:00:02,50,B,clear,640.0,400.0\n"
    "2,1,00:00:04,100,B,smash,600.0,420.0\n"
)
SYNTHETIC_MATCH_CSV = "id,video,set\n1,Synthetic_Match,1\n"
SYNTHETIC_HOMOGRAPHY_CSV = (
    'id,video,homography_matrix,upleft_x\n'
    '1,Synthetic_Match,"[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]",400.0\n'
)


def documented_columns(table_name: str) -> list[tuple[str, str]]:
    table_section = LOAD_DOC_PATH.read_text().split(f"### `{table_name}`")[1].split("###")[0]
    return [
        (row.group(1), row.group(2))
        for row in re.finditer(r"^\| (\w+) \| ([\w\[\]]+) \|", table_section, re.MULTILINE)
        if row.group(1) != "Column"
    ]


@pytest.fixture
def synthetic_set_dir(tmp_path) -> Path:
    set_dir = tmp_path / "set"
    (set_dir / "Synthetic_Match").mkdir(parents=True)
    (set_dir / "Synthetic_Match" / "set1.csv").write_text(SYNTHETIC_SET_CSV)
    (set_dir / "match.csv").write_text(SYNTHETIC_MATCH_CSV)
    (set_dir / "homography.csv").write_text(SYNTHETIC_HOMOGRAPHY_CSV)
    return set_dir


@pytest.fixture(scope="module")
def shuttleset_connection(tmp_path_factory):
    if not (PROJECT_ROOT / SHUTTLESET_SET_DIR).exists():
        pytest.skip("run ingest/fetch_shuttleset.sh first")
    database_path = tmp_path_factory.mktemp("docs_scout") / "scout.duckdb"
    build_database(PROJECT_ROOT / SHUTTLESET_SET_DIR, database_path)
    connection = duckdb.connect(str(database_path), read_only=True)
    yield connection
    connection.close()


@requires_shuttleset
@pytest.mark.parametrize("table_name", ["raw_shots", "matches", "homography"])
def test_documented_columns_match_describe_in_order(shuttleset_connection, table_name):
    actual_columns = [
        (column_name, column_type)
        for column_name, column_type, *_ in shuttleset_connection.execute(f"DESCRIBE {table_name}").fetchall()
    ]
    assert documented_columns(table_name) == actual_columns


@requires_shuttleset
def test_documented_row_counts_match_tables(shuttleset_connection):
    doc_text = LOAD_DOC_PATH.read_text()
    for table_name in ("raw_shots", "matches", "homography"):
        documented_count = int(re.search(rf"### `{table_name}` \(([\d,]+) rows\)", doc_text).group(1).replace(",", ""))
        actual_count = shuttleset_connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        assert documented_count == actual_count


@requires_shuttleset
def test_command_output_matches_documented_output(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "ingest.load_shuttleset", str(tmp_path / "cli.duckdb")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    documented_output = LOAD_DOC_PATH.read_text().split("Output on the full dataset:")[1].split("```")[1].strip()
    assert completed.stdout.strip() == documented_output


def test_documented_default_paths_match_code():
    doc_text = LOAD_DOC_PATH.read_text()
    assert f"`{DATABASE_PATH.as_posix()}`" in doc_text
    assert f"`{SHUTTLESET_SET_DIR.as_posix()}`" in doc_text


def test_build_on_synthetic_set_dir_gives_documented_derived_columns(synthetic_set_dir, tmp_path):
    database_path = tmp_path / "nested" / "scout.duckdb"
    build_database(synthetic_set_dir, database_path)
    connection = duckdb.connect(str(database_path), read_only=True)
    shot_rows = connection.execute(
        "SELECT raw_shots.match_id, raw_shots.set_no, raw_shots.time_seconds FROM raw_shots"
    ).fetchall()
    fps_row = connection.execute("SELECT matches.fps, matches.match_id FROM matches").fetchone()
    matrix_row = connection.execute("SELECT homography.homography_matrix FROM homography").fetchone()
    connection.close()
    assert shot_rows == [("Synthetic_Match", 1, 1), ("Synthetic_Match", 1, 2), ("Synthetic_Match", 1, 4)]
    assert fps_row == (25, "Synthetic_Match")
    assert [list(matrix_row_values) for matrix_row_values in matrix_row[0]] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_missing_homography_rolls_back_and_keeps_old_tables(synthetic_set_dir, tmp_path):
    database_path = tmp_path / "scout.duckdb"
    build_database(synthetic_set_dir, database_path)
    (synthetic_set_dir / "homography.csv").unlink()
    with pytest.raises(duckdb.IOException):
        build_database(synthetic_set_dir, database_path)
    connection = duckdb.connect(str(database_path), read_only=True)
    table_counts = connection.execute(
        "SELECT (SELECT COUNT(*) FROM raw_shots), (SELECT COUNT(*) FROM matches), (SELECT COUNT(*) FROM homography)"
    ).fetchone()
    connection.close()
    assert table_counts == (3, 1, 1)


def test_missing_match_csv_raises_io_exception(synthetic_set_dir, tmp_path):
    (synthetic_set_dir / "match.csv").unlink()
    with pytest.raises(duckdb.IOException):
        build_database(synthetic_set_dir, tmp_path / "scout.duckdb")


def test_failed_build_into_new_path_leaves_empty_database(tmp_path):
    database_path = tmp_path / "empty.duckdb"
    with pytest.raises(duckdb.IOException):
        build_database(tmp_path / "no_such_set_dir", database_path)
    connection = duckdb.connect(str(database_path), read_only=True)
    table_count = connection.execute("SELECT COUNT(*) FROM information_schema.tables").fetchone()[0]
    connection.close()
    assert table_count == 0


def test_print_summary_on_missing_database_raises_io_exception(tmp_path):
    with pytest.raises(duckdb.IOException):
        print_summary(tmp_path / "missing.duckdb")


def test_print_summary_without_tables_raises_catalog_exception(tmp_path):
    database_path = tmp_path / "no_tables.duckdb"
    duckdb.connect(str(database_path)).close()
    with pytest.raises(duckdb.CatalogException):
        print_summary(database_path)


def test_print_summary_on_synthetic_database(synthetic_set_dir, tmp_path, capsys):
    database_path = tmp_path / "scout.duckdb"
    build_database(synthetic_set_dir, database_path)
    print_summary(database_path)
    assert capsys.readouterr().out == (
        "raw_shots: 3 shots, 1 matches, 1 sets\n"
        "fps 25: 1 matches (estimates 25.00-25.00)\n"
    )


def test_fetch_doc_paths_match_script():
    script_text = FETCH_SCRIPT_PATH.read_text()
    doc_text = FETCH_DOC_PATH.read_text()
    repo_dir = re.search(r'repo_dir="([^"]+)"', script_text).group(1)
    assert f"`{repo_dir}/`" in doc_text
    assert "sparse-checkout set ShuttleSet" in script_text
    assert "pull --ff-only" in script_text and "--ff-only" in doc_text
    assert f"{repo_dir}/ShuttleSet/set" == SHUTTLESET_SET_DIR.as_posix()


def test_loader_has_no_comments_or_docstrings():
    source_text = LOADER_PATH.read_text()
    module = ast.parse(source_text)
    assert ast.get_docstring(module) is None
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node) is None
    comment_lines = [line for line in source_text.splitlines() if line.lstrip().startswith("#")]
    assert comment_lines == []


def test_fetch_script_has_no_comments_besides_shebang():
    script_lines = FETCH_SCRIPT_PATH.read_text().splitlines()
    assert [line for line in script_lines[1:] if line.lstrip().startswith("#")] == []


def test_docs_show_module_command_and_never_the_script_path_command():
    load_doc_text = LOAD_DOC_PATH.read_text()
    results_text = INGEST_RESULTS_PATH.read_text()
    assert f"{LOADER_MODULE_COMMAND} [database_path]" in load_doc_text
    assert LOADER_MODULE_COMMAND in results_text
    for doc_text in (load_doc_text, results_text, FETCH_DOC_PATH.read_text()):
        assert "python ingest/load_shuttleset.py" not in doc_text


@pytest.mark.parametrize("ingest_source_path", sorted((PROJECT_ROOT / "ingest").glob("*.py")), ids=lambda path: path.name)
def test_ingest_source_has_no_import_path_workaround(ingest_source_path):
    source_text = ingest_source_path.read_text()
    assert "sys.path" not in source_text
    assert "PROJECT_ROOT" not in source_text
    assert "__file__" not in source_text


def test_importing_loader_leaves_sys_path_unchanged():
    completed = subprocess.run(
        [sys.executable, "-c", "import sys; before = list(sys.path); import ingest.load_shuttleset; print(sys.path == before)"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "True"


def test_loader_doc_no_longer_lists_project_root_constant():
    assert "PROJECT_ROOT" not in LOAD_DOC_PATH.read_text()
    assert "sys.path" not in LOAD_DOC_PATH.read_text()


def documented_column_note(column_name: str) -> str:
    return re.search(rf"^\| {column_name} \| VARCHAR \| (.*) \|$", LOAD_DOC_PATH.read_text(), re.MULTILINE).group(1)


@requires_shuttleset
def test_reason_column_docs_match_rally_counts(shuttleset_connection):
    rally_count, rallies_with_reason = shuttleset_connection.execute(
        "SELECT COUNT(*), COUNT(rally_reason) FROM ("
        "SELECT MAX(raw_shots.lose_reason) AS rally_reason FROM raw_shots "
        "GROUP BY raw_shots.match_id, raw_shots.set_no, raw_shots.rally)"
    ).fetchone()
    reason_row_count = shuttleset_connection.execute(
        "SELECT COUNT(*) FROM raw_shots WHERE raw_shots.lose_reason IS NOT NULL"
    ).fetchone()[0]
    assert (rally_count, rallies_with_reason, reason_row_count) == (3683, 3508, 3509)
    assert f"{rallies_with_reason:,} of {rally_count:,} rallies" in documented_column_note("lose_reason")
    assert "same rows as `lose_reason`" in documented_column_note("win_reason")
    assert shuttleset_connection.execute(
        "SELECT COUNT(*) FROM raw_shots WHERE (raw_shots.lose_reason IS NULL) <> (raw_shots.win_reason IS NULL)"
    ).fetchone()[0] == 0


@requires_shuttleset
def test_reasons_only_on_a_rallys_last_shot(shuttleset_connection):
    assert shuttleset_connection.execute(
        "SELECT COUNT(*) FROM ("
        "SELECT raw_shots.lose_reason, raw_shots.ball_round, "
        "MAX(raw_shots.ball_round) OVER (PARTITION BY raw_shots.match_id, raw_shots.set_no, raw_shots.rally) AS last_ball_round "
        "FROM raw_shots) WHERE lose_reason IS NOT NULL AND ball_round < last_ball_round"
    ).fetchone()[0] == 0


def documented_reason_pair_counts() -> dict[tuple[str, str], tuple[int, int]]:
    rows = re.findall(r"^\| ([a-z ,]+) \|\s*([a-z ,]*?)\s*\| ([\d,]+) \| ([\d,]+) \|$", INGEST_RESULTS_PATH.read_text(), re.MULTILINE)
    return {
        (lose_reason.strip(), win_reason.strip()): (int(shots.replace(",", "")), int(rallies.replace(",", "")))
        for lose_reason, win_reason, shots, rallies in rows
    }


@requires_shuttleset
def test_results_reason_table_shots_and_rallies_match_data(shuttleset_connection):
    measured_counts = {
        (lose_reason, win_reason): (shot_count, rally_count)
        for lose_reason, win_reason, shot_count, rally_count in shuttleset_connection.execute(
            "SELECT raw_shots.lose_reason, raw_shots.win_reason, COUNT(*), "
            "COUNT(DISTINCT (raw_shots.match_id, raw_shots.set_no, raw_shots.rally)) FROM raw_shots "
            "WHERE raw_shots.lose_reason IS NOT NULL GROUP BY raw_shots.lose_reason, raw_shots.win_reason"
        ).fetchall()
    }
    measured_counts[("total", "")] = shuttleset_connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT (raw_shots.match_id, raw_shots.set_no, raw_shots.rally)) FROM raw_shots "
        "WHERE raw_shots.lose_reason IS NOT NULL"
    ).fetchone()
    assert documented_reason_pair_counts() == measured_counts
    assert "3,508 of 3,683 rallies" in INGEST_RESULTS_PATH.read_text()
