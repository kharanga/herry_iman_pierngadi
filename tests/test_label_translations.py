import ast
import re
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

from ingest.label_translations import (
    ENGLISH_BY_CHINESE_BY_RAW_SHOTS_COLUMN,
    LOSE_REASON_ENGLISH_BY_CHINESE,
    SHOT_TYPE_ENGLISH_BY_CHINESE,
    WIN_REASON_ENGLISH_BY_CHINESE,
    UnmappedLabelError,
    english_label_by_source_label,
    translate_label,
)
from ingest.load_shuttleset import RAW_SHOTS_SQL, SHUTTLESET_SET_DIR, build_database

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_SET_DIR = PROJECT_ROOT / SHUTTLESET_SET_DIR
REAL_DATABASE_PATH = PROJECT_ROOT / "data" / "scout.duckdb"
TRANSLATED_COLUMNS = ["type", "lose_reason", "win_reason"]
CJK_PATTERN = "[　-〿㐀-䶿一-鿿豈-﫿＀-￯]"

requires_shuttleset = pytest.mark.skipif(not REAL_SET_DIR.exists(), reason="run ingest/fetch_shuttleset.sh first")

SYNTHETIC_HEADER = "rally,ball_round,time,frame_num,player,type,lose_reason,win_reason\n"
SYNTHETIC_ROWS = (
    "1,1,0:00:01,25,A,發短球,,\n"
    "1,2,0:00:02,50,B,過度切球,,\n"
    "1,3,0:00:03,75,A,殺球,出界,對手出界\n"
    "2,1,00:00:04,100,B,未知球種,對手落地判斷失誤,落地判斷失誤\n"
)


EXPECTED_SHOT_TYPE_ENGLISH_BY_CHINESE = {
    "放小球": "net shot",
    "擋小球": "return net",
    "殺球": "smash",
    "點扣": "wrist smash",
    "挑球": "lob",
    "防守回挑": "defensive return lob",
    "長球": "clear",
    "平球": "drive",
    "小平球": "driven flight",
    "後場抽平球": "back-court drive",
    "切球": "drop",
    "過度切球": "passive drop",
    "過渡切球": "passive drop",
    "推球": "push",
    "撲球": "rush",
    "防守回抽": "defensive return drive",
    "勾球": "cross-court net shot",
    "發短球": "short service",
    "發長球": "long service",
    "未知球種": "unknown",
}

EXPECTED_REASON_PAIRS = [
    (("對手落地致勝", "opponent winner"), ("落地致勝", "winner")),
    (("出界", "out"), ("對手出界", "opponent out")),
    (("掛網", "net"), ("對手掛網", "opponent net")),
    (("未過網", "not over net"), ("對手未過網", "opponent not over net")),
    (("落點判斷失誤", "misjudged landing"), ("對手落點判斷失誤", "opponent misjudged landing")),
    (("對手落地判斷失誤", "misjudged opponent landing"), ("落地判斷失誤", "landed, opponent misjudged")),
    (("犯規", "foul"), ("對手犯規", "opponent foul")),
]
EXPECTED_LOSE_REASON_ENGLISH_BY_CHINESE = dict(lose_pair for lose_pair, _ in EXPECTED_REASON_PAIRS)
EXPECTED_WIN_REASON_ENGLISH_BY_CHINESE = dict(win_pair for _, win_pair in EXPECTED_REASON_PAIRS)
LABEL_DOC_PATH = PROJECT_ROOT / "docs" / "analytics" / "label_translations.md"


def markdown_table_rows(markdown_section: str) -> list[list[str]]:
    table_lines = [line for line in markdown_section.splitlines() if line.startswith("|")]
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in table_lines[2:]]


def documented_shot_type_map() -> dict[str, str]:
    shot_type_section = LABEL_DOC_PATH.read_text().split("Shot types:")[1].split("Rally-end reasons")[0]
    shot_type_map = {}
    for chinese_cell, english in markdown_table_rows(shot_type_section):
        for chinese in chinese_cell.split(", "):
            shot_type_map[chinese] = english
    return shot_type_map


def documented_reason_pairs() -> list[tuple[tuple[str, str], tuple[str, str]]]:
    reason_section = LABEL_DOC_PATH.read_text().split("Rally-end reasons")[1].split("\n\n", 1)[1].split("\n\n")[0]
    return [
        ((lose_chinese, lose_english), (win_chinese, win_english))
        for lose_chinese, lose_english, win_chinese, win_english in markdown_table_rows(reason_section)
    ]


def write_synthetic_set_dir(set_dir: Path, rows: str) -> Path:
    (set_dir / "Synthetic_Match").mkdir(parents=True, exist_ok=True)
    (set_dir / "Synthetic_Match" / "set1.csv").write_text(SYNTHETIC_HEADER + rows)
    (set_dir / "match.csv").write_text("id,video,set\n1,Synthetic_Match,1\n")
    (set_dir / "homography.csv").write_text(
        'id,video,homography_matrix\n1,Synthetic_Match,"[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]"\n'
    )
    return set_dir


def fetch_all(database_path: Path, sql: str) -> list[tuple]:
    connection = duckdb.connect(str(database_path), read_only=True)
    rows = connection.execute(sql).fetchall()
    connection.close()
    return rows


def varchar_columns(connection, table_name: str) -> list[str]:
    return [
        column_name
        for column_name, column_type, *_ in connection.execute(f"DESCRIBE {table_name}").fetchall()
        if column_type == "VARCHAR"
    ]


def test_shot_type_map_matches_expected_table_exactly():
    assert SHOT_TYPE_ENGLISH_BY_CHINESE == EXPECTED_SHOT_TYPE_ENGLISH_BY_CHINESE
    assert len(SHOT_TYPE_ENGLISH_BY_CHINESE) == 20


def test_reason_maps_match_expected_table_exactly():
    assert LOSE_REASON_ENGLISH_BY_CHINESE == EXPECTED_LOSE_REASON_ENGLISH_BY_CHINESE
    assert WIN_REASON_ENGLISH_BY_CHINESE == EXPECTED_WIN_REASON_ENGLISH_BY_CHINESE


def test_documented_tables_match_expected_tables():
    assert documented_shot_type_map() == EXPECTED_SHOT_TYPE_ENGLISH_BY_CHINESE
    assert documented_reason_pairs() == EXPECTED_REASON_PAIRS


def test_misjudged_pairs_stay_separate():
    assert LOSE_REASON_ENGLISH_BY_CHINESE["落點判斷失誤"] == "misjudged landing"
    assert LOSE_REASON_ENGLISH_BY_CHINESE["對手落地判斷失誤"] == "misjudged opponent landing"
    assert WIN_REASON_ENGLISH_BY_CHINESE["對手落點判斷失誤"] == "opponent misjudged landing"
    assert WIN_REASON_ENGLISH_BY_CHINESE["落地判斷失誤"] == "landed, opponent misjudged"
    assert len(set(LOSE_REASON_ENGLISH_BY_CHINESE.values())) == 7
    assert len(set(WIN_REASON_ENGLISH_BY_CHINESE.values())) == 7


def test_english_values_contain_no_chinese():
    for english_by_chinese in ENGLISH_BY_CHINESE_BY_RAW_SHOTS_COLUMN.values():
        assert all(re.search(CJK_PATTERN, english) is None for english in english_by_chinese.values())
        assert all(re.search(CJK_PATTERN, chinese) is not None for chinese in english_by_chinese)


def test_both_passive_drop_spellings_and_unknown():
    assert translate_label("type", "過度切球") == "passive drop"
    assert translate_label("type", "過渡切球") == "passive drop"
    assert translate_label("type", "未知球種") == "unknown"


def test_translate_label_none_stays_none():
    for column_name in TRANSLATED_COLUMNS:
        assert translate_label(column_name, None) is None


@pytest.mark.parametrize("label", ["殺球 ", " 殺球", "smash ", "Smash", "", "飛球", "net shot\n"])
def test_translate_label_near_miss_labels_raise(label):
    with pytest.raises(UnmappedLabelError) as raised:
        translate_label("type", label)
    assert raised.value.column_name == "type"
    assert raised.value.label == label
    assert "raw_shots.type" in str(raised.value) and repr(label) in str(raised.value)


def test_english_label_from_another_column_is_not_accepted():
    assert translate_label("lose_reason", "net") == "net"
    with pytest.raises(UnmappedLabelError):
        translate_label("win_reason", "net")
    with pytest.raises(UnmappedLabelError):
        translate_label("type", "out")


def test_english_label_by_source_label_unknown_column_raises_key_error():
    with pytest.raises(KeyError):
        english_label_by_source_label("player")


def test_synthetic_build_translates_and_keeps_nulls(tmp_path):
    set_dir = write_synthetic_set_dir(tmp_path / "set", SYNTHETIC_ROWS)
    database_path = tmp_path / "scout.duckdb"
    build_database(set_dir, database_path)
    assert fetch_all(
        database_path,
        "SELECT raw_shots.type, raw_shots.lose_reason, raw_shots.win_reason FROM raw_shots ORDER BY raw_shots.frame_num",
    ) == [
        ("short service", None, None),
        ("passive drop", None, None),
        ("smash", "out", "opponent out"),
        ("unknown", "misjudged opponent landing", "landed, opponent misjudged"),
    ]
    assert fetch_all(database_path, "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'label_translations'") == [(0,)]


@pytest.mark.parametrize(
    "bad_row, column_name, label",
    [
        ("2,2,0:00:05,125,A,飛球,,\n", "type", "飛球"),
        ("2,2,0:00:05,125,A,殺球,新原因,對手出界\n", "lose_reason", "新原因"),
        ("2,2,0:00:05,125,A,殺球,出界,新原因\n", "win_reason", "新原因"),
        ("2,2,0:00:05,125,A,殺球 ,,\n", "type", "殺球 "),
    ],
)
def test_unmapped_label_fails_build_naming_column_and_label_and_rolls_back(tmp_path, bad_row, column_name, label):
    set_dir = write_synthetic_set_dir(tmp_path / "set", SYNTHETIC_ROWS)
    database_path = tmp_path / "scout.duckdb"
    build_database(set_dir, database_path)
    rows_before = fetch_all(database_path, "SELECT * FROM raw_shots ORDER BY raw_shots.frame_num")
    write_synthetic_set_dir(set_dir, SYNTHETIC_ROWS + bad_row)
    with pytest.raises(UnmappedLabelError) as raised:
        build_database(set_dir, database_path)
    assert raised.value.column_name == column_name
    assert raised.value.label == label
    assert f"raw_shots.{column_name}" in str(raised.value) and repr(label) in str(raised.value)
    assert fetch_all(database_path, "SELECT * FROM raw_shots ORDER BY raw_shots.frame_num") == rows_before


def test_unmapped_label_into_new_path_leaves_no_tables(tmp_path):
    set_dir = write_synthetic_set_dir(tmp_path / "set", SYNTHETIC_ROWS + "2,2,0:00:05,125,A,飛球,,\n")
    database_path = tmp_path / "new.duckdb"
    with pytest.raises(UnmappedLabelError):
        build_database(set_dir, database_path)
    assert fetch_all(database_path, "SELECT COUNT(*) FROM information_schema.tables") == [(0,)]


@pytest.fixture(scope="module")
def translated_and_source_databases(tmp_path_factory):
    if not REAL_SET_DIR.exists():
        pytest.skip("run ingest/fetch_shuttleset.sh first")
    database_dir = tmp_path_factory.mktemp("translation")
    translated_path = database_dir / "translated.duckdb"
    source_path = database_dir / "source.duckdb"
    build_database(REAL_SET_DIR, translated_path)
    source_connection = duckdb.connect(str(source_path))
    source_connection.execute(RAW_SHOTS_SQL, [str(REAL_SET_DIR / "*" / "set*.csv")])
    source_connection.close()
    connection = duckdb.connect()
    connection.execute(f"ATTACH '{translated_path}' AS translated (READ_ONLY)")
    connection.execute(f"ATTACH '{source_path}' AS source (READ_ONLY)")
    yield connection, translated_path
    connection.close()


@requires_shuttleset
def test_row_count_and_schema_unchanged(translated_and_source_databases):
    connection, _ = translated_and_source_databases
    assert connection.execute("SELECT COUNT(*) FROM translated.raw_shots").fetchone()[0] == 36484
    assert connection.execute("DESCRIBE translated.raw_shots").fetchall() == connection.execute(
        "DESCRIBE source.raw_shots"
    ).fetchall()


@requires_shuttleset
def test_every_other_column_unchanged_row_by_row(translated_and_source_databases):
    connection, _ = translated_and_source_databases
    untouched_columns = "* EXCLUDE (type, lose_reason, win_reason)"
    for first_schema, second_schema in (("translated", "source"), ("source", "translated")):
        assert connection.execute(
            f"SELECT COUNT(*) FROM (SELECT {untouched_columns} FROM {first_schema}.raw_shots "
            f"EXCEPT ALL SELECT {untouched_columns} FROM {second_schema}.raw_shots)"
        ).fetchone()[0] == 0


@requires_shuttleset
@pytest.mark.parametrize("column_name", TRANSLATED_COLUMNS)
def test_each_row_is_its_own_source_label_translated(translated_and_source_databases, column_name):
    connection, _ = translated_and_source_databases
    row_key = "match_id, set_no, rally, ball_round, frame_num"
    translated_rows = connection.execute(
        f"SELECT {row_key}, {column_name} FROM translated.raw_shots ORDER BY {row_key}"
    ).fetchall()
    source_rows = connection.execute(f"SELECT {row_key}, {column_name} FROM source.raw_shots ORDER BY {row_key}").fetchall()
    assert len(translated_rows) == len(source_rows) == 36484
    expected_rows = [source_row[:-1] + (translate_label(column_name, source_row[-1]),) for source_row in source_rows]
    assert translated_rows == expected_rows


@requires_shuttleset
@pytest.mark.parametrize("column_name", TRANSLATED_COLUMNS)
def test_english_value_counts_equal_chinese_source_counts(translated_and_source_databases, column_name):
    connection, _ = translated_and_source_databases
    english_counts = dict(
        connection.execute(f"SELECT {column_name}, COUNT(*) FROM translated.raw_shots GROUP BY {column_name}").fetchall()
    )
    expected_counts = {}
    for source_label, source_count in connection.execute(
        f"SELECT {column_name}, COUNT(*) FROM source.raw_shots GROUP BY {column_name}"
    ).fetchall():
        english_label = translate_label(column_name, source_label)
        expected_counts[english_label] = expected_counts.get(english_label, 0) + source_count
    assert english_counts == expected_counts
    assert set(english_counts) - {None} <= set(ENGLISH_BY_CHINESE_BY_RAW_SHOTS_COLUMN[column_name].values())


@requires_shuttleset
@pytest.mark.parametrize("column_name", ["lose_reason", "win_reason"])
def test_reason_nulls_stay_null(translated_and_source_databases, column_name):
    connection, _ = translated_and_source_databases
    null_count_sql = "SELECT COUNT(*) FROM {schema}.raw_shots WHERE {column_name} IS NULL"
    translated_nulls = connection.execute(null_count_sql.format(schema="translated", column_name=column_name)).fetchone()[0]
    source_nulls = connection.execute(null_count_sql.format(schema="source", column_name=column_name)).fetchone()[0]
    assert translated_nulls == source_nulls == 32975


@requires_shuttleset
def test_type_has_no_nulls_and_only_english_labels(translated_and_source_databases):
    connection, _ = translated_and_source_databases
    assert connection.execute("SELECT COUNT(*) FROM translated.raw_shots WHERE type IS NULL").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM translated.raw_shots WHERE type = 'passive drop'").fetchone()[0] == 1356
    assert connection.execute("SELECT COUNT(*) FROM translated.raw_shots WHERE type = 'unknown'").fetchone()[0] == 1407


def assert_no_chinese_in_any_table(connection, schema_prefix: str) -> None:
    for table_name in ("raw_shots", "matches", "homography"):
        qualified_table = f"{schema_prefix}{table_name}"
        for column_name in varchar_columns(connection, qualified_table):
            chinese_count = connection.execute(
                f"SELECT COUNT(*) FROM {qualified_table} WHERE regexp_matches(\"{column_name}\", ?)", [CJK_PATTERN]
            ).fetchone()[0]
            assert chinese_count == 0, f"{table_name}.{column_name}"


@requires_shuttleset
def test_no_chinese_character_in_any_column_of_any_table(translated_and_source_databases):
    connection, _ = translated_and_source_databases
    assert_no_chinese_in_any_table(connection, "translated.")
    assert connection.execute(
        "SELECT COUNT(*) FROM source.raw_shots WHERE regexp_matches(type, ?)", [CJK_PATTERN]
    ).fetchone()[0] == 36484


@requires_shuttleset
def test_reason_pairs_follow_expected_table_rows(translated_and_source_databases):
    connection, _ = translated_and_source_databases
    expected_pairs = {(lose_english, win_english) for (_, lose_english), (_, win_english) in EXPECTED_REASON_PAIRS}
    actual_pairs = set(
        connection.execute(
            "SELECT DISTINCT lose_reason, win_reason FROM translated.raw_shots WHERE lose_reason IS NOT NULL OR win_reason IS NOT NULL"
        ).fetchall()
    )
    assert actual_pairs <= expected_pairs


@requires_shuttleset
def test_rebuild_into_same_path_is_identical(translated_and_source_databases, tmp_path):
    connection, translated_path = translated_and_source_databases
    rebuilt_path = tmp_path / "rebuilt.duckdb"
    build_database(REAL_SET_DIR, rebuilt_path)
    build_database(REAL_SET_DIR, rebuilt_path)
    connection.execute(f"ATTACH '{rebuilt_path}' AS rebuilt (READ_ONLY)")
    try:
        for table_name in ("raw_shots", "matches", "homography"):
            for first_schema, second_schema in (("translated", "rebuilt"), ("rebuilt", "translated")):
                assert connection.execute(
                    f"SELECT COUNT(*) FROM (SELECT * FROM {first_schema}.{table_name} "
                    f"EXCEPT ALL SELECT * FROM {second_schema}.{table_name})"
                ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_catalog = 'rebuilt' "
            "AND table_name IN ('label_translations', 'match_frame_rate')"
        ).fetchone()[0] == 0
    finally:
        connection.execute("DETACH rebuilt")


@pytest.mark.skipif(not REAL_DATABASE_PATH.exists(), reason="data/scout.duckdb not built")
def test_real_database_is_english_only():
    connection = duckdb.connect()
    connection.execute(f"ATTACH '{REAL_DATABASE_PATH}' AS real_database (READ_ONLY)")
    try:
        assert connection.execute("SELECT COUNT(*) FROM real_database.raw_shots").fetchone()[0] == 36484
        assert_no_chinese_in_any_table(connection, "real_database.")
    finally:
        connection.close()


@requires_shuttleset
def test_command_runs_as_module(tmp_path):
    database_path = tmp_path / "cli.duckdb"
    completed = subprocess.run(
        [sys.executable, "-m", "ingest.load_shuttleset", str(database_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("raw_shots: 36484 shots, 44 matches, 104 sets\n")
    assert fetch_all(database_path, "SELECT COUNT(*) FROM raw_shots WHERE type = 'net shot'") == [(6290,)]


@pytest.mark.parametrize("source_name", ["label_translations.py", "load_shuttleset.py"])
def test_ingest_source_has_no_comments_or_docstrings(source_name):
    source_text = (PROJECT_ROOT / "ingest" / source_name).read_text()
    module = ast.parse(source_text)
    assert ast.get_docstring(module) is None
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node) is None
    assert [line for line in source_text.splitlines() if line.lstrip().startswith("#")] == []
