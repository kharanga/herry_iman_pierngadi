import csv
from pathlib import Path

import duckdb
import pytest

from ingest.load_shuttleset import SHUTTLESET_SET_DIR, build_database

pytestmark = pytest.mark.skipif(
    not SHUTTLESET_SET_DIR.exists(), reason="run ingest/fetch_shuttleset.sh first"
)


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return sum(1 for _ in csv.DictReader(csv_file))


def source_set_files() -> list[Path]:
    return sorted(SHUTTLESET_SET_DIR.glob("*/set*.csv"))


@pytest.fixture(scope="module")
def database_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("scout") / "scout.duckdb"
    build_database(SHUTTLESET_SET_DIR, path)
    return path


@pytest.fixture(scope="module")
def connection(database_path):
    connection = duckdb.connect(str(database_path), read_only=True)
    yield connection
    connection.close()


def fetch_value(connection, sql: str):
    return connection.execute(sql).fetchone()[0]


def test_shot_count_matches_source_csvs(connection):
    expected_shot_count = sum(count_csv_rows(path) for path in source_set_files())
    assert fetch_value(connection, "SELECT COUNT(*) FROM raw_shots") == expected_shot_count


def test_shot_count_per_set_matches_its_csv(connection):
    loaded_counts = dict(
        ((match_id, set_no), shot_count)
        for match_id, set_no, shot_count in connection.execute(
            "SELECT match_id, set_no, COUNT(*) FROM raw_shots GROUP BY match_id, set_no"
        ).fetchall()
    )
    source_counts = {
        (path.parent.name, int(path.stem.removeprefix("set"))): count_csv_rows(path)
        for path in source_set_files()
    }
    assert loaded_counts == source_counts


def test_has_44_matches_and_104_sets(connection):
    assert fetch_value(connection, "SELECT COUNT(DISTINCT match_id) FROM raw_shots") == 44
    assert fetch_value(connection, "SELECT COUNT(DISTINCT (match_id, set_no)) FROM raw_shots") == 104
    assert fetch_value(connection, "SELECT COUNT(*) FROM matches") == 44
    assert fetch_value(connection, "SELECT COUNT(*) FROM homography") == 44


def test_every_shot_has_match_id_and_set_no(connection):
    assert fetch_value(
        connection, "SELECT COUNT(*) FROM raw_shots WHERE match_id IS NULL OR match_id = '' OR set_no IS NULL"
    ) == 0


def test_every_shot_belongs_to_a_known_match(connection):
    assert fetch_value(
        connection,
        "SELECT COUNT(*) FROM raw_shots WHERE match_id NOT IN (SELECT matches.match_id FROM matches)",
    ) == 0


def test_set_numbers_agree_with_match_table(connection):
    assert fetch_value(
        connection,
        """
        SELECT COUNT(*)
        FROM matches
        WHERE matches.set <> (
            SELECT COUNT(DISTINCT raw_shots.set_no) FROM raw_shots WHERE raw_shots.match_id = matches.match_id
        )
        """,
    ) == 0


def test_every_match_has_fps_near_25_or_30(connection):
    assert fetch_value(connection, "SELECT COUNT(*) FROM matches WHERE fps NOT IN (25, 30) OR fps IS NULL") == 0
    assert fetch_value(connection, "SELECT MAX(ABS(fps_estimate - fps)) FROM matches") < 0.1


def test_homography_matrix_is_3_by_3_and_normalized(connection):
    assert fetch_value(
        connection, "SELECT COUNT(*) FROM homography WHERE homography_matrix[3][3] <> 1.0"
    ) == 0


def test_rebuilding_is_idempotent(database_path, connection):
    table_summary_sql = """
        SELECT
            (SELECT COUNT(*) FROM raw_shots),
            (SELECT COUNT(*) FROM matches),
            (SELECT COUNT(*) FROM homography),
            (SELECT SUM(fps) FROM matches)
    """
    summary_before = connection.execute(table_summary_sql).fetchone()
    rebuilt_path = database_path.with_name("rebuilt.duckdb")
    build_database(SHUTTLESET_SET_DIR, rebuilt_path)
    build_database(SHUTTLESET_SET_DIR, rebuilt_path)
    rebuilt_connection = duckdb.connect(str(rebuilt_path), read_only=True)
    summary_after = rebuilt_connection.execute(table_summary_sql).fetchone()
    rebuilt_connection.close()
    assert summary_after == summary_before
