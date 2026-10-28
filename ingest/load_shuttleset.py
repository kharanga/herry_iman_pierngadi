import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingest.label_translations import (
    ENGLISH_BY_CHINESE_BY_RAW_SHOTS_COLUMN,
    UnmappedLabelError,
    english_label_by_source_label,
)

SHUTTLESET_SET_DIR = Path("data/CoachAI-Projects/ShuttleSet/set")
DATABASE_PATH = Path("data/scout.duckdb")

RAW_SHOTS_SQL = """
CREATE OR REPLACE TABLE raw_shots AS
SELECT
    regexp_extract(filename, '/set/([^/]+)/set[0-9]+\\.csv$', 1) AS match_id,
    CAST(regexp_extract(filename, '/set([0-9]+)\\.csv$', 1) AS INTEGER) AS set_no,
    CAST(frame_num AS INTEGER) AS frame_num,
    CAST(split_part(time, ':', 1) AS INTEGER) * 3600
        + CAST(split_part(time, ':', 2) AS INTEGER) * 60
        + CAST(split_part(time, ':', 3) AS INTEGER) AS time_seconds,
    * EXCLUDE (filename, frame_num)
FROM read_csv(?, filename = TRUE, union_by_name = TRUE, types = {'time': 'VARCHAR'})
ORDER BY match_id, set_no, rally, ball_round
"""

MATCHES_SQL = """
CREATE OR REPLACE TABLE matches AS
SELECT video AS match_id, *
FROM read_csv(?)
ORDER BY id
"""

HOMOGRAPHY_SQL = """
CREATE OR REPLACE TABLE homography AS
SELECT
    video AS match_id,
    CAST(homography_matrix AS DOUBLE[3][3]) AS homography_matrix,
    * EXCLUDE (homography_matrix)
FROM read_csv(?, types = {'homography_matrix': 'VARCHAR'})
ORDER BY id
"""

LABEL_TRANSLATIONS_TABLE_SQL = """
CREATE OR REPLACE TEMP TABLE label_translations (
    column_name VARCHAR,
    source_label VARCHAR,
    english_label VARCHAR
)
"""

INSERT_LABEL_TRANSLATION_SQL = """
INSERT INTO label_translations VALUES (?, ?, ?)
"""

RAW_SHOTS_COLUMN_NAMES_SQL = """
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'raw_shots'
"""

UNMAPPED_LABELS_SQL_TEMPLATE = """
SELECT DISTINCT raw_shots.{column_name}
FROM raw_shots
WHERE raw_shots.{column_name} IS NOT NULL
    AND raw_shots.{column_name} NOT IN (
        SELECT label_translations.source_label
        FROM label_translations
        WHERE label_translations.column_name = ?
    )
ORDER BY raw_shots.{column_name}
"""

TRANSLATE_COLUMN_SQL_TEMPLATE = """
UPDATE raw_shots
SET {column_name} = (
    SELECT label_translations.english_label
    FROM label_translations
    WHERE label_translations.column_name = ?
        AND label_translations.source_label = raw_shots.{column_name}
)
WHERE raw_shots.{column_name} IS NOT NULL
"""

DROP_LABEL_TRANSLATIONS_SQL = """
DROP TABLE label_translations
"""

FRAME_RATE_SQL = """
CREATE OR REPLACE TABLE match_frame_rate AS
SELECT
    match_id,
    REGR_SLOPE(frame_num, time_seconds) AS fps_estimate,
    REGR_R2(frame_num, time_seconds) AS fps_fit_r2,
    COUNT(*) AS shot_count
FROM raw_shots
GROUP BY match_id
"""

ATTACH_FPS_TO_MATCHES_SQL = """
CREATE OR REPLACE TABLE matches AS
SELECT
    matches.*,
    match_frame_rate.fps_estimate,
    match_frame_rate.fps_fit_r2,
    CAST(ROUND(match_frame_rate.fps_estimate) AS INTEGER) AS fps
FROM matches
LEFT JOIN match_frame_rate ON match_frame_rate.match_id = matches.match_id
ORDER BY matches.id
"""


def load_label_translations(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(LABEL_TRANSLATIONS_TABLE_SQL)
    connection.executemany(
        INSERT_LABEL_TRANSLATION_SQL,
        [
            (column_name, source_label, english_label)
            for column_name in ENGLISH_BY_CHINESE_BY_RAW_SHOTS_COLUMN
            for source_label, english_label in english_label_by_source_label(column_name).items()
        ],
    )


def translated_columns_present(connection: duckdb.DuckDBPyConnection) -> list[str]:
    present_column_names = {row[0] for row in connection.execute(RAW_SHOTS_COLUMN_NAMES_SQL).fetchall()}
    return [column_name for column_name in ENGLISH_BY_CHINESE_BY_RAW_SHOTS_COLUMN if column_name in present_column_names]


def fail_on_unmapped_labels(connection: duckdb.DuckDBPyConnection, column_name: str) -> None:
    unmapped_labels = connection.execute(
        UNMAPPED_LABELS_SQL_TEMPLATE.format(column_name=column_name), [column_name]
    ).fetchall()
    if unmapped_labels:
        raise UnmappedLabelError(column_name, unmapped_labels[0][0])


def translate_raw_shots_to_english(connection: duckdb.DuckDBPyConnection) -> None:
    load_label_translations(connection)
    for column_name in translated_columns_present(connection):
        fail_on_unmapped_labels(connection, column_name)
        connection.execute(TRANSLATE_COLUMN_SQL_TEMPLATE.format(column_name=column_name), [column_name])
    connection.execute(DROP_LABEL_TRANSLATIONS_SQL)


def build_database(set_dir: Path, database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(RAW_SHOTS_SQL, [str(set_dir / "*" / "set*.csv")])
        translate_raw_shots_to_english(connection)
        connection.execute(MATCHES_SQL, [str(set_dir / "match.csv")])
        connection.execute(HOMOGRAPHY_SQL, [str(set_dir / "homography.csv")])
        connection.execute(FRAME_RATE_SQL)
        connection.execute(ATTACH_FPS_TO_MATCHES_SQL)
        connection.execute("DROP TABLE match_frame_rate")
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def print_summary(database_path: Path) -> None:
    connection = duckdb.connect(str(database_path), read_only=True)
    shot_count, match_count, set_count = connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT match_id), COUNT(DISTINCT (match_id, set_no)) FROM raw_shots"
    ).fetchone()
    print(f"raw_shots: {shot_count} shots, {match_count} matches, {set_count} sets")
    for fps, match_count_at_fps, lowest_estimate, highest_estimate in connection.execute(
        "SELECT fps, COUNT(*), MIN(fps_estimate), MAX(fps_estimate) FROM matches GROUP BY fps ORDER BY fps"
    ).fetchall():
        print(f"fps {fps}: {match_count_at_fps} matches (estimates {lowest_estimate:.2f}-{highest_estimate:.2f})")
    connection.close()


if __name__ == "__main__":
    database_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATABASE_PATH
    build_database(SHUTTLESET_SET_DIR, database_path)
    print_summary(database_path)
