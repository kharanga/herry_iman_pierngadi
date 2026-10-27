import sys
from pathlib import Path

import duckdb

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


def build_database(set_dir: Path, database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(RAW_SHOTS_SQL, [str(set_dir / "*" / "set*.csv")])
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
