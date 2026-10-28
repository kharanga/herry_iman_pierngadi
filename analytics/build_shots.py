import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from analytics.court import apply_homography, shot_geometry, shuttleset_pixel_to_court_homography
from analytics.shot_types import type_group_for

DATABASE_PATH = Path("data/scout.duckdb")
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
LEFT_HANDED_PLAYER_NAMES = frozenset({"Carolina MARIN", "Kento MOMOTA"})
POSITION_COLUMNS = {
    "hit": ("hit_x", "hit_y"),
    "landing": ("landing_x", "landing_y"),
    "player": ("player_location_x", "player_location_y"),
    "opponent": ("opponent_location_x", "opponent_location_y"),
}

SHUTTLESET_SHOT_ROWS_SQL = """
WITH distinct_raw_shots AS (
    SELECT DISTINCT * FROM raw_shots
),
rally_scores_after AS (
    SELECT
        distinct_raw_shots.match_id,
        distinct_raw_shots.set_no,
        distinct_raw_shots.rally,
        MAX(distinct_raw_shots.roundscore_A) AS score_a_after,
        MAX(distinct_raw_shots.roundscore_B) AS score_b_after,
        MAX(distinct_raw_shots.getpoint_player) AS labelled_rally_winner
    FROM distinct_raw_shots
    GROUP BY distinct_raw_shots.match_id, distinct_raw_shots.set_no, distinct_raw_shots.rally
),
rally_winners AS (
    SELECT
        rally_scores_after.*,
        COALESCE(
            rally_scores_after.labelled_rally_winner,
            CASE
                WHEN rally_scores_after.score_a_after > LAG(rally_scores_after.score_a_after, 1, 0) OVER (
                    PARTITION BY rally_scores_after.match_id, rally_scores_after.set_no
                    ORDER BY rally_scores_after.rally
                ) THEN 'A'
                ELSE 'B'
            END
        ) AS rally_winner
    FROM rally_scores_after
)
SELECT
    distinct_raw_shots.match_id,
    distinct_raw_shots.set_no,
    CAST(distinct_raw_shots.rally AS INTEGER) AS rally,
    CAST(distinct_raw_shots.ball_round AS INTEGER) AS ball_round,
    distinct_raw_shots.frame_num,
    CAST(matches.fps AS DOUBLE) AS fps,
    distinct_raw_shots.player,
    distinct_raw_shots.type AS type_en,
    COALESCE(distinct_raw_shots.backhand = 1, FALSE) AS backhand,
    distinct_raw_shots.hit_x,
    distinct_raw_shots.hit_y,
    distinct_raw_shots.landing_x,
    distinct_raw_shots.landing_y,
    distinct_raw_shots.player_location_x,
    distinct_raw_shots.player_location_y,
    distinct_raw_shots.opponent_location_x,
    distinct_raw_shots.opponent_location_y,
    CAST(rally_winners.score_a_after - CASE WHEN rally_winners.rally_winner = 'A' THEN 1 ELSE 0 END AS INTEGER) AS roundscore_me,
    CAST(rally_winners.score_b_after - CASE WHEN rally_winners.rally_winner = 'B' THEN 1 ELSE 0 END AS INTEGER) AS roundscore_opp,
    distinct_raw_shots.getpoint_player,
    distinct_raw_shots.lose_reason,
    CASE WHEN distinct_raw_shots.player = 'A' THEN matches.winner ELSE matches.loser END AS hitter_name,
    CASE WHEN distinct_raw_shots.player = 'A' THEN matches.loser ELSE matches.winner END AS receiver_name
FROM distinct_raw_shots
JOIN rally_winners
    ON rally_winners.match_id = distinct_raw_shots.match_id
    AND rally_winners.set_no = distinct_raw_shots.set_no
    AND rally_winners.rally = distinct_raw_shots.rally
JOIN matches ON matches.match_id = distinct_raw_shots.match_id
ORDER BY distinct_raw_shots.match_id, distinct_raw_shots.set_no, distinct_raw_shots.rally, distinct_raw_shots.ball_round
"""

MATCH_HOMOGRAPHIES_SQL = """
SELECT
    homography.match_id,
    homography.homography_matrix,
    homography.downleft_x,
    homography.downleft_y,
    homography.upright_x,
    homography.upright_y
FROM homography
"""

RAW_SHOT_COUNT_SQL = "SELECT COUNT(*) FROM raw_shots"

SHOTS_COLUMNS_SQL = """
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'shots'
ORDER BY ordinal_position
"""

SHOTS_CONSTRAINTS_SQL = """
SELECT constraint_type, constraint_column_names, expression
FROM duckdb_constraints()
WHERE table_name = 'shots'
ORDER BY constraint_type, constraint_column_names, expression
"""

DELETE_SHUTTLESET_SHOTS_SQL = "DELETE FROM shots WHERE shots.source = 'shuttleset'"

INSERT_SHUTTLESET_SHOTS_SQL = """
INSERT INTO shots
SELECT
    'shuttleset' AS source,
    shuttleset_shot_rows.match_id,
    shuttleset_shot_rows.set_no,
    shuttleset_shot_rows.rally,
    shuttleset_shot_rows.ball_round,
    shuttleset_shot_rows.frame_num,
    shuttleset_shot_rows.fps,
    shuttleset_shot_rows.player,
    shuttleset_shot_rows.type_en,
    shuttleset_shot_rows.type_group,
    CAST(NULL AS VARCHAR) AS type_pred,
    CAST(NULL AS DOUBLE) AS type_conf,
    shuttleset_shot_rows.backhand,
    shuttleset_shot_rows.hit_x,
    shuttleset_shot_rows.hit_y,
    shuttleset_shot_rows.landing_x,
    shuttleset_shot_rows.landing_y,
    shuttleset_shot_rows.player_x,
    shuttleset_shot_rows.player_y,
    shuttleset_shot_rows.opponent_x,
    shuttleset_shot_rows.opponent_y,
    shuttleset_shot_rows.hit_zone,
    shuttleset_shot_rows.landing_zone,
    shuttleset_shot_rows.is_cross,
    shuttleset_shot_rows.roundscore_me,
    shuttleset_shot_rows.roundscore_opp,
    shuttleset_shot_rows.getpoint_player,
    shuttleset_shot_rows.lose_reason
FROM shuttleset_shot_rows
"""


class ShotsSchemaMismatchError(RuntimeError):
    def __init__(
        self,
        existing_columns: list[tuple[str, str]],
        expected_columns: list[tuple[str, str]],
        existing_constraints: list[tuple] | None = None,
        expected_constraints: list[tuple] | None = None,
    ):
        differing_constraints = sorted(set(existing_constraints or []) ^ set(expected_constraints or []), key=str)
        super().__init__(
            "the shots table in the database does not match analytics/schema.sql "
            "(drop the shots table and rebuild); "
            f"existing columns {existing_columns}, expected {expected_columns}, "
            f"differing constraints {differing_constraints}"
        )
        self.existing_columns = existing_columns
        self.expected_columns = expected_columns
        self.existing_constraints = existing_constraints or []
        self.expected_constraints = expected_constraints or []


def shots_columns(connection: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    return connection.execute(SHOTS_COLUMNS_SQL).fetchall()


def shots_constraints(connection: duckdb.DuckDBPyConnection) -> list[tuple]:
    return [
        (constraint_type, tuple(column_names), expression)
        for constraint_type, column_names, expression in connection.execute(SHOTS_CONSTRAINTS_SQL).fetchall()
    ]


def schema_columns_and_constraints() -> tuple[list[tuple[str, str]], list[tuple]]:
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(SCHEMA_PATH.read_text())
        return shots_columns(connection), shots_constraints(connection)
    finally:
        connection.close()


def schema_columns() -> list[tuple[str, str]]:
    return schema_columns_and_constraints()[0]


def create_shots_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(SCHEMA_PATH.read_text())
    existing_columns, existing_constraints = shots_columns(connection), shots_constraints(connection)
    expected_columns, expected_constraints = schema_columns_and_constraints()
    if existing_columns != expected_columns or existing_constraints != expected_constraints:
        raise ShotsSchemaMismatchError(existing_columns, expected_columns, existing_constraints, expected_constraints)


def pixel_to_court_homography_by_match(connection: duckdb.DuckDBPyConnection) -> dict[str, np.ndarray]:
    return {
        match_id: shuttleset_pixel_to_court_homography(matrix, (near_left_x, near_left_y), (far_right_x, far_right_y))
        for match_id, matrix, near_left_x, near_left_y, far_right_x, far_right_y in connection.execute(
            MATCH_HOMOGRAPHIES_SQL
        ).fetchall()
    }


def positions_in_court_meters(shot_rows: pd.DataFrame, homography_by_match: dict[str, np.ndarray]) -> pd.DataFrame:
    meters = pd.DataFrame(index=shot_rows.index)
    for match_id, match_rows in shot_rows.groupby("match_id"):
        for position_name, (x_column, y_column) in POSITION_COLUMNS.items():
            court_points = apply_homography(homography_by_match[match_id], match_rows[[x_column, y_column]].to_numpy(float))
            meters.loc[match_rows.index, f"{position_name}_x"] = court_points[:, 0]
            meters.loc[match_rows.index, f"{position_name}_y"] = court_points[:, 1]
    return meters


def point_or_none(x: float, y: float) -> tuple[float, float] | None:
    if pd.isna(x) or pd.isna(y):
        return None
    return (float(x), float(y))


def geometry_columns(
    shot_rows: pd.DataFrame, meters: pd.DataFrame, left_handed_player_names: frozenset[str]
) -> pd.DataFrame:
    geometry_rows = []
    for index, shot in shot_rows.iterrows():
        geometry = shot_geometry(
            **{
                position_name: point_or_none(meters.at[index, f"{position_name}_x"], meters.at[index, f"{position_name}_y"])
                for position_name in POSITION_COLUMNS
            },
            hitter_left_handed=shot["hitter_name"] in left_handed_player_names,
            receiver_left_handed=shot["receiver_name"] in left_handed_player_names,
        )
        geometry_row = {"hit_zone": geometry["hit_zone"], "landing_zone": geometry["landing_zone"], "is_cross": geometry["is_cross"]}
        for position_name in POSITION_COLUMNS:
            point = geometry[position_name]
            geometry_row[f"{position_name}_x"] = None if point is None else point[0]
            geometry_row[f"{position_name}_y"] = None if point is None else point[1]
        geometry_rows.append(geometry_row)
    return pd.DataFrame(geometry_rows, index=shot_rows.index).astype(
        {"is_cross": "boolean", **{f"{name}_{axis}": "float64" for name in POSITION_COLUMNS for axis in "xy"}}
    )


def shuttleset_shots_frame(
    connection: duckdb.DuckDBPyConnection, left_handed_player_names: frozenset[str]
) -> pd.DataFrame:
    shot_rows = connection.execute(SHUTTLESET_SHOT_ROWS_SQL).df()
    meters = positions_in_court_meters(shot_rows, pixel_to_court_homography_by_match(connection))
    geometry = geometry_columns(shot_rows, meters, left_handed_player_names)
    pixel_columns = [column for columns in POSITION_COLUMNS.values() for column in columns]
    shots_frame = pd.concat([shot_rows.drop(columns=pixel_columns), geometry], axis=1)
    shots_frame["type_group"] = shots_frame["type_en"].map(type_group_for)
    return shots_frame


def build_shots(database_path: Path, left_handed_player_names: frozenset[str] = LEFT_HANDED_PLAYER_NAMES) -> dict:
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute("BEGIN TRANSACTION")
        create_shots_table(connection)
        shuttleset_shot_rows = shuttleset_shots_frame(connection, frozenset(left_handed_player_names))
        connection.execute(DELETE_SHUTTLESET_SHOTS_SQL)
        connection.register("shuttleset_shot_rows", shuttleset_shot_rows)
        connection.execute(INSERT_SHUTTLESET_SHOTS_SQL)
        connection.unregister("shuttleset_shot_rows")
        raw_shot_count = connection.execute(RAW_SHOT_COUNT_SQL).fetchone()[0]
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return {
        "raw_shots": raw_shot_count,
        "shuttleset_shots": len(shuttleset_shot_rows),
        "duplicates_dropped": raw_shot_count - len(shuttleset_shot_rows),
    }


if __name__ == "__main__":
    database_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATABASE_PATH
    summary = build_shots(database_path)
    print(
        f"shots: {summary['shuttleset_shots']} shuttleset rows "
        f"from {summary['raw_shots']} raw shots ({summary['duplicates_dropped']} exact duplicates dropped)"
    )
