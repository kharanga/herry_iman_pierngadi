import duckdb
import numpy as np
import pandas as pd

from analytics.court import NET_Y_M

FEATURE_NAMES = (
    "hit_x",
    "hit_y",
    "landing_x",
    "landing_y",
    "receiver_x",
    "receiver_y",
    "shot_dx",
    "shot_dy",
    "shot_distance",
    "shot_lateral_distance",
    "is_cross",
    "ball_round",
    "flight_seconds",
    "seconds_since_previous_hit",
    "previous_shot_distance",
    "previous_hit_y",
)

SHOT_FEATURE_ROWS_SQL = """
SELECT
    shots.rowid AS row_id,
    shots.source,
    shots.match_id,
    shots.set_no,
    shots.rally,
    shots.ball_round,
    shots.frame_num,
    shots.fps,
    shots.type_en,
    shots.type_group,
    shots.hit_x,
    shots.hit_y,
    shots.landing_x,
    shots.landing_y,
    shots.player_x,
    shots.player_y,
    shots.opponent_x,
    shots.opponent_y,
    shots.is_cross,
    LAG(shots.frame_num) OVER rally_order AS previous_frame_num,
    LEAD(shots.frame_num) OVER rally_order AS next_frame_num,
    LAG(shots.hit_x) OVER rally_order AS previous_hit_x,
    LAG(shots.hit_y) OVER rally_order AS previous_hit_y,
    LAG(shots.landing_x) OVER rally_order AS previous_landing_x,
    LAG(shots.landing_y) OVER rally_order AS previous_landing_y
FROM shots
WINDOW rally_order AS (
    PARTITION BY shots.source, shots.match_id, shots.set_no, shots.rally
    ORDER BY shots.ball_round, shots.frame_num
)
ORDER BY shots.rowid
"""


def load_shot_feature_rows(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute(SHOT_FEATURE_ROWS_SQL).df()


def hitter_point_columns(shot_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    hit_x = shot_rows["hit_x"].to_numpy(float)
    hit_y = shot_rows["hit_y"].to_numpy(float)
    player_x = shot_rows["player_x"].to_numpy(float)
    player_y = shot_rows["player_y"].to_numpy(float)
    hit_is_usable = ~np.isnan(hit_x) & ~np.isnan(hit_y) & (hit_y <= NET_Y_M)
    player_is_missing = np.isnan(player_x) | np.isnan(player_y)
    use_hit = hit_is_usable | player_is_missing
    return np.where(use_hit, hit_x, player_x), np.where(use_hit, hit_y, player_y)


def seconds_between(later_frames: pd.Series, earlier_frames: pd.Series, fps: pd.Series) -> np.ndarray:
    return ((later_frames.astype(float) - earlier_frames.astype(float)) / fps.astype(float)).to_numpy(float)


def has_hitter_and_landing_point(features: pd.DataFrame) -> pd.Series:
    return features[["hit_x", "hit_y", "landing_x", "landing_y"]].notna().all(axis=1)


def shot_features(shot_rows: pd.DataFrame) -> pd.DataFrame:
    hitter_x, hitter_y = hitter_point_columns(shot_rows)
    landing_x = shot_rows["landing_x"].to_numpy(float)
    landing_y = shot_rows["landing_y"].to_numpy(float)
    shot_dx = landing_x - hitter_x
    shot_dy = landing_y - hitter_y
    previous_dx = shot_rows["previous_landing_x"].to_numpy(float) - shot_rows["previous_hit_x"].to_numpy(float)
    previous_dy = shot_rows["previous_landing_y"].to_numpy(float) - shot_rows["previous_hit_y"].to_numpy(float)
    features = pd.DataFrame(
        {
            "hit_x": hitter_x,
            "hit_y": hitter_y,
            "landing_x": landing_x,
            "landing_y": landing_y,
            "receiver_x": shot_rows["opponent_x"].to_numpy(float),
            "receiver_y": shot_rows["opponent_y"].to_numpy(float),
            "shot_dx": shot_dx,
            "shot_dy": shot_dy,
            "shot_distance": np.hypot(shot_dx, shot_dy),
            "shot_lateral_distance": np.abs(shot_dx),
            "is_cross": shot_rows["is_cross"].astype("Float64").to_numpy(float, na_value=np.nan),
            "ball_round": shot_rows["ball_round"].to_numpy(float),
            "flight_seconds": seconds_between(shot_rows["next_frame_num"], shot_rows["frame_num"], shot_rows["fps"]),
            "seconds_since_previous_hit": seconds_between(
                shot_rows["frame_num"], shot_rows["previous_frame_num"], shot_rows["fps"]
            ),
            "previous_shot_distance": np.hypot(previous_dx, previous_dy),
            "previous_hit_y": shot_rows["previous_hit_y"].to_numpy(float),
        },
        index=shot_rows.index,
    )
    return features[list(FEATURE_NAMES)]
