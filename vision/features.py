import argparse
import sys
from pathlib import Path

import cv2
import duckdb
import numpy as np
import pandas as pd

from analytics.build_shots import create_shots_table, schema_columns
from analytics.court import COURT_LENGTH_M, COURT_WIDTH_M, shot_geometry
from vision.alignment import SHUTTLESET_SET_DIR, label_scale, load_set_shots, open_video
from vision.court import calibration_path, court_to_pixels, load_calibration, video_id
from vision.hits import HITS_DIR, frames_for_fps, output_paths, video_fps_and_width
from vision.players import PLAYERS, PLAYERS_DIR, effective_fps, frame_stride_of, is_default_dir, require_video_file
from vision.players import positions_path as player_positions_path

DATABASE_PATH = Path("data/scout.duckdb")
CHECKS_DIR = Path("data/checks/features")
SOURCE_OWN = "own"
HIT_POINT_SOURCES = ("feet",)
SPOT_CHECK_ROWS = 20
ORDERED_SHEET_ROWS = 10
SPOT_CHECK_SEED = 1
LABEL_VS_FEET_ROWS = 6
TILE_WIDTH, TILE_HEIGHT = 440, 280
HIT_COLOR_BGR, OPPONENT_COLOR_BGR, LANDING_COLOR_BGR, LABEL_COLOR_BGR, TEXT_COLOR_BGR = (0, 255, 255), (255, 160, 0), (0, 0, 255), (0, 0, 255), (0, 255, 255)
FEET_MAX_FRAME_GAP = 3
POINT_NAMES = ("hit", "landing", "player", "opponent")
DELETE_OWN_MATCH_SQL = "DELETE FROM shots WHERE shots.source = 'own' AND shots.match_id = ?"
DELETE_OWN_MATCH_SET_SQL = "DELETE FROM shots WHERE shots.source = 'own' AND shots.match_id = ? AND shots.set_no = ?"
OWN_MATCH_COUNT_SQL = "SELECT COUNT(*) FROM shots WHERE shots.source = 'own' AND shots.match_id = ?"
SHUTTLESET_MATCH_SHOTS_SQL = """
SELECT
    shots.set_no,
    shots.rally,
    shots.ball_round,
    shots.frame_num,
    shots.player,
    shots.roundscore_me,
    shots.roundscore_opp,
    shots.type_group,
    shots.hit_x,
    shots.hit_y,
    shots.landing_x,
    shots.landing_y,
    shots.player_x,
    shots.player_y,
    shots.hit_zone,
    shots.landing_zone,
    shots.is_cross
FROM shots
WHERE shots.source = 'shuttleset' AND shots.match_id = ?
ORDER BY shots.set_no, shots.frame_num
"""
SHUTTLESET_DOWNCOURT_SQL = "SELECT matches.downcourt FROM matches WHERE matches.match_id = ?"
END_SWITCH_SCORE = 11
OWN_MATCH_SHOTS_SQL = """
SELECT
    shots.set_no,
    shots.rally,
    shots.ball_round,
    shots.frame_num,
    shots.player,
    shots.type_pred,
    shots.type_conf,
    shots.hit_x,
    shots.hit_y,
    shots.landing_x,
    shots.landing_y,
    shots.player_x,
    shots.player_y,
    shots.hit_zone,
    shots.landing_zone,
    shots.is_cross
FROM shots
WHERE shots.source = 'own' AND shots.match_id = ?
ORDER BY shots.set_no, shots.frame_num
"""


def video_fps(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    try:
        return float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()


def feet_by_frame(positions: pd.DataFrame) -> dict[str, dict[int, tuple[float, float]]]:
    return {
        player: {int(row.frame): (float(row.x_m), float(row.y_m)) for row in player_rows.itertuples()}
        for player, player_rows in positions.groupby("player")
    }


def feet_at(feet: dict[str, dict[int, tuple[float, float]]], player: str, frame: int, max_gap: int = FEET_MAX_FRAME_GAP) -> tuple[float, float] | None:
    player_feet = feet.get(player, {})
    nearest = min((abs(known - frame), known) for known in player_feet) if player_feet else None
    if nearest is None or nearest[0] > max_gap:
        return None
    return player_feet[nearest[1]]


def other_player(player: str) -> str:
    return PLAYERS[1] if player == PLAYERS[0] else PLAYERS[0]


def point_columns(name: str, point: tuple[float, float] | None) -> dict:
    return {f"{name}_x": None if point is None else point[0], f"{name}_y": None if point is None else point[1]}


def shot_row(
    match_id: str,
    set_no: int,
    rally: int,
    ball_round: int,
    frame: int,
    fps: float,
    hitter: str,
    hit: tuple[float, float] | None,
    landing: tuple[float, float] | None,
    opponent: tuple[float, float] | None,
    left_handed: frozenset[str],
    feet: tuple[float, float] | None = None,
) -> dict:
    geometry = shot_geometry(
        hit,
        landing,
        hit if feet is None else feet,
        opponent,
        hitter_left_handed=hitter in left_handed,
        receiver_left_handed=other_player(hitter) in left_handed,
    )
    return {
        "source": SOURCE_OWN,
        "match_id": match_id,
        "set_no": set_no,
        "rally": rally,
        "ball_round": ball_round,
        "frame_num": frame,
        "fps": fps,
        "player": hitter,
        "type_en": None,
        "type_detail": None,
        "type_group": None,
        "type_detail_pred": None,
        "type_pred": None,
        "type_conf": None,
        "backhand": None,
        **{column: value for name in POINT_NAMES for column, value in point_columns(name, geometry[name]).items()},
        "hit_zone": geometry["hit_zone"],
        "landing_zone": geometry["landing_zone"],
        "is_cross": geometry["is_cross"],
        "roundscore_me": None,
        "roundscore_opp": None,
        "getpoint_player": None,
        "lose_reason": None,
    }


def rally_shot_rows(
    rally_hits: pd.DataFrame,
    feet: dict,
    match_id: str,
    set_no: int,
    rally: int,
    fps: float,
    left_handed: frozenset[str],
    feet_max_gap: int = FEET_MAX_FRAME_GAP,
) -> list[dict]:
    ordered = rally_hits.sort_values("frame")
    hit_points = [feet_at(feet, hit.hitter, int(hit.frame), feet_max_gap) for hit in ordered.itertuples()]
    rows = []
    for ball_round, (hit, hit_point) in enumerate(zip(ordered.itertuples(), hit_points), start=1):
        landing = hit_points[ball_round] if ball_round < len(hit_points) else None
        opponent = feet_at(feet, other_player(hit.hitter), int(hit.frame), feet_max_gap)
        rows.append(shot_row(match_id, set_no, rally, ball_round, int(hit.frame), fps, hit.hitter, hit_point, landing, opponent, left_handed))
    return rows


def effective_frame_numbers(table: pd.DataFrame, frame_stride: int) -> pd.DataFrame:
    if frame_stride == 1:
        return table
    return table.assign(frame=table["frame"] // frame_stride)


def own_shot_rows(
    hits: pd.DataFrame,
    positions: pd.DataFrame,
    match_id: str,
    set_no: int,
    fps: float,
    first_rally: int = 1,
    left_handed: frozenset[str] = frozenset(),
    frame_stride: int = 1,
) -> list[dict]:
    hits = effective_frame_numbers(hits, frame_stride)
    feet = feet_by_frame(effective_frame_numbers(positions, frame_stride))
    feet_max_gap = frames_for_fps(FEET_MAX_FRAME_GAP, fps)
    rows = []
    for rally_offset, (_, rally_hits) in enumerate(hits.groupby("rally")):
        rows.extend(rally_shot_rows(rally_hits, feet, match_id, set_no, first_rally + rally_offset, fps, left_handed, feet_max_gap))
    return rows


def pandas_dtype_for(duckdb_type: str) -> str:
    if duckdb_type == "INTEGER":
        return "Int64"
    if duckdb_type == "DOUBLE":
        return "float64"
    if duckdb_type == "BOOLEAN":
        return "boolean"
    return "object"


def schema_frame(rows: list[dict]) -> pd.DataFrame:
    columns = schema_columns()
    frame = pd.DataFrame(rows, columns=[name for name, _ in columns])
    for name, duckdb_type in columns:
        frame[name] = frame[name].astype(pandas_dtype_for(duckdb_type))
    return frame


def write_own_shots(database_path: Path, match_id: str, shots_frame: pd.DataFrame, replace_sets=None) -> int:
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute("BEGIN TRANSACTION")
        create_shots_table(connection)
        if replace_sets is None:
            connection.execute(DELETE_OWN_MATCH_SQL, [match_id])
        else:
            for set_no in sorted(set(int(set_no) for set_no in replace_sets)):
                connection.execute(DELETE_OWN_MATCH_SET_SQL, [match_id, set_no])
        connection.register("own_shot_rows", shots_frame)
        connection.execute(f"INSERT INTO shots ({', '.join(shots_frame.columns)}) SELECT * FROM own_shot_rows")
        connection.unregister("own_shot_rows")
        written = connection.execute(OWN_MATCH_COUNT_SQL, [match_id]).fetchone()[0]
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return int(written)


def shuttleset_player_side(player: str, set_no: int, roundscore_a, roundscore_b, downcourt: int) -> str:
    a_starts_near = int(downcourt) == 0
    a_near_this_set = a_starts_near if set_no in (1, 3) else not a_starts_near
    switched_at_eleven = set_no == 3 and max(float(roundscore_a or 0), float(roundscore_b or 0)) >= END_SWITCH_SCORE
    a_is_near = a_near_this_set != switched_at_eleven
    player_is_near = a_is_near if player == "A" else not a_is_near
    return PLAYERS[0] if player_is_near else PLAYERS[1]


def with_labelled_sides(labelled: pd.DataFrame, downcourt: int) -> pd.DataFrame:
    sides = [
        shuttleset_player_side(row.player, int(row.set_no), row.roundscore_me, row.roundscore_opp, downcourt)
        for row in labelled.itertuples()
    ]
    return labelled.assign(side=sides)


def matched_shot_pairs(own: pd.DataFrame, labelled: pd.DataFrame, tolerance: int = 5) -> pd.DataFrame:
    pairs = []
    for set_no, own_set in own.groupby("set_no"):
        labelled_set = labelled[labelled["set_no"] == set_no]
        candidates = sorted(
            (abs(int(own_row.frame_num) - int(label.frame_num)), own_index, label_index)
            for own_index, own_row in own_set.iterrows()
            for label_index, label in labelled_set.iterrows()
            if abs(int(own_row.frame_num) - int(label.frame_num)) <= tolerance
        )
        used_own, used_labels = set(), set()
        for _, own_index, label_index in candidates:
            if own_index not in used_own and label_index not in used_labels:
                used_own.add(own_index)
                used_labels.add(label_index)
                pairs.append((own_index, label_index))
    own_rows = own.loc[[own_index for own_index, _ in pairs]].reset_index(drop=True).add_prefix("own_")
    label_rows = labelled.loc[[label_index for _, label_index in pairs]].reset_index(drop=True).add_prefix("label_")
    return pd.concat([own_rows, label_rows], axis=1)


def point_errors(pairs: pd.DataFrame, name: str) -> pd.Series:
    return np.hypot(pairs[f"own_{name}_x"] - pairs[f"label_{name}_x"], pairs[f"own_{name}_y"] - pairs[f"label_{name}_y"])


def comparison_summary(pairs: pd.DataFrame, own_count: int, labelled_count: int) -> dict:
    with_prediction = pairs[pairs["own_type_pred"].notna() & pairs["label_type_group"].notna() & (pairs["label_type_group"] != "unknown")]
    hit_errors = point_errors(pairs, "hit").dropna()
    player_errors = point_errors(pairs, "player").dropna()
    landing_errors = point_errors(pairs, "landing").dropna()
    return {
        "own_shots": own_count,
        "labelled_shots": labelled_count,
        "matched_pairs": len(pairs),
        "hitter_agreement": float((pairs["own_player"] == pairs["label_side"]).mean()) if len(pairs) else None,
        "type_pred_accuracy": float((with_prediction["own_type_pred"] == with_prediction["label_type_group"]).mean()) if len(with_prediction) else None,
        "type_pred_judged": len(with_prediction),
        "hit_point_median_error_m": float(hit_errors.median()) if len(hit_errors) else None,
        "hit_point_mean_error_m": float(hit_errors.mean()) if len(hit_errors) else None,
        "player_point_median_error_m": float(player_errors.median()) if len(player_errors) else None,
        "player_point_mean_error_m": float(player_errors.mean()) if len(player_errors) else None,
        "landing_point_median_error_m": float(landing_errors.median()) if len(landing_errors) else None,
        "landing_point_mean_error_m": float(landing_errors.mean()) if len(landing_errors) else None,
        "hit_zone_agreement": float((pairs["own_hit_zone"] == pairs["label_hit_zone"]).mean()) if len(pairs) else None,
        "landing_zone_agreement": float((pairs["own_landing_zone"] == pairs["label_landing_zone"])[pairs["label_landing_zone"].notna()].mean()) if pairs["label_landing_zone"].notna().any() else None,
        "is_cross_agreement": float((pairs["own_is_cross"] == pairs["label_is_cross"])[pairs["label_is_cross"].notna() & pairs["own_is_cross"].notna()].mean()) if (pairs["label_is_cross"].notna() & pairs["own_is_cross"].notna()).any() else None,
    }


def confusion(pairs: pd.DataFrame) -> pd.DataFrame:
    judged = pairs[pairs["own_type_pred"].notna() & pairs["label_type_group"].notna()]
    return pd.crosstab(judged["label_type_group"], judged["own_type_pred"], rownames=["labelled"], colnames=["predicted"])


def compare_with_shuttleset(database_path: Path, own_match_id: str, shuttleset_match_id: str, tolerance: int = 5) -> tuple[dict, pd.DataFrame]:
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        own = connection.execute(OWN_MATCH_SHOTS_SQL, [own_match_id]).df()
        labelled = connection.execute(SHUTTLESET_MATCH_SHOTS_SQL, [shuttleset_match_id]).df()
        downcourt = connection.execute(SHUTTLESET_DOWNCOURT_SQL, [shuttleset_match_id]).fetchone()[0]
    finally:
        connection.close()
    pairs = matched_shot_pairs(own, with_labelled_sides(labelled, downcourt), tolerance)
    return comparison_summary(pairs, len(own), len(labelled)), pairs


def load_range_tracks(
    video_path: Path, start_frame: int, end_frame: int, players_dir: Path = PLAYERS_DIR, hits_dir: Path = HITS_DIR
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hits_file, rallies_file = output_paths(video_path, start_frame, end_frame, *(() if is_default_dir(hits_dir, HITS_DIR) else (hits_dir,)))
    files = {
        "hits": (hits_file, f"python -m vision.hits {video_path} --start {start_frame} --end {end_frame}"),
        "rallies": (rallies_file, f"python -m vision.hits {video_path} --start {start_frame} --end {end_frame}"),
        "positions": (positions_file_for(video_path, start_frame, end_frame, players_dir), f"python -m vision.players {video_path} --start {start_frame} --end {end_frame}"),
    }
    for path, command in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"track file not found: {path} (run: {command})")
    return tuple(pd.read_csv(path) for path, _ in files.values())


def positions_file_for(video_path: Path, start_frame: int, end_frame: int, players_dir: Path = PLAYERS_DIR) -> Path:
    if is_default_dir(players_dir, PLAYERS_DIR):
        return player_positions_path(video_path, start_frame, end_frame)
    return player_positions_path(video_path, start_frame, end_frame, players_dir)


def range_frame_stride(video_path: Path, start_frame: int, end_frame: int, players_dir: Path = PLAYERS_DIR) -> int:
    return frame_stride_of(positions_file_for(video_path, start_frame, end_frame, players_dir))


def build_own_match(
    video_path: Path,
    match_id: str,
    ranges: list[tuple[int, int, int]],
    database_path: Path = DATABASE_PATH,
    left_handed: frozenset[str] = frozenset(),
    hit_point: str = "feet",
    keep_other_sets: bool = False,
    players_dir: Path = PLAYERS_DIR,
    hits_dir: Path = HITS_DIR,
) -> pd.DataFrame:
    if hit_point not in HIT_POINT_SOURCES:
        raise ValueError(f"hit point must be one of {HIT_POINT_SOURCES}, got {hit_point!r}")
    require_video_file(video_path)
    video_fps, _ = video_fps_and_width(video_path)
    next_rally_by_set: dict[int, int] = {}
    rows = []
    for start_frame, end_frame, set_no in ranges:
        hits, _, positions = load_range_tracks(video_path, start_frame, end_frame, players_dir, hits_dir)
        frame_stride = range_frame_stride(video_path, start_frame, end_frame, players_dir)
        fps = effective_fps(video_fps, frame_stride)
        first_rally = next_rally_by_set.get(set_no, 1)
        rows.extend(own_shot_rows(hits, positions, match_id, set_no, fps, first_rally, left_handed, frame_stride))
        next_rally_by_set[set_no] = first_rally + hits["rally"].nunique()
    shots_frame = schema_frame(rows)
    replace_sets = sorted({set_no for _, _, set_no in ranges}) if keep_other_sets else None
    write_own_shots(database_path, match_id, shots_frame, replace_sets)
    return shots_frame


def rally_count(shots_frame: pd.DataFrame) -> int:
    return int(shots_frame[["set_no", "rally"]].drop_duplicates().shape[0])


def not_over_net_breakdown(own: pd.DataFrame) -> dict:
    ordered = own.sort_values(["set_no", "rally", "ball_round"])
    next_player = ordered.groupby(["set_no", "rally"])["player"].shift(-1)
    not_over_net = ordered["landing_zone"] == "not_over_net"
    followed_by_same_player = not_over_net & (next_player == ordered["player"])
    return {
        "rows": len(own),
        "not_over_net": int(not_over_net.sum()),
        "followed_by_same_player": int(followed_by_same_player.sum()),
        "rally_end_rest_point": int((not_over_net & next_player.isna()).sum()),
        "other": int((not_over_net & ~followed_by_same_player & next_player.notna()).sum()),
    }


def sample_tally(sample: pd.DataFrame) -> dict:
    has_label = sample["label_type_group"].notna()
    has_prediction = sample["type_pred"].notna()
    judged = has_label & has_prediction
    return {
        "rows": len(sample),
        "right": int((judged & (sample["type_pred"] == sample["label_type_group"])).sum()),
        "wrong": int((judged & (sample["type_pred"] != sample["label_type_group"])).sum()),
        "without_label": int((~has_label & has_prediction).sum()),
        "without_prediction": int((~has_prediction).sum()),
    }


def unnormalised_point(x: float, y: float, player: str) -> tuple[float, float] | None:
    if pd.isna(x) or pd.isna(y):
        return None
    return (COURT_WIDTH_M - float(x), COURT_LENGTH_M - float(y)) if player == PLAYERS[1] else (float(x), float(y))


def crop_around(frame: np.ndarray, centre: tuple[int, int], caption: str) -> np.ndarray:
    height, width = frame.shape[:2]
    left = int(np.clip(centre[0] - TILE_WIDTH // 2, 0, max(width - TILE_WIDTH, 0)))
    top = int(np.clip(centre[1] - TILE_HEIGHT * 5 // 7, 0, max(height - TILE_HEIGHT, 0)))
    tile = frame[top : top + TILE_HEIGHT, left : left + TILE_WIDTH].copy()
    cv2.putText(tile, caption, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR_BGR, 2)
    return tile


def draw_point(frame: np.ndarray, pixel, color: tuple[int, int, int], label: str) -> tuple[int, int]:
    x, y = int(round(pixel[0])), int(round(pixel[1]))
    cv2.circle(frame, (x, y), 9, color, 2)
    cv2.putText(frame, label, (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return x, y


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    frame_was_read, frame = capture.read()
    if not frame_was_read:
        raise ValueError(f"could not read frame {frame_index}")
    return frame


def own_rows_with_labels(database_path: Path, match_id: str, shuttleset_match_id: str | None, tolerance: int = 5) -> pd.DataFrame:
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        own = connection.execute("SELECT * FROM shots WHERE shots.source = 'own' AND shots.match_id = ? ORDER BY shots.set_no, shots.rally, shots.ball_round", [match_id]).df()
        if shuttleset_match_id is None:
            return own.assign(label_type_group=None, label_player=None)
        labelled = connection.execute(SHUTTLESET_MATCH_SHOTS_SQL, [shuttleset_match_id]).df()
    finally:
        connection.close()
    pairs = matched_shot_pairs(own[["set_no", "frame_num"]], labelled, tolerance)
    label_by_frame = {(int(pair.own_set_no), int(pair.own_frame_num)): (pair.label_type_group, pair.label_player) for pair in pairs.itertuples()}
    labels = [label_by_frame.get((int(row.set_no), int(row.frame_num)), (None, None)) for row in own.itertuples()]
    return own.assign(label_type_group=[group for group, _ in labels], label_player=[player for _, player in labels])


def spot_check_sample(own: pd.DataFrame, row_count: int = SPOT_CHECK_ROWS, seed: int = SPOT_CHECK_SEED) -> pd.DataFrame:
    picks = np.random.default_rng(seed).choice(len(own), size=min(row_count, len(own)), replace=False)
    return own.iloc[sorted(picks)]


def direction_word(is_cross) -> str:
    if pd.isna(is_cross):
        return "no direction"
    return "cross" if bool(is_cross) else "straight"


def frame_centre(frame: np.ndarray) -> tuple[int, int]:
    return frame.shape[1] // 2, frame.shape[0] // 2


def frame_stride_of_row(row, video_fps: float) -> int:
    if not video_fps or pd.isna(row.fps) or not row.fps:
        return 1
    return max(1, int(round(video_fps / float(row.fps))))


def video_frame_of_row(row, video_fps: float) -> int:
    return int(row.frame_num) * frame_stride_of_row(row, video_fps)


def is_inferred_serve(row, serve_frames: set[int], video_fps: float) -> bool:
    frame_stride = frame_stride_of_row(row, video_fps)
    return int(row.frame_num) in {frame // frame_stride for frame in serve_frames}


def spot_check_tile_pair(capture: cv2.VideoCapture, homography: np.ndarray, row, next_row, inferred_serve: bool = False) -> np.ndarray:
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_index = video_frame_of_row(row, video_fps)
    frame = read_frame(capture, frame_index)
    hit_point = unnormalised_point(row.hit_x, row.hit_y, row.player)
    centre = draw_point(frame, court_to_pixels(homography, [hit_point])[0], HIT_COLOR_BGR, "hit") if hit_point is not None else frame_centre(frame)
    opponent = unnormalised_point(row.opponent_x, row.opponent_y, row.player)
    if opponent is not None:
        draw_point(frame, court_to_pixels(homography, [opponent])[0], OPPONENT_COLOR_BGR, "opp")
    prediction = f"pred {row.type_pred} ({row.type_conf:.2f})" if pd.notna(row.type_pred) else "no pred"
    label = f"label {row.label_type_group}/{row.label_player}" if pd.notna(row.label_type_group) else "no label"
    kind = " inferred serve" if inferred_serve else ""
    left = crop_around(frame, centre, f"s{row.set_no} r{row.rally} #{row.ball_round}{kind} f{frame_index} {row.player} {prediction} {label}")
    geometry = f"{row.hit_zone} -> {row.landing_zone} {direction_word(row.is_cross)}"
    if next_row is None:
        right = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
        landing = "set" if pd.notna(row.landing_x) else "NULL"
        cv2.putText(right, f"last shot: landing {landing}; {geometry}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR_BGR, 2)
    else:
        next_frame_index = video_frame_of_row(next_row, video_fps)
        next_frame = read_frame(capture, next_frame_index)
        landing_point = unnormalised_point(next_row.hit_x, next_row.hit_y, next_row.player)
        landing_centre = draw_point(next_frame, court_to_pixels(homography, [landing_point])[0], LANDING_COLOR_BGR, "landing") if landing_point is not None else frame_centre(next_frame)
        right = crop_around(next_frame, landing_centre, f"next hit f{next_frame_index} by {next_row.player}: {geometry}")
    return np.hstack([left, right])


def inferred_serve_frames(video_path: Path, hits_dir: Path = HITS_DIR) -> set[int]:
    folder = Path(hits_dir) / video_id(video_path)
    frames: set[int] = set()
    for hits_file in sorted(folder.glob("hits_*.csv")):
        hits = pd.read_csv(hits_file)
        if "score" in hits.columns:
            frames.update(int(frame) for frame in hits.loc[hits["score"].isna(), "frame"])
    return frames


def first_rows(own: pd.DataFrame, row_count: int) -> pd.DataFrame:
    return own.sort_values(["set_no", "rally", "ball_round"]).head(row_count)


def ordered_sheet_path(match_id: str, row_count: int, sheet_index: int, checks_dir: Path = CHECKS_DIR) -> Path:
    return checks_dir / f"{match_id}_first{row_count}_{sheet_index}.jpg"


def write_ordered_spot_check_sheets(
    video_path: Path, homography: np.ndarray, own: pd.DataFrame, row_count: int, match_id: str, serve_frames: set[int], checks_dir: Path = CHECKS_DIR, rows_per_sheet: int = ORDERED_SHEET_ROWS
) -> list[Path]:
    checks_dir.mkdir(parents=True, exist_ok=True)
    rows = first_rows(own, row_count)
    capture = open_video(video_path)
    try:
        tile_pairs = [spot_check_tile_pair(capture, homography, row, next_row_in_rally(own, row), is_inferred_serve(row, serve_frames, capture.get(cv2.CAP_PROP_FPS))) for row in rows.itertuples()]
    finally:
        capture.release()
    written = []
    for sheet_index, first in enumerate(range(0, len(tile_pairs), rows_per_sheet)):
        path = ordered_sheet_path(match_id, row_count, sheet_index, checks_dir)
        cv2.imwrite(str(path), np.vstack(tile_pairs[first : first + rows_per_sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(path)
    return written


def ordered_spot_check(video_path: Path, match_id: str, row_count: int, database_path: Path = DATABASE_PATH, shuttleset_match_id: str | None = None, checks_dir: Path = CHECKS_DIR, hits_dir: Path = HITS_DIR) -> list[Path]:
    require_video_file(video_path)
    homography = load_calibration(calibration_path(video_path))["pixel_to_court_homography"]
    own = own_rows_with_labels(database_path, match_id, shuttleset_match_id)
    if own.empty:
        raise ValueError(f"no own rows for match {match_id!r} in {database_path}")
    return write_ordered_spot_check_sheets(video_path, homography, own, row_count, match_id, inferred_serve_frames(video_path, hits_dir), checks_dir)


def next_row_in_rally(own: pd.DataFrame, row):
    following = own[(own["set_no"] == row.set_no) & (own["rally"] == row.rally) & (own["ball_round"] == row.ball_round + 1)]
    return None if following.empty else following.iloc[0]


def write_spot_check_sheets(video_path: Path, homography: np.ndarray, own: pd.DataFrame, sample: pd.DataFrame, match_id: str, checks_dir: Path = CHECKS_DIR) -> list[Path]:
    checks_dir.mkdir(parents=True, exist_ok=True)
    capture = open_video(video_path)
    try:
        tile_pairs = [spot_check_tile_pair(capture, homography, row, next_row_in_rally(own, row)) for row in sample.itertuples()]
    finally:
        capture.release()
    written = []
    for sheet_index, first in enumerate(range(0, len(tile_pairs), 5)):
        path = checks_dir / f"{match_id}_spot_check_{sheet_index}.jpg"
        cv2.imwrite(str(path), np.vstack(tile_pairs[first : first + 5]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(path)
    return written


def write_label_vs_feet_sheet(video_path: Path, homography: np.ndarray, own: pd.DataFrame, shuttleset_match_id: str, match_id: str, checks_dir: Path = CHECKS_DIR, row_count: int = LABEL_VS_FEET_ROWS) -> Path | None:
    labelled_rows = own[own["label_type_group"].notna()]
    if labelled_rows.empty:
        return None
    capture = open_video(video_path)
    scale = label_scale(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    tiles = []
    try:
        for row in spot_check_sample(labelled_rows, row_count, SPOT_CHECK_SEED).itertuples():
            shots = load_set_shots(SHUTTLESET_SET_DIR, shuttleset_match_id, int(row.set_no))
            nearby = shots[(shots["frame_num"] - row.frame_num).abs() <= 5].dropna(subset=["hit_x", "hit_y"])
            if nearby.empty:
                continue
            label = nearby.iloc[0]
            frame = read_frame(capture, int(row.frame_num))
            feet_pixel = court_to_pixels(homography, [unnormalised_point(row.player_x, row.player_y, row.player)])[0]
            label_pixel = draw_point(frame, (label.hit_x * scale, label.hit_y * scale), LABEL_COLOR_BGR, "label")
            feet = draw_point(frame, feet_pixel, HIT_COLOR_BGR, "feet")
            centre = ((label_pixel[0] + feet[0]) // 2, (label_pixel[1] + feet[1]) // 2)
            tiles.append(crop_around(frame, centre, f"f{row.frame_num} {row.player} label {row.label_type_group}"))
    finally:
        capture.release()
    if not tiles:
        return None
    while len(tiles) % 3:
        tiles.append(np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8))
    path = checks_dir / f"{match_id}_label_vs_feet.jpg"
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[first : first + 3]) for first in range(0, len(tiles), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def spot_check(video_path: Path, match_id: str, database_path: Path = DATABASE_PATH, shuttleset_match_id: str | None = None, checks_dir: Path = CHECKS_DIR) -> dict:
    require_video_file(video_path)
    homography = load_calibration(calibration_path(video_path))["pixel_to_court_homography"]
    own = own_rows_with_labels(database_path, match_id, shuttleset_match_id)
    if own.empty:
        raise ValueError(f"no own shots for match {match_id!r} in {database_path}")
    sample = spot_check_sample(own)
    sheets = write_spot_check_sheets(video_path, homography, own, sample, match_id, checks_dir)
    label_sheet = write_label_vs_feet_sheet(video_path, homography, own, shuttleset_match_id, match_id, checks_dir) if shuttleset_match_id else None
    return {
        "not_over_net": not_over_net_breakdown(own),
        "sample": sample_tally(sample),
        "sample_frames": [int(frame) for frame in sample["frame_num"]],
        "sheets": sheets + ([label_sheet] if label_sheet else []),
    }


def parse_range(text: list[str]) -> tuple[int, int, int]:
    start_frame, end_frame, set_no = (int(value) for value in text)
    if set_no not in (1, 2, 3):
        raise ValueError(f"set number must be 1, 2 or 3, got {set_no}")
    return start_frame, end_frame, set_no


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vision.features",
        description="Turn detected hits of tracked frame ranges into rows of the shots table (source = 'own').",
    )
    parser.add_argument("video", type=Path, help="path to the video (its calibration and track files must exist)")
    parser.add_argument("--match-id", required=True, help="recording id for the own rows, e.g. club_2026-11-02")
    parser.add_argument("--range", nargs=3, action="append", metavar=("START", "END", "SET"), default=[], help="a tracked frame range and its set number; repeatable (required unless --spot-check)")
    parser.add_argument("--left-handed", default="", help="comma-separated players who are left-handed: me, opp")
    parser.add_argument(
        "--keep-other-sets", action="store_true", help="replace only this match's rows of the sets in --range, keep its other sets (one video per set)"
    )
    parser.add_argument("--database", type=Path, default=DATABASE_PATH, help="DuckDB file (default data/scout.duckdb)")
    parser.add_argument("--players-dir", type=Path, default=PLAYERS_DIR, help="root of the player tracking files (default data/players)")
    parser.add_argument("--hits-dir", type=Path, default=HITS_DIR, help="root of the hit files (default data/hits)")
    parser.add_argument("--compare", metavar="SHUTTLESET_MATCH", help="after writing, compare with this ShuttleSet match's labelled rows")
    parser.add_argument("--spot-check", action="store_true", help="don't write rows; draw the spot-check sheets for the match's rows already in the database (with --compare, also label vs feet)")
    parser.add_argument("--spot-check-first", type=int, metavar="N", help="don't write rows; draw the match's first N rows in order (set, rally, ball_round), 10 per sheet, to data/checks/features/<match-id>_first<N>_<k>.jpg")
    parsed = parser.parse_args(arguments)
    if not parsed.range and not parsed.spot_check and parsed.spot_check_first is None:
        parser.error("give at least one --range START END SET, --spot-check or --spot-check-first N")
    return parsed


def run_spot_check(parsed: argparse.Namespace) -> int:
    result = spot_check(parsed.video, parsed.match_id, parsed.database, parsed.compare)
    print(f"rows: {result['not_over_net']['rows']}, landing_zone not_over_net: {result['not_over_net']['not_over_net']} "
          f"(followed by the same player: {result['not_over_net']['followed_by_same_player']}, rally-end rest point: {result['not_over_net']['rally_end_rest_point']}, other: {result['not_over_net']['other']})")
    tally = result["sample"]
    print(f"sample of {tally['rows']} rows (frames {result['sample_frames']}): predictions right {tally['right']}, wrong {tally['wrong']}, "
          f"without a label {tally['without_label']}, without a prediction {tally['without_prediction']}")
    for path in result["sheets"]:
        print(f"saved {path}")
    return 0


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        if parsed.spot_check_first is not None:
            for path in ordered_spot_check(parsed.video, parsed.match_id, parsed.spot_check_first, parsed.database, parsed.compare, hits_dir=parsed.hits_dir):
                print(f"saved {path}")
            return 0
        if parsed.spot_check:
            return run_spot_check(parsed)
        ranges = [parse_range(text) for text in parsed.range]
        left_handed = frozenset(player for player in parsed.left_handed.split(",") if player)
        shots_frame = build_own_match(
            parsed.video, parsed.match_id, ranges, parsed.database, left_handed, keep_other_sets=parsed.keep_other_sets,
            players_dir=parsed.players_dir, hits_dir=parsed.hits_dir,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"wrote {len(shots_frame)} own shots for match {parsed.match_id!r} "
        f"({rally_count(shots_frame)} rallies, {shots_frame['landing_x'].notna().sum()} with a landing point) to {parsed.database}"
    )
    if parsed.compare:
        summary, _ = compare_with_shuttleset(parsed.database, parsed.match_id, parsed.compare)
        for key, value in summary.items():
            print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
