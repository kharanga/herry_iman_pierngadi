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
from vision.court import calibration_path, court_to_pixels, load_calibration, pixels_to_court
from vision.hits import REST_FRAME_COUNT, is_at_rest, output_paths
from vision.players import PLAYERS, require_video_file
from vision.players import positions_path as player_positions_path
from vision.shuttle import positions_path as shuttle_positions_path

DATABASE_PATH = Path("data/scout.duckdb")
CHECKS_DIR = Path("data/checks/features")
SOURCE_OWN = "own"
SPOT_CHECK_ROWS = 20
SPOT_CHECK_SEED = 1
LABEL_VS_FEET_ROWS = 6
TILE_WIDTH, TILE_HEIGHT = 440, 280
HIT_COLOR_BGR, OPPONENT_COLOR_BGR, LANDING_COLOR_BGR, LABEL_COLOR_BGR, TEXT_COLOR_BGR = (0, 255, 255), (255, 160, 0), (0, 0, 255), (0, 0, 255), (0, 255, 255)
FEET_MAX_FRAME_GAP = 3
POINT_NAMES = ("hit", "landing", "player", "opponent")
DELETE_OWN_MATCH_SQL = "DELETE FROM shots WHERE shots.source = 'own' AND shots.match_id = ?"
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


def floor_point_at_rest(shuttle: pd.DataFrame, end_frame: int, homography: np.ndarray, rest_frame_count: int = REST_FRAME_COUNT) -> tuple[float, float] | None:
    window = shuttle[(shuttle["frame"] >= end_frame) & (shuttle["frame"] < end_frame + rest_frame_count)].sort_values("frame")
    if len(window) < rest_frame_count or not is_at_rest(window[["x", "y"]].to_numpy(float), window["visibility"].to_numpy() == 1):
        return None
    x_m, y_m = pixels_to_court(homography, [window[["x", "y"]].to_numpy(float)[0]])[0]
    return float(x_m), float(y_m)


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
) -> dict:
    geometry = shot_geometry(
        hit, landing, hit, opponent, hitter_left_handed=hitter in left_handed, receiver_left_handed=other_player(hitter) in left_handed
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
        "type_group": None,
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
    last_landing: tuple[float, float] | None,
    match_id: str,
    set_no: int,
    rally: int,
    fps: float,
    left_handed: frozenset[str],
) -> list[dict]:
    ordered = rally_hits.sort_values("frame")
    hit_points = [feet_at(feet, hit.hitter, int(hit.frame)) for hit in ordered.itertuples()]
    rows = []
    for ball_round, (hit, hit_point) in enumerate(zip(ordered.itertuples(), hit_points), start=1):
        landing = hit_points[ball_round] if ball_round < len(hit_points) else last_landing
        opponent = feet_at(feet, other_player(hit.hitter), int(hit.frame))
        rows.append(shot_row(match_id, set_no, rally, ball_round, int(hit.frame), fps, hit.hitter, hit_point, landing, opponent, left_handed))
    return rows


def own_shot_rows(
    hits: pd.DataFrame,
    rallies: pd.DataFrame,
    positions: pd.DataFrame,
    shuttle: pd.DataFrame,
    homography: np.ndarray,
    match_id: str,
    set_no: int,
    fps: float,
    first_rally: int = 1,
    left_handed: frozenset[str] = frozenset(),
) -> list[dict]:
    feet = feet_by_frame(positions)
    end_frame_by_rally = dict(zip(rallies["rally"], rallies["end_frame"]))
    rows = []
    for rally_offset, (rally_in_range, rally_hits) in enumerate(hits.groupby("rally")):
        last_landing = floor_point_at_rest(shuttle, int(end_frame_by_rally[rally_in_range]), homography)
        rows.extend(rally_shot_rows(rally_hits, feet, last_landing, match_id, set_no, first_rally + rally_offset, fps, left_handed))
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


def write_own_shots(database_path: Path, match_id: str, shots_frame: pd.DataFrame) -> int:
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute("BEGIN TRANSACTION")
        create_shots_table(connection)
        connection.execute(DELETE_OWN_MATCH_SQL, [match_id])
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


def load_range_tracks(video_path: Path, start_frame: int, end_frame: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hits_file, rallies_file = output_paths(video_path, start_frame, end_frame)
    files = {
        "hits": (hits_file, f"python -m vision.hits {video_path} --start {start_frame} --end {end_frame}"),
        "rallies": (rallies_file, f"python -m vision.hits {video_path} --start {start_frame} --end {end_frame}"),
        "positions": (player_positions_path(video_path, start_frame, end_frame), f"python -m vision.players {video_path} --start {start_frame} --end {end_frame}"),
        "shuttle": (shuttle_positions_path(video_path, start_frame, end_frame), f"python -m vision.shuttle {video_path} --start {start_frame} --end {end_frame}"),
    }
    for path, command in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"track file not found: {path} (run: {command})")
    return tuple(pd.read_csv(path) for path, _ in files.values())


def build_own_match(
    video_path: Path,
    match_id: str,
    ranges: list[tuple[int, int, int]],
    database_path: Path = DATABASE_PATH,
    left_handed: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    require_video_file(video_path)
    homography = load_calibration(calibration_path(video_path))["pixel_to_court_homography"]
    fps = video_fps(video_path)
    next_rally_by_set: dict[int, int] = {}
    rows = []
    for start_frame, end_frame, set_no in ranges:
        hits, rallies, positions, shuttle = load_range_tracks(video_path, start_frame, end_frame)
        first_rally = next_rally_by_set.get(set_no, 1)
        rows.extend(own_shot_rows(hits, rallies, positions, shuttle, homography, match_id, set_no, fps, first_rally, left_handed))
        next_rally_by_set[set_no] = first_rally + hits["rally"].nunique()
    shots_frame = schema_frame(rows)
    write_own_shots(database_path, match_id, shots_frame)
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


def spot_check_tile_pair(capture: cv2.VideoCapture, homography: np.ndarray, row, next_row) -> np.ndarray:
    frame = read_frame(capture, int(row.frame_num))
    hit_pixel = court_to_pixels(homography, [unnormalised_point(row.hit_x, row.hit_y, row.player)])[0]
    centre = draw_point(frame, hit_pixel, HIT_COLOR_BGR, "hit")
    opponent = unnormalised_point(row.opponent_x, row.opponent_y, row.player)
    if opponent is not None:
        draw_point(frame, court_to_pixels(homography, [opponent])[0], OPPONENT_COLOR_BGR, "opp")
    prediction = f"pred {row.type_pred} ({row.type_conf:.2f})" if pd.notna(row.type_pred) else "no pred"
    label = f"label {row.label_type_group}/{row.label_player}" if pd.notna(row.label_type_group) else "no label"
    left = crop_around(frame, centre, f"s{row.set_no} r{row.rally} #{row.ball_round} f{row.frame_num} {row.player} {prediction} {label}")
    geometry = f"{row.hit_zone} -> {row.landing_zone} cross={row.is_cross}"
    if next_row is None:
        right = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
        landing = "set" if pd.notna(row.landing_x) else "NULL"
        cv2.putText(right, f"last shot: landing {landing}; {geometry}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR_BGR, 2)
    else:
        next_frame = read_frame(capture, int(next_row.frame_num))
        landing_pixel = court_to_pixels(homography, [unnormalised_point(next_row.hit_x, next_row.hit_y, next_row.player)])[0]
        right = crop_around(next_frame, draw_point(next_frame, landing_pixel, LANDING_COLOR_BGR, "landing"), f"next hit f{next_row.frame_num} by {next_row.player}: {geometry}")
    return np.hstack([left, right])


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
            feet_pixel = court_to_pixels(homography, [unnormalised_point(row.hit_x, row.hit_y, row.player)])[0]
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
    parser.add_argument("--database", type=Path, default=DATABASE_PATH, help="DuckDB file (default data/scout.duckdb)")
    parser.add_argument("--compare", metavar="SHUTTLESET_MATCH", help="after writing, compare with this ShuttleSet match's labelled rows")
    parser.add_argument("--spot-check", action="store_true", help="don't write rows; draw the spot-check sheets for the match's rows already in the database (with --compare, also label vs feet)")
    parsed = parser.parse_args(arguments)
    if not parsed.range and not parsed.spot_check:
        parser.error("give at least one --range START END SET, or --spot-check")
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
        if parsed.spot_check:
            return run_spot_check(parsed)
        ranges = [parse_range(text) for text in parsed.range]
        left_handed = frozenset(player for player in parsed.left_handed.split(",") if player)
        shots_frame = build_own_match(parsed.video, parsed.match_id, ranges, parsed.database, left_handed)
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
