import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vision.label import reviewed_rows_only
from vision.court import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M, video_id
from vision.players import PLAYERS, PLAYERS_DIR, boxes_path, require_video_file, video_fps_and_width
from vision.players import positions_path as player_positions_path

CHECKS_DIR = Path("data/checks/track_check")
OUTSIDE_MARGIN_M = 0.3
STILL_SPEED_M_PER_S = 0.3
STILL_SECONDS = 2.0
SPEED_SMOOTHING_FRAMES = 5
SIDE_EDGE_PX = 4
PASSER_HEIGHT_RATIO = 1.6
GAP_SECONDS = 0.5
HIT_WINDOW_SECONDS = 1.0
MOVING_SPEED_M_PER_S = 1.0
SHEET_FRAMES = 12
TILE_SIZE = (480, 270)
ERROR_TYPES = ("still_stand_in", "passer_by", "swap", "rally_gap")


def outside_lines(x_m: np.ndarray, y_m: np.ndarray, margin: float = OUTSIDE_MARGIN_M) -> np.ndarray:
    return (x_m < -margin) | (x_m > COURT_WIDTH_M + margin) | (y_m < -margin) | (y_m > COURT_LENGTH_M + margin)


def player_rows(positions: pd.DataFrame, boxes: pd.DataFrame, player: str) -> pd.DataFrame:
    rows = positions[positions["player"] == player][["frame", "x_m", "y_m"]].drop_duplicates("frame")
    box_rows = boxes[boxes["player"] == player][["frame", "left", "top", "right", "bottom"]].drop_duplicates("frame")
    return rows.merge(box_rows, on="frame", how="left").sort_values("frame").reset_index(drop=True)


def speeds(rows: pd.DataFrame, fps: float, smoothing: int = SPEED_SMOOTHING_FRAMES) -> np.ndarray:
    if len(rows) < 2:
        return np.zeros(len(rows))
    step = np.hypot(np.diff(rows["x_m"]), np.diff(rows["y_m"])) / (np.diff(rows["frame"]) / fps)
    raw = np.concatenate([[step[0]], step])
    return pd.Series(raw).rolling(smoothing, center=True, min_periods=1).median().to_numpy()


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate([[False], np.asarray(mask, dtype=bool), [False]]).astype(int)))
    return [(int(start), int(end) - 1) for start, end in zip(edges[::2], edges[1::2])]


def still_stand_in_runs(rows: pd.DataFrame, fps: float) -> list[tuple[int, int]]:
    mask = outside_lines(rows["x_m"].to_numpy(), rows["y_m"].to_numpy()) & (speeds(rows, fps) < STILL_SPEED_M_PER_S)
    shortest = int(round(STILL_SECONDS * fps))
    return [(int(rows["frame"].iloc[start]), int(rows["frame"].iloc[end])) for start, end in runs(mask) if rows["frame"].iloc[end] - rows["frame"].iloc[start] + 1 >= shortest]


def at_side_edge(rows: pd.DataFrame, frame_width: float) -> pd.Series:
    return (rows["left"] <= SIDE_EDGE_PX) | (rows["right"] >= frame_width - SIDE_EDGE_PX)


def passer_by_runs(rows: pd.DataFrame, frame_width: float) -> list[tuple[int, int]]:
    if rows.empty:
        return []
    height = rows["bottom"] - rows["top"]
    mask = (at_side_edge(rows, frame_width) & (height > PASSER_HEIGHT_RATIO * height.median())).to_numpy()
    return [(int(rows["frame"].iloc[start]), int(rows["frame"].iloc[end])) for start, end in runs(mask)]


def swap_frames(positions: pd.DataFrame) -> list[int]:
    wide = positions.drop_duplicates(["frame", "player"]).pivot(index="frame", columns="player", values="y_m")
    if not set(PLAYERS) <= set(wide.columns):
        return []
    swapped = (wide[PLAYERS[0]] > NET_Y_M) & (wide[PLAYERS[1]] < NET_Y_M)
    return [int(frame) for frame in wide.index[swapped]]


def rally_gap_runs(positions: pd.DataFrame, rallies: pd.DataFrame, fps: float) -> list[tuple[str, int, int]]:
    shortest = int(round(GAP_SECONDS * fps))
    found = []
    for player in PLAYERS:
        tracked = set(positions.loc[positions["player"] == player, "frame"].astype(int))
        for start, end in zip(rallies["start_frame"].astype(int), rallies["end_frame"].astype(int)):
            frames = np.arange(start, end + 1)
            missing = np.array([frame not in tracked for frame in frames])
            found += [(player, int(frames[first]), int(frames[last])) for first, last in runs(missing) if last - first + 1 >= shortest]
    return found


def rally_frames(rallies: pd.DataFrame) -> int:
    return int((rallies["end_frame"] - rallies["start_frame"] + 1).sum()) if len(rallies) else 0


def label_agreement(positions: pd.DataFrame, boxes: pd.DataFrame, labels: pd.DataFrame, fps: float) -> dict:
    window = int(round(HIT_WINDOW_SECONDS * fps))
    counts = {"labelled": 0, "hitter_tracked": 0, "right_half": 0, "moving": 0, "right_half_and_moving": 0}
    for player in PLAYERS:
        rows = player_rows(positions, boxes, player)
        speed = pd.Series(speeds(rows, fps), index=rows["frame"].to_numpy())
        y_by_frame = dict(zip(rows["frame"], rows["y_m"]))
        for frame in labels.loc[labels["player"] == player, "frame_num"].astype(int):
            counts["labelled"] += 1
            if frame not in y_by_frame:
                continue
            counts["hitter_tracked"] += 1
            on_half = (y_by_frame[frame] < NET_Y_M) == (player == PLAYERS[0])
            nearby = speed[(speed.index >= frame - window) & (speed.index <= frame + window)]
            moving = bool(len(nearby)) and float(nearby.max()) >= MOVING_SPEED_M_PER_S
            counts["right_half"] += on_half
            counts["moving"] += moving
            counts["right_half_and_moving"] += on_half and moving
    return counts


def diagnosis(positions: pd.DataFrame, boxes: pd.DataFrame, rallies: pd.DataFrame | None, fps: float, frame_width: float) -> dict:
    result = {"still_stand_in": [], "passer_by": [], "swap": swap_frames(positions), "rally_gap": []}
    for player in PLAYERS:
        rows = player_rows(positions, boxes, player)
        result["still_stand_in"] += [(player, start, end) for start, end in still_stand_in_runs(rows, fps)]
        result["passer_by"] += [(player, start, end) for start, end in passer_by_runs(rows, frame_width)]
    if rallies is not None and len(rallies):
        result["rally_gap"] = rally_gap_runs(positions, rallies, fps)
    return result


def run_frames(found: list[tuple[str, int, int]]) -> int:
    return sum(end - start + 1 for _, start, end in found)


def frames_inside(found: list[tuple[str, int, int]], rallies: pd.DataFrame | None) -> int:
    if rallies is None or not len(rallies):
        return 0
    spans = list(zip(rallies["start_frame"].astype(int), rallies["end_frame"].astype(int)))
    return sum(max(0, min(end, last) - max(start, first) + 1) for _, start, end in found for first, last in spans)


def summary_lines(name: str, found: dict, rallies: pd.DataFrame | None, total_frames: int, fps: float, agreement: dict | None) -> list[str]:
    in_rallies = rally_frames(rallies) if rallies is not None else 0
    lines = [
        f"| {name} | still stand-ins (still for {STILL_SECONDS:g} s or more, over {OUTSIDE_MARGIN_M:g} m outside the lines) | {len(found['still_stand_in'])} runs, {run_frames(found['still_stand_in'])} player-frames ({run_frames(found['still_stand_in']) / fps / 60:.1f} min), {frames_inside(found['still_stand_in'], rallies)} inside rallies |",
        f"| {name} | passers-by (a box at the image's side edge over {PASSER_HEIGHT_RATIO:g} x the player's median box height) | {len(found['passer_by'])} runs, {run_frames(found['passer_by'])} player-frames, {frames_inside(found['passer_by'], rallies)} inside rallies |",
        f"| {name} | me/opp swapped (me on the far half and opp on the near half) | {len(found['swap'])} frames of {total_frames} ({len(found['swap']) / max(total_frames, 1):.1%}) |",
    ]
    if rallies is not None:
        lines.append(f"| {name} | gaps of {GAP_SECONDS:g} s or more inside rallies | {len(found['rally_gap'])} runs, {run_frames(found['rally_gap'])} player-frames of {2 * in_rallies} in {len(rallies)} rallies |")
    if agreement is not None:
        labelled = max(agreement["labelled"], 1)
        lines.append(
            f"| {name} | labelled hits: hitter tracked / on its half / moving within {HIT_WINDOW_SECONDS:g} s / both | "
            f"{agreement['hitter_tracked']} / {agreement['right_half']} / {agreement['moving']} / {agreement['right_half_and_moving']} of {agreement['labelled']} "
            f"({agreement['right_half_and_moving'] / labelled:.1%} both) |"
        )
    return lines


def sample_frames(found: dict, error_type: str, count: int = SHEET_FRAMES) -> list[tuple[int, str]]:
    if error_type == "swap":
        frames = [(frame, "") for frame in found["swap"]]
    else:
        frames = [((start + end) // 2, player) for player, start, end in found[error_type]]
    if len(frames) <= count:
        return frames
    picks = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return [frames[index] for index in picks]


def write_sheet(video_path: Path, positions: pd.DataFrame, boxes: pd.DataFrame, picks: list[tuple[int, str]], output_path: Path) -> Path | None:
    from vision.alignment import open_video, read_frame
    from vision.hits import HIT_COLUMNS, RALLY_COLUMNS
    from vision.overlay import draw_tracking_overlay, frame_state, put_text

    if not picks:
        return None
    capture = open_video(Path(video_path))
    tiles = []
    try:
        for frame_index, player in picks:
            frame = read_frame(capture, frame_index)
            if frame is None:
                continue
            state = frame_state(frame_index, positions, boxes, pd.DataFrame(columns=HIT_COLUMNS), pd.DataFrame(columns=RALLY_COLUMNS))
            tile = cv2.resize(draw_tracking_overlay(frame, state), TILE_SIZE)
            put_text(tile, f"f{frame_index} {player}".strip(), (8, TILE_SIZE[1] - 10), 0.5)
            tiles.append(tile)
    finally:
        capture.release()
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack([np.hstack(tiles[start : start + 3]) for start in range(0, len(tiles), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return output_path


def sheet_path(video_path: Path, error_type: str, tag: str = "", checks_dir: Path = CHECKS_DIR) -> Path:
    return Path(checks_dir) / f"{video_id(video_path)}_{error_type}{'_' + tag if tag else ''}.jpg"


def rallies_for(labels_file: Path | None, rallies_file: Path | None) -> pd.DataFrame | None:
    if rallies_file is not None:
        return pd.read_csv(rallies_file)
    if labels_file is not None:
        own = Path(labels_file).with_name(Path(labels_file).stem + "_rallies.csv")
        return pd.read_csv(own) if own.is_file() else None
    return None


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vision.track_check",
        description="Count player-tracking errors (still stand-ins, passers-by, me/opp swaps, gaps inside rallies) and check the tracks against hand labels.",
    )
    parser.add_argument("video", type=Path, help="path to the video")
    parser.add_argument("--start", type=int, default=0, help="first frame of the tracked range (default 0)")
    parser.add_argument("--end", type=int, required=True, help="last frame of the tracked range")
    parser.add_argument("--labels", type=Path, help="hand labels CSV: hits for the agreement check, and its _rallies.csv for the gaps")
    parser.add_argument("--rallies", type=Path, help="rallies CSV (start_frame, end_frame) for the gaps, instead of the labels' rallies")
    parser.add_argument("--players-dir", type=Path, default=PLAYERS_DIR, help="root of the tracking files (default data/players)")
    parser.add_argument("--sheets", action="store_true", help="also draw a sample sheet per error type to data/checks/track_check/")
    parser.add_argument("--tag", default="", help="suffix for the sheet names, e.g. before / after")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        require_video_file(parsed.video)
        positions_file = player_positions_path(parsed.video, parsed.start, parsed.end, parsed.players_dir)
        boxes_file = boxes_path(parsed.video, parsed.start, parsed.end, parsed.players_dir)
        for path in (positions_file, boxes_file):
            if not path.is_file():
                raise FileNotFoundError(f"track file not found: {path}")
        positions, boxes = pd.read_csv(positions_file), pd.read_csv(boxes_file)
        fps, frame_width = video_fps_and_width(parsed.video)
        rallies = rallies_for(parsed.labels, parsed.rallies)
        labels = reviewed_rows_only(pd.read_csv(parsed.labels), parsed.labels) if parsed.labels else None
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    found = diagnosis(positions, boxes, rallies, fps, frame_width)
    agreement = label_agreement(positions, boxes, labels, fps) if labels is not None else None
    print("| Video | Check | Count |")
    print("|---|---|---|")
    for line in summary_lines(video_id(parsed.video), found, rallies, parsed.end - parsed.start + 1, fps, agreement):
        print(line)
    if parsed.sheets:
        for error_type in ERROR_TYPES:
            written = write_sheet(parsed.video, positions, boxes, sample_frames(found, error_type), sheet_path(parsed.video, error_type, parsed.tag))
            if written:
                print(f"saved {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
