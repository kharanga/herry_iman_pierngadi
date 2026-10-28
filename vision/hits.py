import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vision.alignment import SHUTTLESET_SET_DIR, load_set_shots
from vision.court import NET_Y_M, calibration_path, load_calibration, pixels_to_court
from vision.players import PLAYERS, boxes_path, split_frames_by_view
from vision.players import positions_path as player_positions_path
from vision.shuttle import positions_path as shuttle_positions_path

VELOCITY_WINDOW_FRAMES = 3
SPEED_FLOOR_PX_PER_FRAME = 2.0
HIT_SCORE_MIN = 0.4
SPEED_AFTER_MIN_PX_PER_FRAME = 8.0
REACH_MAX_BODY_HEIGHTS = 0.75
MIN_HIT_GAP_FRAMES = 8
RALLY_GAP_FRAMES = 120
REST_RADIUS_PX = 8.0
REST_FRAME_COUNT = 20
MATCH_TOLERANCE_FRAMES = 5
HITS_DIR = Path("data/hits")
HIT_COLUMNS = ["rally", "frame", "hitter", "score", "reach"]
RALLY_COLUMNS = ["rally", "start_frame", "end_frame", "hits"]


@dataclass(frozen=True)
class HitDetectionSettings:
    velocity_window_frames: int = VELOCITY_WINDOW_FRAMES
    speed_floor_px_per_frame: float = SPEED_FLOOR_PX_PER_FRAME
    hit_score_min: float = HIT_SCORE_MIN
    speed_after_min_px_per_frame: float = SPEED_AFTER_MIN_PX_PER_FRAME
    reach_max_body_heights: float = REACH_MAX_BODY_HEIGHTS
    min_hit_gap_frames: int = MIN_HIT_GAP_FRAMES


@dataclass(frozen=True)
class FrameTracks:
    frames: np.ndarray
    shuttle_points: np.ndarray
    shuttle_valid: np.ndarray
    near_boxes: np.ndarray
    far_boxes: np.ndarray
    main_view: np.ndarray | None = None
    shuttle_inpainted: np.ndarray | None = None


def window_velocities(points, valid, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_array = np.asarray(points, dtype=float).reshape(-1, 2)
    valid_array = np.asarray(valid, dtype=bool)
    frame_count = len(point_array)
    before = np.zeros((frame_count, 2))
    after = np.zeros((frame_count, 2))
    defined = np.zeros(frame_count, dtype=bool)
    for index in range(window, frame_count - window):
        if valid_array[index - window] and valid_array[index] and valid_array[index + window]:
            before[index] = (point_array[index] - point_array[index - window]) / window
            after[index] = (point_array[index + window] - point_array[index]) / window
            defined[index] = True
    return before, after, defined


def hit_scores(before: np.ndarray, after: np.ndarray, defined: np.ndarray, speed_floor: float) -> np.ndarray:
    change = np.linalg.norm(after - before, axis=1)
    speeds = np.linalg.norm(before, axis=1) + np.linalg.norm(after, axis=1)
    return np.where(defined, change / (speeds + speed_floor), 0.0)


def distance_to_box(point, box) -> float:
    x, y = point
    left, top, right, bottom = box
    dx = max(left - x, 0.0, x - right)
    dy = max(top - y, 0.0, y - bottom)
    return float(np.hypot(dx, dy))


def reach_distances(points, boxes) -> np.ndarray:
    point_array = np.asarray(points, dtype=float).reshape(-1, 2)
    box_array = np.asarray(boxes, dtype=float).reshape(-1, 4)
    distances = np.full(len(point_array), np.inf)
    for index, (point, box) in enumerate(zip(point_array, box_array)):
        box_height = box[3] - box[1]
        if np.all(np.isfinite(box)) and box_height > 0:
            distances[index] = distance_to_box(point, box) / box_height
    return distances


def peaks_with_minimum_gap(candidate_indices, scores: np.ndarray, min_gap: int) -> list[int]:
    accepted: list[int] = []
    for index in sorted(candidate_indices, key=lambda candidate: -scores[candidate]):
        if all(abs(index - kept) >= min_gap for kept in accepted):
            accepted.append(int(index))
    return sorted(accepted)


def nearest_player(near_reach_now: float, far_reach_now: float) -> str:
    return PLAYERS[0] if near_reach_now <= far_reach_now else PLAYERS[1]


def detect_hits(tracks: FrameTracks, settings: HitDetectionSettings = HitDetectionSettings()) -> pd.DataFrame:
    before, after, defined = window_velocities(tracks.shuttle_points, tracks.shuttle_valid, settings.velocity_window_frames)
    scores = hit_scores(before, after, defined, settings.speed_floor_px_per_frame)
    speed_after = np.linalg.norm(after, axis=1)
    near_reach = reach_distances(tracks.shuttle_points, tracks.near_boxes)
    far_reach = reach_distances(tracks.shuttle_points, tracks.far_boxes)
    closest_reach = np.minimum(near_reach, far_reach)
    candidates = np.flatnonzero(
        (scores >= settings.hit_score_min)
        & (speed_after >= settings.speed_after_min_px_per_frame)
        & (closest_reach <= settings.reach_max_body_heights)
    )
    hit_indices = peaks_with_minimum_gap(candidates, scores, settings.min_hit_gap_frames)
    return pd.DataFrame(
        {
            "frame": tracks.frames[hit_indices].astype(int),
            "hitter": [nearest_player(near_reach[index], far_reach[index]) for index in hit_indices],
            "score": np.round(scores[hit_indices], 3),
            "reach": np.round(closest_reach[hit_indices], 3),
        },
        columns=["frame", "hitter", "score", "reach"],
    )


def is_at_rest(points, valid, rest_radius: float = REST_RADIUS_PX) -> bool:
    point_array = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(point_array) == 0 or not np.all(valid):
        return False
    return bool(np.all(np.linalg.norm(point_array - point_array[0], axis=1) <= rest_radius))


def shuttle_rest_frame(
    tracks: FrameTracks,
    from_frame: int,
    to_frame: int,
    rest_radius: float = REST_RADIUS_PX,
    rest_frame_count: int = REST_FRAME_COUNT,
) -> int | None:
    first = int(np.searchsorted(tracks.frames, from_frame))
    last = int(np.searchsorted(tracks.frames, to_frame, side="right"))
    for index in range(first + 1, last - rest_frame_count + 1):
        window = slice(index, index + rest_frame_count)
        if is_at_rest(tracks.shuttle_points[window], tracks.shuttle_valid[window], rest_radius):
            return int(tracks.frames[index])
    return None


def last_valid_frame_before(tracks: FrameTracks, from_frame: int, to_frame: int) -> int:
    first = int(np.searchsorted(tracks.frames, from_frame))
    last = int(np.searchsorted(tracks.frames, to_frame, side="right"))
    valid_indices = np.flatnonzero(tracks.shuttle_valid[first:last]) + first
    return int(tracks.frames[valid_indices[-1]]) if len(valid_indices) else int(from_frame)


def rally_numbers(hit_frames, tracks: FrameTracks, rally_gap_frames: int = RALLY_GAP_FRAMES) -> list[int]:
    numbers, rally = [], 0
    previous_frame = None
    for frame in hit_frames:
        long_gap = previous_frame is None or frame - previous_frame > rally_gap_frames
        rested = previous_frame is not None and shuttle_rest_frame(tracks, previous_frame, frame) is not None
        if long_gap or rested:
            rally += 1
        numbers.append(rally)
        previous_frame = frame
    return numbers


def rally_end_frame(last_hit_frame: int, next_rally_start: int, tracks: FrameTracks) -> int:
    rest_frame = shuttle_rest_frame(tracks, last_hit_frame, next_rally_start - 1)
    return rest_frame if rest_frame is not None else last_valid_frame_before(tracks, last_hit_frame, next_rally_start - 1)


def segment_rallies(hits: pd.DataFrame, tracks: FrameTracks, rally_gap_frames: int = RALLY_GAP_FRAMES) -> tuple[pd.DataFrame, pd.DataFrame]:
    numbered = hits.assign(rally=rally_numbers(hits["frame"].tolist(), tracks, rally_gap_frames))[HIT_COLUMNS]
    first_frames = numbered.groupby("rally")["frame"].min()
    next_starts = first_frames.shift(-1).fillna(int(tracks.frames[-1]) + 1 if len(tracks.frames) else 0).astype(int)
    rallies = [
        {
            "rally": rally,
            "start_frame": int(rally_hits["frame"].min()),
            "end_frame": rally_end_frame(int(rally_hits["frame"].max()), int(next_starts[rally]), tracks),
            "hits": len(rally_hits),
        }
        for rally, rally_hits in numbered.groupby("rally")
    ]
    return numbered, pd.DataFrame(rallies, columns=RALLY_COLUMNS)


def match_hits(detected_frames, labelled_frames, tolerance: int = MATCH_TOLERANCE_FRAMES) -> list[tuple[int, int]]:
    pairs = sorted(
        (abs(detected - labelled), detected, labelled)
        for detected in detected_frames
        for labelled in labelled_frames
        if abs(detected - labelled) <= tolerance
    )
    used_detected, used_labelled, matched = set(), set(), []
    for _, detected, labelled in pairs:
        if detected not in used_detected and labelled not in used_labelled:
            used_detected.add(detected)
            used_labelled.add(labelled)
            matched.append((detected, labelled))
    return sorted(matched, key=lambda pair: pair[1])


def precision_recall(matched_count: int, detected_count: int, labelled_count: int) -> tuple[float, float]:
    precision = matched_count / detected_count if detected_count else 0.0
    recall = matched_count / labelled_count if labelled_count else 0.0
    return precision, recall


def side_of_hitter(homography: np.ndarray, hitter_pixel) -> str:
    _, y_m = pixels_to_court(homography, [hitter_pixel])[0]
    return PLAYERS[0] if y_m < NET_Y_M else PLAYERS[1]


def labelled_hits(shots: pd.DataFrame, homography: np.ndarray, start_frame: int, end_frame: int, label_scale: float = 1.0) -> pd.DataFrame:
    in_range = shots[(shots["frame_num"] >= start_frame) & (shots["frame_num"] <= end_frame)].dropna(subset=["frame_num"])
    rows = []
    for shot in in_range.itertuples():
        hitter_pixel = (shot.player_location_x * label_scale, shot.player_location_y * label_scale)
        hitter = side_of_hitter(homography, hitter_pixel) if np.all(np.isfinite(hitter_pixel)) else None
        rows.append({"rally": int(shot.rally), "frame": int(shot.frame_num), "hitter": hitter, "type": shot.type})
    return pd.DataFrame(rows, columns=["rally", "frame", "hitter", "type"])


def boxes_by_frame(boxes: pd.DataFrame, frames: np.ndarray, player: str) -> np.ndarray:
    player_boxes = boxes[boxes["player"] == player].set_index("frame")[["left", "top", "right", "bottom"]]
    return player_boxes.reindex(frames).to_numpy(dtype=float)


def frame_tracks(shuttle: pd.DataFrame, boxes: pd.DataFrame, main_view_frames) -> FrameTracks:
    shuttle = shuttle.sort_values("frame")
    frames = shuttle["frame"].to_numpy(dtype=int)
    in_main_view = np.isin(frames, np.asarray(list(main_view_frames), dtype=int))
    return FrameTracks(
        frames=frames,
        shuttle_points=shuttle[["x", "y"]].to_numpy(dtype=float),
        shuttle_valid=(shuttle["visibility"].to_numpy() == 1) & in_main_view,
        near_boxes=boxes_by_frame(boxes, frames, PLAYERS[0]),
        far_boxes=boxes_by_frame(boxes, frames, PLAYERS[1]),
        main_view=in_main_view,
        shuttle_inpainted=shuttle["inpainted"].to_numpy() == 1,
    )


def require_track_file(path: Path, command: str) -> Path:
    if not Path(path).is_file():
        raise FileNotFoundError(f"track file not found: {path} (run: {command})")
    return Path(path)


def load_frame_tracks(video_path: Path, start_frame: int, end_frame: int) -> FrameTracks:
    range_arguments = f"{video_path} --start {start_frame} --end {end_frame}"
    shuttle_file = require_track_file(
        shuttle_positions_path(video_path, start_frame, end_frame), f"python -m vision.shuttle {range_arguments}"
    )
    boxes_file = require_track_file(boxes_path(video_path, start_frame, end_frame), f"python -m vision.players {range_arguments}")
    require_track_file(player_positions_path(video_path, start_frame, end_frame), f"python -m vision.players {range_arguments}")
    main_view_frames, _ = split_frames_by_view(video_path, start_frame, end_frame)
    return frame_tracks(pd.read_csv(shuttle_file), pd.read_csv(boxes_file), main_view_frames)


def output_paths(video_path: Path, start_frame: int, end_frame: int, hits_dir: Path = HITS_DIR) -> tuple[Path, Path]:
    folder = hits_dir / Path(video_path).stem
    return folder / f"hits_{start_frame}_{end_frame}.csv", folder / f"rallies_{start_frame}_{end_frame}.csv"


def evaluation_lines(hits: pd.DataFrame, labels: pd.DataFrame, tolerance: int = MATCH_TOLERANCE_FRAMES) -> list[str]:
    matched = match_hits(hits["frame"].tolist(), labels["frame"].tolist(), tolerance)
    precision, recall = precision_recall(len(matched), len(hits), len(labels))
    detected_hitter = dict(zip(hits["frame"], hits["hitter"]))
    labelled_hitter = dict(zip(labels["frame"], labels["hitter"]))
    judged = [
        (detected_hitter[detected], labelled_hitter[labelled])
        for detected, labelled in matched
        if pd.notna(labelled_hitter[labelled])
    ]
    correct_hitters = sum(detected == labelled for detected, labelled in judged)
    offsets = [detected - labelled for detected, labelled in matched]
    return [
        f"labelled hits: {len(labels)}, detected: {len(hits)}, matched within ±{tolerance} frames: {len(matched)}",
        f"precision {precision:.1%}, recall {recall:.1%}",
        f"hitter correct: {correct_hitters}/{len(judged)}" + (f" = {correct_hitters / len(judged):.1%}" if judged else ""),
        f"frame offset detected - labelled: mean {np.mean(offsets):+.2f}, median {np.median(offsets):+.1f}" if offsets else "frame offset: no matches",
    ]


def miss_cause(
    label_frame: int, is_first_in_rally: bool, tracks: FrameTracks, settings: HitDetectionSettings, tolerance: int
) -> str:
    prefix = "serve: " if is_first_in_rally else ""
    index_by_frame = {int(frame): index for index, frame in enumerate(tracks.frames)}
    if label_frame not in index_by_frame:
        return prefix + "frame not tracked"
    index = index_by_frame[label_frame]
    if tracks.main_view is not None and not tracks.main_view[index]:
        return prefix + "not main view"
    before, after, defined = window_velocities(tracks.shuttle_points, tracks.shuttle_valid, settings.velocity_window_frames)
    scores = hit_scores(before, after, defined, settings.speed_floor_px_per_frame)
    speed_after = np.linalg.norm(after, axis=1)
    closest_reach = np.minimum(
        reach_distances(tracks.shuttle_points, tracks.near_boxes), reach_distances(tracks.shuttle_points, tracks.far_boxes)
    )
    window = slice(max(index - tolerance, 0), index + tolerance + 1)
    if not defined[window].any():
        return prefix + "velocity window undefined"
    score_ok = scores[window] >= settings.hit_score_min
    speed_ok = speed_after[window] >= settings.speed_after_min_px_per_frame
    reach_ok = closest_reach[window] <= settings.reach_max_body_heights
    if not score_ok.any():
        return prefix + "score below threshold"
    if not (score_ok & speed_ok).any():
        return prefix + "speed after the hit below threshold"
    if not (score_ok & speed_ok & reach_ok).any():
        return prefix + "reach above limit"
    return prefix + "suppressed by a stronger candidate within the minimum gap"


def false_positive_cause(frame: int, labelled_rally_spans: list[tuple[int, int]], tracks: FrameTracks) -> str:
    if labelled_rally_spans and frame < labelled_rally_spans[0][0]:
        return "before the first labelled hit"
    if any(last < frame and (next_first is None or frame < next_first) for (_, last), next_first in zip(labelled_rally_spans, [span[0] for span in labelled_rally_spans[1:]] + [None])):
        return "after the last hit of a rally"
    if is_inpainted_at(tracks, frame):
        return "within a rally, on an InpaintNet-filled frame"
    return "within a rally, no labelled hit nearby"


def is_inpainted_at(tracks: FrameTracks, frame: int) -> bool:
    if tracks.shuttle_inpainted is None:
        return False
    index = int(np.searchsorted(tracks.frames, frame))
    return bool(index < len(tracks.frames) and tracks.frames[index] == frame and tracks.shuttle_inpainted[index])


def error_breakdown(
    hits: pd.DataFrame,
    labels: pd.DataFrame,
    tracks: FrameTracks,
    settings: HitDetectionSettings = HitDetectionSettings(),
    tolerance: int = MATCH_TOLERANCE_FRAMES,
) -> pd.DataFrame:
    matched = match_hits(hits["frame"].tolist(), labels["frame"].tolist(), tolerance)
    matched_detected = {detected for detected, _ in matched}
    matched_labelled = {labelled for _, labelled in matched}
    first_frame_by_rally = labels.groupby("rally")["frame"].min()
    spans = sorted((int(rally_labels["frame"].min()), int(rally_labels["frame"].max())) for _, rally_labels in labels.groupby("rally"))
    rows = [
        {
            "kind": "miss",
            "frame": int(label.frame),
            "cause": miss_cause(int(label.frame), int(label.frame) == int(first_frame_by_rally[label.rally]), tracks, settings, tolerance),
            "inpainted": is_inpainted_at(tracks, int(label.frame)),
        }
        for label in labels.itertuples()
        if int(label.frame) not in matched_labelled
    ] + [
        {
            "kind": "false positive",
            "frame": int(hit.frame),
            "cause": false_positive_cause(int(hit.frame), spans, tracks),
            "inpainted": is_inpainted_at(tracks, int(hit.frame)),
        }
        for hit in hits.itertuples()
        if int(hit.frame) not in matched_detected
    ]
    return pd.DataFrame(rows, columns=["kind", "frame", "cause", "inpainted"]).sort_values(["kind", "frame"]).reset_index(drop=True)


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vision.hits",
        description="Detect hits and rallies from the player and shuttle tracks of a frame range.",
    )
    parser.add_argument("video", type=Path, help="path to the match video")
    parser.add_argument("--start", type=int, required=True, help="first frame of the tracked range")
    parser.add_argument("--end", type=int, required=True, help="last frame of the tracked range")
    parser.add_argument(
        "--evaluate", nargs=2, metavar=("MATCH_FOLDER", "SET_NO"), help="compare with the ShuttleSet hit frames of this set"
    )
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        tracks = load_frame_tracks(parsed.video, parsed.start, parsed.end)
        hits, rallies = segment_rallies(detect_hits(tracks), tracks)
        labels = None
        if parsed.evaluate:
            match_folder, set_no = parsed.evaluate
            homography = load_calibration(calibration_path(parsed.video))["pixel_to_court_homography"]
            labels = labelled_hits(load_set_shots(SHUTTLESET_SET_DIR, match_folder, int(set_no)), homography, parsed.start, parsed.end)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    hits_file, rallies_file = output_paths(parsed.video, parsed.start, parsed.end)
    hits_file.parent.mkdir(parents=True, exist_ok=True)
    hits.to_csv(hits_file, index=False)
    rallies.to_csv(rallies_file, index=False)
    print(f"rallies: {len(rallies)}, hits: {len(hits)}")
    if labels is not None:
        for line in evaluation_lines(hits, labels):
            print(line)
        errors = error_breakdown(hits, labels, tracks)
        for error in errors.itertuples():
            print(f"{error.kind} at {error.frame}: {error.cause}" + (" (InpaintNet-filled frame)" if error.inpainted else ""))
        for kind, kind_errors in errors.groupby("kind"):
            plural = "misses" if kind == "miss" else "false positives"
            print(f"{plural} on InpaintNet-filled frames: {int(kind_errors['inpainted'].sum())}/{len(kind_errors)}")
    print(f"saved {hits_file}")
    print(f"saved {rallies_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
