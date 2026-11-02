import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from vision.alignment import SHUTTLESET_SET_DIR, load_set_shots, open_video, read_frame
from vision.court import calibration_path, load_calibration, video_id
from vision.hits import (
    HIT_COLUMNS,
    HITS_DIR,
    MATCH_TOLERANCE_SECONDS,
    RALLY_COLUMNS,
    REFERENCE_FPS,
    REFERENCE_FRAME_WIDTH_PX,
    evaluation_lines,
    frames_for_fps,
    labelled_hits,
    match_hits,
    match_tolerances,
    tolerance_frames,
    tolerance_label,
    output_paths,
    require_track_file,
    video_fps_and_width,
)
from vision.players import PLAYERS, PLAYERS_DIR, boxes_path, effective_fps, frame_stride_of, is_default_dir, require_video_file, sampled_frames
from vision.players import positions_path as player_positions_path

MODEL_PATH = Path("data/models/movement_hits.joblib")
SERVE_TO_RETURN_SECONDS = 0.76
SAME_HIT_AS_SERVE_SECONDS = 0.25
SPOT_CHECK_DIR = Path("data/checks/hits")
SPOT_CHECK_SEED = 1
SPOT_CHECK_OFFSET_FRAMES = 6
TILE_WIDTH, TILE_HEIGHT = 480, 360
BOX_COLOR_BGR = {"me": (255, 160, 0), "opp": (0, 0, 255)}
SPEED_CAP_M_PER_S = 12.0
SELF_FEATURE_NAMES = (
    "speed", "speed_before", "speed_after", "acceleration", "acceleration_before", "acceleration_after",
    "direction_change", "box_height_rise", "box_height_fall", "box_width_rise", "box_width_fall",
    "y_m", "y_move", "detected",
)
OTHER_FEATURE_NAMES = ("speed", "speed_after", "acceleration", "acceleration_after", "direction_change", "y_move")
FEATURE_NAMES = tuple(f"self_{name}" for name in SELF_FEATURE_NAMES) + tuple(f"other_{name}" for name in OTHER_FEATURE_NAMES)
HIT_MODEL_PARAMETERS = {"max_depth": 4, "learning_rate": 0.05, "max_iter": 200, "class_weight": "balanced", "random_state": 0}


@dataclass(frozen=True)
class MovementSettings:
    position_smoothing_frames: int = 5
    speed_smoothing_frames: int = 3
    direction_smoothing_frames: int = 7
    context_frames: int = 5
    score_smoothing_frames: int = 3
    interpolation_limit_frames: int = 10
    label_tolerance_frames: int = 2
    hit_score_min: float = 0.5
    rally_end_score_min: float = 0.7
    min_hit_gap_frames: int = 10
    max_hit_gap_frames: int = 70
    usual_gap_frames: tuple[int, int] = (15, 55)
    unusual_gap_penalty_per_10_frames: float = 0.3
    still_speed_m_per_s: float = 0.8
    still_frames: int = 15
    infer_serves: bool = True
    serve_to_return_seconds: float = SERVE_TO_RETURN_SECONDS
    fps: float = REFERENCE_FPS
    pixel_scale: float = 1.0


def settings_for_video(fps: float, frame_width: float, base: MovementSettings = MovementSettings()) -> MovementSettings:
    scale = lambda frames: frames_for_fps(frames, fps)
    return replace(
        base,
        position_smoothing_frames=scale(base.position_smoothing_frames),
        speed_smoothing_frames=scale(base.speed_smoothing_frames),
        direction_smoothing_frames=scale(base.direction_smoothing_frames),
        context_frames=scale(base.context_frames),
        score_smoothing_frames=scale(base.score_smoothing_frames),
        interpolation_limit_frames=scale(base.interpolation_limit_frames),
        label_tolerance_frames=scale(base.label_tolerance_frames),
        min_hit_gap_frames=scale(base.min_hit_gap_frames),
        max_hit_gap_frames=scale(base.max_hit_gap_frames),
        usual_gap_frames=(scale(base.usual_gap_frames[0]), scale(base.usual_gap_frames[1])),
        unusual_gap_penalty_per_10_frames=base.unusual_gap_penalty_per_10_frames * REFERENCE_FPS / fps,
        still_frames=scale(base.still_frames),
        fps=fps,
        pixel_scale=REFERENCE_FRAME_WIDTH_PX / frame_width,
    )


@dataclass(frozen=True)
class PlayerTracks:
    frames: np.ndarray
    near_points: np.ndarray
    far_points: np.ndarray
    near_boxes: np.ndarray
    far_boxes: np.ndarray

    def points_of(self, player: str) -> np.ndarray:
        return self.near_points if player == PLAYERS[0] else self.far_points

    def boxes_of(self, player: str) -> np.ndarray:
        return self.near_boxes if player == PLAYERS[0] else self.far_boxes


def rows_by_frame(table: pd.DataFrame, frames: np.ndarray, player: str, columns: list[str]) -> np.ndarray:
    player_rows = table[table["player"] == player].drop_duplicates("frame").set_index("frame")[columns]
    return player_rows.reindex(frames).to_numpy(dtype=float)


def player_tracks(positions: pd.DataFrame, boxes: pd.DataFrame, start_frame: int, end_frame: int, frame_stride: int = 1) -> PlayerTracks:
    frames = sampled_frames(start_frame, end_frame, frame_stride)
    return PlayerTracks(
        frames,
        rows_by_frame(positions, frames, PLAYERS[0], ["x_m", "y_m"]),
        rows_by_frame(positions, frames, PLAYERS[1], ["x_m", "y_m"]),
        rows_by_frame(boxes, frames, PLAYERS[0], ["left", "top", "right", "bottom"]),
        rows_by_frame(boxes, frames, PLAYERS[1], ["left", "top", "right", "bottom"]),
    )


def filled(values, interpolation_limit: int) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    if series.notna().sum() == 0:
        return np.zeros(len(series))
    return series.interpolate(limit=interpolation_limit).bfill().ffill().to_numpy()


def moving_average(values, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) == 0:
        return values.copy()
    before, after = (window - 1) // 2, window // 2
    padded = np.concatenate([np.full(before, values[0]), values, np.full(after, values[-1])])
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def shifted(values: np.ndarray, offset: int) -> np.ndarray:
    if len(values) == 0 or offset == 0:
        return values.copy()
    result = np.empty_like(values)
    if offset > 0:
        result[:-offset] = values[offset:]
        result[-offset:] = values[-1]
    else:
        result[-offset:] = values[:offset]
        result[:-offset] = values[0]
    return result


def motion_features(points: np.ndarray, boxes: np.ndarray, settings: MovementSettings) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(points) < 2:
        return {name: np.zeros(len(points)) for name in SELF_FEATURE_NAMES}
    x = moving_average(filled(points[:, 0], settings.interpolation_limit_frames), settings.position_smoothing_frames)
    y = moving_average(filled(points[:, 1], settings.interpolation_limit_frames), settings.position_smoothing_frames)
    vx, vy = np.gradient(x) * settings.fps, np.gradient(y) * settings.fps
    speed = np.minimum(np.hypot(vx, vy), SPEED_CAP_M_PER_S)
    acceleration = np.gradient(moving_average(speed, settings.speed_smoothing_frames)) * settings.fps
    heading = np.column_stack([moving_average(vx, settings.direction_smoothing_frames), moving_average(vy, settings.direction_smoothing_frames)])
    context = settings.context_frames
    direction_change = np.einsum("ij,ij->i", shifted(heading, -context), shifted(heading, context))
    height = moving_average(filled(boxes[:, 3] - boxes[:, 1], settings.interpolation_limit_frames), settings.speed_smoothing_frames) * settings.pixel_scale
    width = moving_average(filled(boxes[:, 2] - boxes[:, 0], settings.interpolation_limit_frames), settings.speed_smoothing_frames) * settings.pixel_scale
    return {
        "speed": speed,
        "speed_before": shifted(speed, -context),
        "speed_after": shifted(speed, context),
        "acceleration": acceleration,
        "acceleration_before": shifted(acceleration, -context),
        "acceleration_after": shifted(acceleration, context),
        "direction_change": direction_change,
        "box_height_rise": height - shifted(height, -context),
        "box_height_fall": shifted(height, context) - height,
        "box_width_rise": width - shifted(width, -context),
        "box_width_fall": shifted(width, context) - width,
        "y_m": y,
        "y_move": shifted(y, context) - shifted(y, -context),
        "detected": np.isfinite(points[:, 0]).astype(float),
    }


def other_player(player: str) -> str:
    return PLAYERS[1] if player == PLAYERS[0] else PLAYERS[0]


def hit_feature_matrix(tracks: PlayerTracks, player: str, settings: MovementSettings) -> np.ndarray:
    own = motion_features(tracks.points_of(player), tracks.boxes_of(player), settings)
    other = motion_features(tracks.points_of(other_player(player)), tracks.boxes_of(other_player(player)), settings)
    columns = [own[name] for name in SELF_FEATURE_NAMES] + [other[name] for name in OTHER_FEATURE_NAMES]
    return np.column_stack(columns) if len(tracks.frames) else np.zeros((0, len(FEATURE_NAMES)))


def hit_frame_labels(frames: np.ndarray, hit_frames, tolerance: int) -> np.ndarray:
    frames = np.asarray(frames, dtype=int)
    labels = np.zeros(len(frames))
    for hit_frame in hit_frames:
        labels[np.abs(frames - int(hit_frame)) <= tolerance] = 1.0
    return labels


def fit_hit_model(feature_matrix: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray | None = None) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**HIT_MODEL_PARAMETERS).fit(feature_matrix, labels, sample_weight=sample_weight)


def hit_scores(model, feature_matrix: np.ndarray, settings: MovementSettings) -> np.ndarray:
    if len(feature_matrix) == 0:
        return np.zeros(0)
    return moving_average(model.predict_proba(feature_matrix)[:, 1], settings.score_smoothing_frames)


def local_maxima(values: np.ndarray, min_gap: int, floor: float) -> list[int]:
    values = np.asarray(values, dtype=float)
    order = [int(index) for index in np.argsort(-values, kind="stable") if values[index] > floor]
    chosen = []
    for index in order:
        if all(abs(index - other) >= min_gap for other in chosen):
            chosen.append(index)
    return sorted(chosen)


@dataclass(frozen=True)
class HitCandidate:
    frame: int
    hitter: str
    score: float


def hit_candidates(frames: np.ndarray, scores_by_player: dict[str, np.ndarray], settings: MovementSettings, floor: float = 0.05) -> list[HitCandidate]:
    candidates = [
        HitCandidate(int(frames[index]), player, float(scores[index]))
        for player, scores in scores_by_player.items()
        for index in local_maxima(scores, settings.min_hit_gap_frames, floor)
    ]
    return sorted(candidates, key=lambda candidate: (candidate.frame, candidate.hitter))


def gap_penalty(gap: int, settings: MovementSettings) -> float:
    low, high = settings.usual_gap_frames
    return settings.unusual_gap_penalty_per_10_frames * (max(0, low - gap) + max(0, gap - high)) / 10


def alternating_hit_chain(candidates: list[HitCandidate], settings: MovementSettings) -> list[HitCandidate]:
    candidates = sorted(candidates, key=lambda candidate: (candidate.frame, candidate.hitter))
    best, previous = [], []
    for index, candidate in enumerate(candidates):
        weight = candidate.score - settings.hit_score_min
        best_total, best_previous = weight, -1
        for earlier in range(index - 1, -1, -1):
            gap = candidate.frame - candidates[earlier].frame
            if gap > settings.max_hit_gap_frames:
                total = best[earlier] + weight
            elif candidates[earlier].hitter != candidate.hitter and gap >= settings.min_hit_gap_frames:
                total = best[earlier] + weight - gap_penalty(gap, settings)
            else:
                continue
            if total > best_total:
                best_total, best_previous = total, earlier
        best.append(best_total)
        previous.append(best_previous)
    if not best:
        return []
    index, chain = int(np.argmax(best)), []
    while index >= 0:
        chain.append(candidates[index])
        index = previous[index]
    return chain[::-1]


def split_into_rallies(chain: list[HitCandidate], max_gap: int) -> list[list[HitCandidate]]:
    rallies, current = [], []
    for candidate in chain:
        if current and candidate.frame - current[-1].frame > max_gap:
            rallies.append(current)
            current = []
        current.append(candidate)
    if current:
        rallies.append(current)
    return rallies


def trim_rally_ends(rally: list[HitCandidate], end_score_min: float) -> list[HitCandidate]:
    while rally and rally[-1].score < end_score_min:
        rally = rally[:-1]
    while rally and rally[0].score < end_score_min:
        rally = rally[1:]
    return rally


def both_still_from(frame_index: int, near_speed: np.ndarray, far_speed: np.ndarray, settings: MovementSettings) -> bool:
    window = slice(frame_index, frame_index + settings.still_frames)
    if frame_index + settings.still_frames > len(near_speed):
        return False
    return bool(np.all(near_speed[window] < settings.still_speed_m_per_s) and np.all(far_speed[window] < settings.still_speed_m_per_s))


def rally_end_frame(last_hit_frame: int, next_rally_start: int, frames: np.ndarray, near_speed: np.ndarray, far_speed: np.ndarray, settings: MovementSettings) -> int:
    first = int(np.searchsorted(frames, last_hit_frame)) + 1
    latest = min(int(frames[-1]), last_hit_frame + settings.max_hit_gap_frames, next_rally_start - 1)
    for index in range(first, int(np.searchsorted(frames, latest, side="right"))):
        if both_still_from(index, near_speed, far_speed, settings):
            return int(frames[index])
    return latest


def serve_gap_frames(settings: MovementSettings) -> int:
    return max(1, int(round(settings.serve_to_return_seconds * settings.fps)))


def inferred_serve(rally: list[HitCandidate], first_frame: int, settings: MovementSettings) -> HitCandidate | None:
    serve_frame = rally[0].frame - serve_gap_frames(settings)
    if serve_frame < int(first_frame):
        return None
    return HitCandidate(serve_frame, other_player(rally[0].hitter), float("nan"))


def with_inferred_serves(rallies: list[list[HitCandidate]], first_frame: int, settings: MovementSettings) -> list[list[HitCandidate]]:
    if not settings.infer_serves:
        return rallies
    served = []
    for rally in rallies:
        serve = inferred_serve(rally, first_frame, settings) if rally else None
        served.append(rally if serve is None else [serve] + rally)
    return served


def rally_tables(rallies: list[list[HitCandidate]], tracks: PlayerTracks, settings: MovementSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    near_speed = motion_features(tracks.near_points, tracks.near_boxes, settings)["speed"]
    far_speed = motion_features(tracks.far_points, tracks.far_boxes, settings)["speed"]
    hit_rows, rally_rows = [], []
    rallies = with_inferred_serves(rallies, int(tracks.frames[0]) if len(tracks.frames) else 0, settings)
    for number, rally in enumerate(rallies, start=1):
        next_start = rallies[number][0].frame if number < len(rallies) else int(tracks.frames[-1]) + 1
        for candidate in rally:
            hit_rows.append({"rally": number, "frame": candidate.frame, "hitter": candidate.hitter, "score": round(candidate.score, 3), "reach": np.nan})
        rally_rows.append(
            {
                "rally": number,
                "start_frame": rally[0].frame,
                "end_frame": rally_end_frame(rally[-1].frame, next_start, tracks.frames, near_speed, far_speed, settings),
                "hits": len(rally),
            }
        )
    return pd.DataFrame(hit_rows, columns=HIT_COLUMNS), pd.DataFrame(rally_rows, columns=RALLY_COLUMNS)


def sample_spacing(tracks: PlayerTracks) -> int:
    frames = np.asarray(tracks.frames, dtype=int)
    return int(frames[1] - frames[0]) if len(frames) > 1 else 1


def tracks_on_sample_indices(tracks: PlayerTracks) -> PlayerTracks:
    return replace(tracks, frames=np.arange(len(tracks.frames)))


def video_frames_of(sample_indices, tracks: PlayerTracks) -> np.ndarray:
    return np.asarray(tracks.frames, dtype=int)[np.asarray(sample_indices, dtype=int)]


def tables_in_video_frames(hits: pd.DataFrame, rallies: pd.DataFrame, tracks: PlayerTracks) -> tuple[pd.DataFrame, pd.DataFrame]:
    hits = hits.assign(frame=video_frames_of(hits["frame"], tracks)) if len(hits) else hits
    if len(rallies):
        rallies = rallies.assign(start_frame=video_frames_of(rallies["start_frame"], tracks), end_frame=video_frames_of(rallies["end_frame"], tracks))
    return hits, rallies


def sample_hit_scores(tracks: PlayerTracks, model, settings: MovementSettings) -> dict[str, np.ndarray]:
    samples = tracks_on_sample_indices(tracks)
    return {player: hit_scores(model, hit_feature_matrix(samples, player, settings), settings) for player in PLAYERS}


def hits_from_scores(tracks: PlayerTracks, scores: dict[str, np.ndarray], settings: MovementSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = tracks_on_sample_indices(tracks)
    chain = alternating_hit_chain(hit_candidates(samples.frames, scores, settings), settings)
    rallies = [trimmed for rally in split_into_rallies(chain, settings.max_hit_gap_frames) if len(trimmed := trim_rally_ends(rally, settings.rally_end_score_min)) >= 2]
    hits, rally_table = rally_tables(rallies, samples, settings)
    return tables_in_video_frames(hits, rally_table, tracks)


def detect_hits(tracks: PlayerTracks, model, settings: MovementSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    return hits_from_scores(tracks, sample_hit_scores(tracks, model, settings), settings)


def newer_hits_in_rally(newer_hits: pd.DataFrame, serve_frame: int, end_frame: int, same_hit_frames: int) -> pd.DataFrame:
    frames = newer_hits["frame"]
    return newer_hits[(frames > serve_frame + same_hit_frames) & (frames <= end_frame)]


def rallies_with_newer_hits(rally_hits: pd.DataFrame, rallies: pd.DataFrame, newer_hits: pd.DataFrame, same_hit_frames: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    newer_detected = detected_hits_only(newer_hits).sort_values("frame")
    hit_tables, counts = [], []
    for rally in rallies.itertuples(index=False):
        serve_row = rally_hits[rally_hits["rally"] == rally.rally].sort_values("frame").head(1)
        following = newer_hits_in_rally(newer_detected, int(serve_row["frame"].iloc[0]), int(rally.end_frame), same_hit_frames) if len(serve_row) else newer_detected.iloc[:0]
        table = pd.concat([serve_row, following.assign(rally=rally.rally)], ignore_index=True)
        hit_tables.append(table)
        counts.append(len(table))
    hits = pd.concat(hit_tables, ignore_index=True)[HIT_COLUMNS] if hit_tables else pd.DataFrame(columns=HIT_COLUMNS)
    return hits, rallies.assign(hits=counts)


def save_hit_model(model, model_path: Path = MODEL_PATH, trained_on: list[str] | None = None) -> Path:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": list(FEATURE_NAMES), "trained_on": trained_on or []}, model_path)
    return model_path


def load_hit_model(model_path: Path = MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"movement hit model not found: {model_path} (train with: python -m vision.movement --train ...)")
    stored = joblib.load(model_path)
    if tuple(stored.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError(f"model {model_path} was trained on other features; retrain with python -m vision.movement --train")
    return stored["model"]


def players_files(video_path: Path, start_frame: int, end_frame: int, players_dir: Path = PLAYERS_DIR) -> tuple[Path, Path]:
    custom_dir = () if is_default_dir(players_dir, PLAYERS_DIR) else (players_dir,)
    return player_positions_path(video_path, start_frame, end_frame, *custom_dir), boxes_path(video_path, start_frame, end_frame, *custom_dir)


def load_player_tracks(video_path: Path, start_frame: int, end_frame: int, players_dir: Path = PLAYERS_DIR) -> PlayerTracks:
    range_arguments = f"{video_path} --start {start_frame} --end {end_frame}"
    positions_file, boxes_file = players_files(video_path, start_frame, end_frame, players_dir)
    require_track_file(positions_file, f"python -m vision.players {range_arguments}")
    require_track_file(boxes_file, f"python -m vision.players {range_arguments}")
    return player_tracks(pd.read_csv(positions_file), pd.read_csv(boxes_file), start_frame, end_frame, frame_stride_of(positions_file))


def track_frame_stride(video_path: Path, start_frame: int, end_frame: int, players_dir: Path = PLAYERS_DIR) -> int:
    return frame_stride_of(players_files(video_path, start_frame, end_frame, players_dir)[0])


def settings_for_tracks(video_path: Path, frame_stride: int, base: MovementSettings = MovementSettings()) -> MovementSettings:
    video_fps, frame_width = video_fps_and_width(video_path)
    return settings_for_video(effective_fps(video_fps, frame_stride), frame_width, base)


@dataclass(frozen=True)
class LabelledRange:
    video_path: Path
    start_frame: int
    end_frame: int
    set_no: int
    match_folder: str
    players_dir: Path = PLAYERS_DIR

    @property
    def name(self) -> str:
        return f"{video_id(self.video_path)} {self.start_frame}-{self.end_frame} set {self.set_no}"


def labelled_range_data(labelled_range: LabelledRange, settings: MovementSettings) -> tuple[PlayerTracks, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    tracks = load_player_tracks(labelled_range.video_path, labelled_range.start_frame, labelled_range.end_frame, labelled_range.players_dir)
    homography = load_calibration(calibration_path(labelled_range.video_path))["pixel_to_court_homography"]
    shots = load_set_shots(SHUTTLESET_SET_DIR, labelled_range.match_folder, labelled_range.set_no)
    labels = labelled_hits(shots, homography, labelled_range.start_frame, labelled_range.end_frame)
    features = {player: hit_feature_matrix(tracks, player, settings) for player in PLAYERS}
    targets = {
        player: hit_frame_labels(tracks.frames, labels.loc[labels["hitter"] == player, "frame"], settings.label_tolerance_frames * sample_spacing(tracks))
        for player in PLAYERS
    }
    return tracks, labels, features, targets


def stacked_training_data(range_data: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    feature_rows = [features[player] for _, _, features, _ in range_data for player in PLAYERS]
    target_rows = [targets[player] for _, _, _, targets in range_data for player in PLAYERS]
    return np.vstack(feature_rows), np.concatenate(target_rows)


def detected_hits_only(hits: pd.DataFrame) -> pd.DataFrame:
    return hits[hits["score"].notna()] if "score" in hits.columns else hits


def inferred_serves_only(hits: pd.DataFrame) -> pd.DataFrame:
    return hits[hits["score"].isna()] if "score" in hits.columns else hits.iloc[:0]


def serve_evaluation_line(hits: pd.DataFrame, labels: pd.DataFrame, tolerance: int) -> str:
    serves = inferred_serves_only(hits)
    first_shots = labels.loc[labels.groupby("rally")["frame"].idxmin()] if len(labels) else labels
    matched = match_hits(serves["frame"].tolist(), first_shots["frame"].tolist(), tolerance)
    serve_hitter = dict(zip(serves["frame"], serves["hitter"]))
    labelled_hitter = dict(zip(first_shots["frame"], first_shots["hitter"]))
    judged = [(serve_hitter[serve], labelled_hitter[labelled]) for serve, labelled in matched if pd.notna(labelled_hitter[labelled])]
    offsets = [serve - labelled for serve, labelled in matched]
    offset_text = f", offset mean {np.mean(offsets):+.1f}" if offsets else ""
    return (
        f"inferred serves: {len(serves)}, labelled first shots: {len(first_shots)}, within ±{tolerance} frames: {len(matched)}, "
        f"server right: {sum(a == b for a, b in judged)}/{len(judged)}{offset_text}"
    )


def evaluation_summary(hits: pd.DataFrame, labels: pd.DataFrame, tolerance: int) -> dict:
    hits = detected_hits_only(hits)
    matched = match_hits(hits["frame"].tolist(), labels["frame"].tolist(), tolerance)
    detected_hitter = dict(zip(hits["frame"], hits["hitter"]))
    labelled_hitter = dict(zip(labels["frame"], labels["hitter"]))
    judged = [(detected_hitter[detected], labelled_hitter[labelled]) for detected, labelled in matched if pd.notna(labelled_hitter[labelled])]
    return {
        "labelled": len(labels),
        "detected": len(hits),
        "matched": len(matched),
        "hitter_correct": sum(detected == labelled for detected, labelled in judged),
        "hitter_judged": len(judged),
    }


def summary_line(name: str, totals: dict, window: str) -> str:
    precision = totals["matched"] / totals["detected"] if totals["detected"] else 0.0
    recall = totals["matched"] / totals["labelled"] if totals["labelled"] else 0.0
    hitter = f"{totals['hitter_correct']}/{totals['hitter_judged']}"
    return f"{name} {window}: labelled {totals['labelled']}, detected {totals['detected']}, matched {totals['matched']}, precision {precision:.1%}, recall {recall:.1%}, hitter correct {hitter}"


def add_totals(totals: dict, summary: dict) -> dict:
    return {key: totals.get(key, 0) + value for key, value in summary.items()}


def cross_validation_lines(labelled_ranges: list[LabelledRange], settings: MovementSettings, seconds: tuple[float, ...] = MATCH_TOLERANCE_SECONDS) -> list[str]:
    range_data = [labelled_range_data(labelled_range, settings) for labelled_range in labelled_ranges]
    video_fps = video_fps_and_width(labelled_ranges[0].video_path)[0] if labelled_ranges else settings.fps
    windows = match_tolerances(video_fps, seconds)
    lines, totals = [], {window: {} for window, _ in windows}
    for held_out, (tracks, labels, _, _) in zip(labelled_ranges, range_data):
        features, targets = stacked_training_data([data for data, other in zip(range_data, labelled_ranges) if other is not held_out])
        hits, _ = detect_hits(tracks, fit_hit_model(features, targets), settings)
        for window, frames in windows:
            summary = evaluation_summary(hits, labels, frames)
            totals[window] = add_totals(totals[window], summary)
            lines.append(summary_line(f"held out {held_out.name}", summary, tolerance_label(window, video_fps)))
    for window, _ in windows:
        lines.append(summary_line("all ranges, each held out", totals[window], tolerance_label(window, video_fps)))
    return lines


def miss_lines(hits: pd.DataFrame, labels: pd.DataFrame, tolerance: int) -> list[str]:
    hits = detected_hits_only(hits)
    matched = match_hits(hits["frame"].tolist(), labels["frame"].tolist(), tolerance)
    matched_labelled = {labelled for _, labelled in matched}
    matched_detected = {detected for detected, _ in matched}
    first_frames = labels.groupby("rally")["frame"].min()
    lines = [
        f"miss at {int(label.frame)}: {label.type}" + (" (first shot of the rally)" if int(label.frame) == int(first_frames[label.rally]) else "")
        for label in labels.itertuples()
        if int(label.frame) not in matched_labelled
    ]
    spans = sorted((int(rally["frame"].min()), int(rally["frame"].max())) for _, rally in labels.groupby("rally"))
    for hit in hits.itertuples():
        if int(hit.frame) in matched_detected:
            continue
        inside = any(start - tolerance <= hit.frame <= end + tolerance for start, end in spans)
        lines.append(f"false positive at {int(hit.frame)} ({hit.hitter}): " + ("within a labelled rally" if inside else "outside the labelled rallies"))
    return lines


def spot_check_path(video_path: Path, start_frame: int, end_frame: int, checks_dir: Path = SPOT_CHECK_DIR) -> Path:
    return checks_dir / f"{video_id(video_path)}_movement_spot_check_{start_frame}_{end_frame}.jpg"


def spot_check_sample(hits: pd.DataFrame, hit_count: int, seed: int = SPOT_CHECK_SEED) -> pd.DataFrame:
    ordered = hits.sort_values("frame").reset_index(drop=True)
    picks = np.random.default_rng(seed).choice(len(ordered), size=min(hit_count, len(ordered)), replace=False)
    return ordered.iloc[sorted(picks)]


def box_row_at(boxes: pd.DataFrame, player: str, frame_index: int):
    rows = boxes[(boxes["player"] == player) & (boxes["frame"] == frame_index)]
    return None if rows.empty else rows.iloc[0]


def crop_around(frame: np.ndarray, centre, caption: str, width: int = TILE_WIDTH, height: int = TILE_HEIGHT) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    left = int(min(max(centre[0] - width / 2, 0), max(frame_width - width, 0)))
    top = int(min(max(centre[1] - height / 2, 0), max(frame_height - height, 0)))
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    crop = frame[top : top + height, left : left + width]
    tile[: crop.shape[0], : crop.shape[1]] = crop
    cv2.putText(tile, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(tile, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return tile


def hit_tile(frame: np.ndarray, frame_index: int, boxes: pd.DataFrame, centre, caption: str) -> np.ndarray:
    drawn = frame.copy()
    for player in PLAYERS:
        row = box_row_at(boxes, player, frame_index)
        if row is not None:
            cv2.rectangle(drawn, (int(row["left"]), int(row["top"])), (int(row["right"]), int(row["bottom"])), BOX_COLOR_BGR[player], 2)
    return crop_around(drawn, centre, caption)


def box_centre(box_row) -> tuple[float, float]:
    if box_row is None:
        return 0.0, 0.0
    return (float(box_row["left"]) + float(box_row["right"])) / 2, (float(box_row["top"]) + float(box_row["bottom"])) / 2


def write_spot_check(video_path: Path, hits: pd.DataFrame, boxes: pd.DataFrame, hit_count: int, output_path: Path, fps: float) -> Path:
    offset = frames_for_fps(SPOT_CHECK_OFFSET_FRAMES, fps)
    capture = open_video(Path(video_path))
    rows = []
    try:
        for hit in spot_check_sample(hits, hit_count).itertuples():
            frame_index = int(hit.frame)
            centre = box_centre(box_row_at(boxes, hit.hitter, frame_index))
            tiles = []
            for shift, label in ((-offset, "before"), (0, "hit"), (offset, "after")):
                frame = read_frame(capture, frame_index + shift)
                if frame is None:
                    tiles.append(np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8))
                    continue
                caption = f"r{hit.rally} f{frame_index} {hit.hitter} score {hit.score:.2f}" if shift == 0 else f"{label} f{frame_index + shift}"
                tiles.append(hit_tile(frame, frame_index + shift, boxes, centre, caption))
            rows.append(np.hstack(tiles))
    finally:
        capture.release()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack(rows) if rows else np.zeros((TILE_HEIGHT, 3 * TILE_WIDTH, 3), dtype=np.uint8))
    return output_path


def labels_from_file(labels_file: Path, start_frame: int, end_frame: int) -> pd.DataFrame:
    from vision.label import reviewed_rows_only

    table = reviewed_rows_only(pd.read_csv(labels_file), labels_file)
    in_range = table[(table["frame_num"] >= start_frame) & (table["frame_num"] <= end_frame)]
    return pd.DataFrame(
        {
            "rally": in_range["rally"].astype(int),
            "frame": in_range["frame_num"].astype(int),
            "hitter": in_range["player"].where(in_range["player"].isin(PLAYERS)),
            "type": in_range["type_group"],
            "detail": in_range["type_detail"].where(in_range["type_detail"].astype(object).map(lambda value: isinstance(value, str) and value != "")) if "type_detail" in in_range.columns else np.nan,
        },
        columns=["rally", "frame", "hitter", "type", "detail"],
    ).reset_index(drop=True)


def type_agreement_line(hits: pd.DataFrame, labels: pd.DataFrame, predictions: pd.DataFrame, tolerance: int) -> str:
    matched = match_hits(detected_hits_only(hits)["frame"].tolist(), labels["frame"].tolist(), tolerance)
    predicted = dict(zip(predictions["frame_num"], predictions["type_pred"]))
    labelled = dict(zip(labels["frame"], labels["type"]))
    judged = [(predicted.get(detected), labelled[labelled_frame]) for detected, labelled_frame in matched]
    judged = [(prediction, group) for prediction, group in judged if pd.notna(prediction) and pd.notna(group) and group != "unknown"]
    agreed = sum(prediction == group for prediction, group in judged)
    return f"shot type: {agreed}/{len(judged)} matched hits with a prediction and a labelled type agree" + (f" = {agreed / len(judged):.1%}" if judged else "")


def detail_agreement_line(hits: pd.DataFrame, labels: pd.DataFrame, predictions: pd.DataFrame, tolerance: int) -> str:
    if "type_detail_pred" not in predictions.columns:
        return "shot detail: no detail predictions in the database"
    matched = match_hits(detected_hits_only(hits)["frame"].tolist(), labels["frame"].tolist(), tolerance)
    predicted = dict(zip(predictions["frame_num"], predictions["type_detail_pred"]))
    labelled = dict(zip(labels["frame"], labels["detail"]))
    judged = [(predicted.get(detected), labelled[labelled_frame]) for detected, labelled_frame in matched]
    judged = [(prediction, detail) for prediction, detail in judged if pd.notna(prediction) and pd.notna(detail) and detail != "unknown"]
    agreed = sum(prediction == detail for prediction, detail in judged)
    return f"shot detail: {agreed}/{len(judged)} matched hits with a predicted and a labelled detail agree" + (f" = {agreed / len(judged):.1%}" if judged else "")


def own_predictions(database_path: Path, match_id: str) -> pd.DataFrame:
    import duckdb

    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        columns = {row[0] for row in connection.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'shots'").fetchall()}
        detail = ", shots.type_detail_pred" if "type_detail_pred" in columns else ""
        return connection.execute(
            f"SELECT shots.frame_num, shots.type_pred{detail} FROM shots WHERE shots.source = 'own' AND shots.match_id = ? AND shots.type_pred IS NOT NULL", [match_id]
        ).df()
    finally:
        connection.close()


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vision.movement",
        description="Detect hits and rallies from the two players' movement alone (no shuttle track).",
    )
    parser.add_argument("video", type=Path, help="path to the match video")
    parser.add_argument("--start", type=int, help="first frame of the tracked range")
    parser.add_argument("--end", type=int, help="last frame of the tracked range")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="hit model file (joblib)")
    parser.add_argument("--evaluate", nargs=2, metavar=("MATCH_FOLDER", "SET_NO"), help="compare with the ShuttleSet hit frames of this set")
    parser.add_argument("--evaluate-labels", type=Path, metavar="CSV", help="compare with hand labels (python -m vision.label): rally, frame_num, player, type_group")
    parser.add_argument("--database", type=Path, help="with --evaluate-labels and --match-id: shots database for the type_pred vs type_group agreement")
    parser.add_argument("--match-id", help="with --evaluate-labels: the own rows' match id in --database")
    parser.add_argument("--tolerance", type=float, action="append", default=[], help="matching window in seconds for --evaluate / --evaluate-labels / --cross-validate, turned into frames at the video's own fps (default 0.25 and 0.5); repeatable")
    parser.add_argument("--train", action="store_true", help="fit the model on the labelled --range ranges of this video and save it to --model")
    parser.add_argument("--cross-validate", action="store_true", help="hold out each --range in turn and report precision and recall")
    parser.add_argument("--range", nargs=3, type=int, action="append", default=[], metavar=("START", "END", "SET_NO"), help="a labelled range for --train or --cross-validate; repeatable")
    parser.add_argument("--match", help="ShuttleSet match folder of the --range ranges")
    parser.add_argument("--hits-dir", type=Path, default=HITS_DIR, help="root of the output files (default data/hits)")
    parser.add_argument("--players-dir", type=Path, default=PLAYERS_DIR, help="root of the player tracking files (default data/players)")
    parser.add_argument("--spot-check", type=int, metavar="N", help="draw N random detected hits (before, at, after) to data/checks/hits/")
    parser.add_argument("--in-play-model", type=Path, metavar="MODEL", help="rallies are the stretches this in-play model finds (python -m vision.in_play), instead of the gap rule; the inferred serve sits at each stretch's start")
    parser.add_argument("--hit-score-min", type=float, default=MovementSettings().hit_score_min, help="hit score a candidate needs (default 0.5)")
    parser.add_argument("--rally-end-score-min", type=float, default=MovementSettings().rally_end_score_min, help="score a rally's first and last hit need (default 0.7)")
    parser.add_argument("--no-serves", action="store_true", help="don't add an inferred serve row at the start of each rally")
    parser.add_argument("--rallies-from", type=Path, metavar="MODEL", help="rallies and serve rows come from this hit model (with the gap rule and the inferred serve); the --model hits fill each rally after its serve row, and its hits outside these rallies are dropped")
    parser.add_argument("--rallies-hit-score-min", type=float, default=MovementSettings().hit_score_min, help="with --rallies-from: that model's hit score minimum (default 0.5)")
    parser.add_argument("--rallies-end-score-min", type=float, default=MovementSettings().rally_end_score_min, help="with --rallies-from: that model's rally end score minimum (default 0.7)")
    parser.add_argument("--serve-stance", action="store_true", help="with --in-play-model: each stretch's serve is the stance rule's (python -m vision.serve_stance): both players still in diagonal service courts, the first to move serves; the --model hits fill the rally after it and its hits after the stretch are dropped")
    parser.add_argument("--stretch-model", type=Path, metavar="MODEL", help="with --in-play-model: the hit model whose scores feed the in-play model (and, with --serve-stance, the stance serve, at its default thresholds); the --model hits fill the stretches (default: --model does both)")
    return parser.parse_args(arguments)


def labelled_ranges_of(parsed: argparse.Namespace) -> list[LabelledRange]:
    if not parsed.range or not parsed.match:
        raise ValueError("--train and --cross-validate need --match and at least one --range START END SET_NO")
    return [LabelledRange(parsed.video, start, end, set_no, parsed.match, parsed.players_dir) for start, end, set_no in parsed.range]


def train_command(parsed: argparse.Namespace, settings: MovementSettings) -> int:
    labelled_ranges = labelled_ranges_of(parsed)
    features, targets = stacked_training_data([labelled_range_data(labelled_range, settings) for labelled_range in labelled_ranges])
    model_path = save_hit_model(fit_hit_model(features, targets), parsed.model, [labelled_range.name for labelled_range in labelled_ranges])
    print(f"trained on {len(labelled_ranges)} ranges, {len(targets)} player-frames, {int(targets.sum())} hit frames")
    print(f"saved {model_path}")
    return 0


def detected_tables(tracks: PlayerTracks, model, in_play_model_path: Path | None, settings: MovementSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    if in_play_model_path is None:
        return detect_hits(tracks, model, settings)
    from vision.in_play import detect_hits_in_play, load_in_play_model

    return detect_hits_in_play(tracks, sample_hit_scores(tracks, model, settings), load_in_play_model(in_play_model_path), settings)


def tables_with_rallies_from(tracks: PlayerTracks, parsed: argparse.Namespace, settings: MovementSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    if parsed.in_play_model is not None:
        raise ValueError("give --rallies-from or --in-play-model, not both")
    if not settings.infer_serves:
        raise ValueError("--rallies-from needs the inferred serves; drop --no-serves")
    rally_settings = replace(settings, hit_score_min=parsed.rallies_hit_score_min, rally_end_score_min=parsed.rallies_end_score_min)
    rally_hits, rallies = detect_hits(tracks, load_hit_model(parsed.rallies_from), rally_settings)
    newer_hits, _ = detect_hits(tracks, load_hit_model(parsed.model), settings)
    same_hit_frames = tolerance_frames(SAME_HIT_AS_SERVE_SECONDS, video_fps_and_width(parsed.video)[0])
    return rallies_with_newer_hits(rally_hits, rallies, newer_hits, same_hit_frames)


def tables_with_stretch_model(tracks: PlayerTracks, parsed: argparse.Namespace, settings: MovementSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    if parsed.in_play_model is None:
        raise ValueError("--serve-stance needs --in-play-model")
    if parsed.serve_stance and not settings.infer_serves:
        raise ValueError("--serve-stance adds the serve rows; drop --no-serves")
    from vision.in_play import load_in_play_model
    from vision.serve_stance import stance_rallies

    from vision.in_play import hits_in_play, in_play_feature_matrix, in_play_probability

    stretch_settings = replace(settings, hit_score_min=MovementSettings().hit_score_min, rally_end_score_min=MovementSettings().rally_end_score_min)
    hit_scores = sample_hit_scores(tracks, load_hit_model(parsed.model), settings)
    stretch_scores = hit_scores if parsed.stretch_model is None else sample_hit_scores(tracks, load_hit_model(parsed.stretch_model), settings)
    in_play_model = load_in_play_model(parsed.in_play_model)
    if not parsed.serve_stance:
        probability = in_play_probability(in_play_model, in_play_feature_matrix(tracks, stretch_scores, settings), settings)
        return hits_in_play(tracks, hit_scores, probability, settings)
    return stance_rallies(tracks, stretch_scores, hit_scores, in_play_model, settings, stretch_settings)


def detect_command(parsed: argparse.Namespace, settings: MovementSettings) -> int:
    if parsed.start is None or parsed.end is None:
        raise ValueError("give --start and --end")
    tracks = load_player_tracks(parsed.video, parsed.start, parsed.end, parsed.players_dir)
    settings = settings_for_tracks(parsed.video, track_frame_stride(parsed.video, parsed.start, parsed.end, parsed.players_dir), settings)
    if parsed.stretch_model is not None and parsed.in_play_model is None:
        raise ValueError("--stretch-model needs --in-play-model")
    if parsed.rallies_from is not None:
        hits, rallies = tables_with_rallies_from(tracks, parsed, settings)
    elif parsed.serve_stance or parsed.stretch_model is not None:
        hits, rallies = tables_with_stretch_model(tracks, parsed, settings)
    else:
        hits, rallies = detected_tables(tracks, load_hit_model(parsed.model), parsed.in_play_model, settings)
    labels = None
    if parsed.evaluate:
        match_folder, set_no = parsed.evaluate
        homography = load_calibration(calibration_path(parsed.video))["pixel_to_court_homography"]
        labels = labelled_hits(load_set_shots(SHUTTLESET_SET_DIR, match_folder, int(set_no)), homography, parsed.start, parsed.end)
    if parsed.evaluate_labels:
        labels = labels_from_file(parsed.evaluate_labels, parsed.start, parsed.end)
    hits_file, rallies_file = output_paths(parsed.video, parsed.start, parsed.end, parsed.hits_dir)
    hits_file.parent.mkdir(parents=True, exist_ok=True)
    hits.to_csv(hits_file, index=False)
    rallies.to_csv(rallies_file, index=False)
    print(f"rallies: {len(rallies)}, hits: {len(detected_hits_only(hits))}, inferred serves: {len(inferred_serves_only(hits))}")
    if labels is not None:
        video_fps = video_fps_and_width(parsed.video)[0]
        windows = match_tolerances(video_fps, tuple(parsed.tolerance) or MATCH_TOLERANCE_SECONDS)
        for window, frames in windows:
            for line in evaluation_lines(detected_hits_only(hits), labels, frames):
                print(f"{tolerance_label(window, video_fps)}: {line}")
            if settings.infer_serves:
                print(f"{tolerance_label(window, video_fps)}: {serve_evaluation_line(hits, labels, frames)}")
        for line in miss_lines(hits, labels, windows[0][1]):
            print(line)
        if parsed.evaluate_labels and parsed.database and parsed.match_id:
            predictions = own_predictions(parsed.database, parsed.match_id)
            for window, frames in windows:
                print(f"{tolerance_label(window, video_fps)}: {type_agreement_line(hits, labels, predictions, frames)}")
                print(f"{tolerance_label(window, video_fps)}: {detail_agreement_line(hits, labels, predictions, frames)}")
    print(f"saved {hits_file}")
    print(f"saved {rallies_file}")
    if parsed.spot_check:
        boxes = pd.read_csv(players_files(parsed.video, parsed.start, parsed.end, parsed.players_dir)[1])
        sheet = write_spot_check(parsed.video, detected_hits_only(hits), boxes, parsed.spot_check, spot_check_path(parsed.video, parsed.start, parsed.end), video_fps_and_width(parsed.video)[0])
        print(f"saved {sheet}")
    return 0


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        require_video_file(parsed.video)
        settings = replace(MovementSettings(), infer_serves=not parsed.no_serves, hit_score_min=parsed.hit_score_min, rally_end_score_min=parsed.rally_end_score_min)
        if parsed.cross_validate or parsed.train:
            labelled_ranges = labelled_ranges_of(parsed)
            strides = {track_frame_stride(r.video_path, r.start_frame, r.end_frame, r.players_dir) for r in labelled_ranges}
            if len(strides) > 1:
                raise ValueError(f"the --range tracks have different frame strides: {sorted(strides)}")
            settings = settings_for_tracks(parsed.video, strides.pop(), settings)
        if parsed.cross_validate:
            for line in cross_validation_lines(labelled_ranges_of(parsed), settings, tuple(parsed.tolerance) or MATCH_TOLERANCE_SECONDS):
                print(line)
            return 0
        if parsed.train:
            return train_command(parsed, settings)
        return detect_command(parsed, settings)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
