import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vision.alignment import SHUTTLESET_SET_DIR, load_set_shots, open_video
from vision.court import calibration_path, load_calibration, video_id
from vision.hits import side_of_hitter
from vision.movement import MODEL_PATH, PLAYERS, load_hit_model, load_player_tracks, sample_hit_scores, settings_for_video
from vision.players import calibration_frame_share, court_line_sample_share, frames_in_range, main_view_min_share, video_fps_and_width

OFFSETS_DIR = Path("data/alignment")
CHECKS_DIR = Path("data/checks/alignment")
VIEW_SAMPLE_STRIDE = 3
MAX_OFFSET_SECONDS = 30.0
CLARITY_EXCLUSION_SECONDS = 10.0
FINE_SEARCH_SECONDS = 3.0
FINE_EXCLUSION_FRAMES = 10
MIN_FINE_CLARITY = 0.2
MIN_PEAK_CLARITY = 0.1
SET_NUMBERS = (1, 2, 3)
STRETCH_PADDING_SECONDS = 2.0
STRETCH_MERGE_SECONDS = 1.0
LABEL_FRAME_WIDTH = 1280.0


@dataclass(frozen=True)
class OffsetResult:
    offset_frames: int
    score: float
    runner_up_score: float
    clarity: float
    label_scale: float


def match_shots(match_folder: str, set_dir: Path = SHUTTLESET_SET_DIR) -> pd.DataFrame:
    sets = []
    for set_no in SET_NUMBERS:
        try:
            sets.append(load_set_shots(set_dir, match_folder, set_no).assign(set_no=set_no))
        except FileNotFoundError:
            continue
    if not sets:
        raise FileNotFoundError(f"no set CSVs for match {match_folder} under {set_dir}")
    return pd.concat(sets, ignore_index=True).dropna(subset=["frame_num"])


def rally_spans(shots: pd.DataFrame, frame_scale: float = 1.0) -> np.ndarray:
    grouped = shots.groupby(["set_no", "rally"])["frame_num"]
    spans = np.column_stack([grouped.min().to_numpy(), grouped.max().to_numpy()]) * frame_scale
    return np.round(spans).astype(int)


def rally_indicator(spans: np.ndarray, frames: np.ndarray, offset: int = 0) -> np.ndarray:
    frames = np.asarray(frames, dtype=int)
    inside = np.zeros(len(frames), dtype=bool)
    for start, end in np.asarray(spans, dtype=int).reshape(-1, 2):
        inside |= (frames >= start + offset) & (frames <= end + offset)
    return inside.astype(float)


def offset_score(view: np.ndarray, spans: np.ndarray, frames: np.ndarray, offset: int) -> float:
    indicator = rally_indicator(spans, frames, offset)
    if indicator.std() == 0 or np.asarray(view).std() == 0:
        return 0.0
    return float(np.corrcoef(indicator, view)[0, 1])


def offset_scores(view: np.ndarray, spans: np.ndarray, frames: np.ndarray, offsets) -> np.ndarray:
    return np.array([offset_score(view, spans, frames, int(offset)) for offset in offsets])


def peak_clarity(offsets, scores, exclusion: int) -> tuple[int, float, float, float]:
    offsets, scores = np.asarray(offsets, dtype=int), np.asarray(scores, dtype=float)
    best = int(np.argmax(scores))
    outside = np.abs(offsets - offsets[best]) > exclusion
    runner_up = float(scores[outside].max()) if outside.any() else float("nan")
    clarity = float(scores[best] - runner_up) if outside.any() else float("nan")
    return int(offsets[best]), float(scores[best]), runner_up, clarity


def offset_from_view(view: np.ndarray, frames: np.ndarray, spans: np.ndarray, max_offset: int, step: int, exclusion: int, frame_scale: float = 1.0) -> OffsetResult:
    offsets = np.arange(-max_offset, max_offset + 1, step)
    best, score, runner_up, clarity = peak_clarity(offsets, offset_scores(view, spans, frames, offsets), exclusion)
    return OffsetResult(best, score, runner_up, clarity, frame_scale)


def fine_offset_scores(label_frames, label_hitters, scores_by_player: dict, frames, offsets) -> np.ndarray:
    frames = np.asarray(frames, dtype=int)
    position_by_frame = {int(frame): index for index, frame in enumerate(frames)}
    means = []
    for offset in offsets:
        values = [
            float(scores_by_player[hitter][position_by_frame[int(frame) + int(offset)]])
            for frame, hitter in zip(label_frames, label_hitters)
            if hitter in scores_by_player and int(frame) + int(offset) in position_by_frame
        ]
        means.append(np.mean(values) if values else 0.0)
    return np.array(means)


def fine_offset(label_frames, label_hitters, scores_by_player: dict, frames, centre: int, half_width: int, exclusion: int = FINE_EXCLUSION_FRAMES) -> OffsetResult:
    offsets = np.arange(centre - half_width, centre + half_width + 1)
    best, score, runner_up, clarity = peak_clarity(offsets, fine_offset_scores(label_frames, label_hitters, scores_by_player, frames, offsets), exclusion)
    return OffsetResult(best, score, runner_up, clarity, 1.0)


def is_clear(result: OffsetResult, min_clarity: float = MIN_PEAK_CLARITY) -> bool:
    return bool(np.isfinite(result.clarity) and result.clarity >= min_clarity)


def tracking_stretches(spans: np.ndarray, fps: float, padding_seconds: float = STRETCH_PADDING_SECONDS, merge_seconds: float = STRETCH_MERGE_SECONDS) -> list[tuple[int, int]]:
    padding, merge_gap = int(round(padding_seconds * fps)), int(round(merge_seconds * fps))
    stretches: list[list[int]] = []
    for start, end in sorted(np.asarray(spans, dtype=int).reshape(-1, 2).tolist()):
        start, end = max(start - padding, 0), end + padding
        if stretches and start <= stretches[-1][1] + merge_gap:
            stretches[-1][1] = max(stretches[-1][1], end)
        else:
            stretches.append([start, end])
    return [(start, end) for start, end in stretches]


def labelled_hitters(shots: pd.DataFrame, homography: np.ndarray, frame_scale: float, offset: int, pixel_scale: float = 1.0) -> pd.DataFrame:
    rows = []
    for shot in shots.itertuples():
        hitter_pixel = (shot.player_location_x * pixel_scale, shot.player_location_y * pixel_scale)
        hitter = side_of_hitter(homography, hitter_pixel) if np.all(np.isfinite(hitter_pixel)) else None
        rows.append({"set_no": int(shot.set_no), "rally": int(shot.rally), "frame": int(round(shot.frame_num * frame_scale)) + int(offset), "hitter": hitter, "type": shot.type})
    return pd.DataFrame(rows, columns=["set_no", "rally", "frame", "hitter", "type"])


def pooled_scores(scored_stretches: list[tuple[np.ndarray, dict[str, np.ndarray]]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frames = np.concatenate([np.asarray(frames, dtype=int) for frames, _ in scored_stretches])
    return frames, {player: np.concatenate([scores[player] for _, scores in scored_stretches]) for player in PLAYERS}


def score_profile(scores: np.ndarray, offsets: np.ndarray, width: int = 900, height: int = 300, score_name: str = "correlation") -> np.ndarray:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    low, high = float(np.min(scores)), float(np.max(scores))
    span = high - low if high > low else 1.0
    points = np.column_stack([
        np.interp(offsets, (offsets.min(), offsets.max()), (10, width - 10)),
        height - 20 - (scores - low) / span * (height - 40),
    ]).round().astype(np.int32)
    cv2.polylines(image, [points.reshape(-1, 1, 2)], False, (180, 60, 0), 2, cv2.LINE_AA)
    zero_x = int(np.interp(0, (offsets.min(), offsets.max()), (10, width - 10)))
    cv2.line(image, (zero_x, 0), (zero_x, height), (160, 160, 160), 1)
    best = int(np.argmax(scores))
    cv2.circle(image, tuple(points[best]), 6, (0, 0, 220), -1)
    cv2.putText(image, f"best offset {int(offsets[best])} frames, {score_name} {scores[best]:.3f}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def view_signal(video_path: Path, calibration: dict, stride: int = VIEW_SAMPLE_STRIDE, end_frame: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    homography = calibration["pixel_to_court_homography"]
    minimum_share = main_view_min_share(calibration_frame_share(video_path, calibration))
    capture = open_video(Path(video_path))
    try:
        last = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1 if end_frame is None else end_frame
        frames, view = [], []
        for frame_index, frame in frames_in_range(capture, 0, last, stride):
            frames.append(frame_index)
            view.append(1.0 if court_line_sample_share(frame, homography) >= minimum_share else 0.0)
    finally:
        capture.release()
    return np.array(frames), np.array(view)


def video_fps(video_path: Path) -> float:
    capture = open_video(Path(video_path))
    try:
        return float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()


def offset_record_path(video_path: Path, offsets_dir: Path = OFFSETS_DIR) -> Path:
    return offsets_dir / f"{video_id(video_path)}.json"


def check_image_path(video_path: Path, checks_dir: Path = CHECKS_DIR) -> Path:
    return checks_dir / f"{video_id(video_path)}_offset.png"


def fine_check_image_path(video_path: Path, checks_dir: Path = CHECKS_DIR) -> Path:
    return checks_dir / f"{video_id(video_path)}_fine_offset.png"


def stretches_path(video_path: Path, offsets_dir: Path = OFFSETS_DIR) -> Path:
    return offsets_dir / f"{video_id(video_path)}_stretches.csv"


def read_offset_record(video_path: Path, offsets_dir: Path = OFFSETS_DIR) -> dict:
    record_path = offset_record_path(video_path, offsets_dir)
    if not record_path.is_file():
        raise FileNotFoundError(f"no offset record {record_path} (run: python -m vision.label_offset {video_path} --match <folder> --label-fps <fps>)")
    return json.loads(record_path.read_text())


def write_stretches(video_path: Path, offsets_dir: Path = OFFSETS_DIR) -> tuple[Path, list[tuple[int, int]]]:
    record = read_offset_record(video_path, offsets_dir)
    spans = rally_spans(match_shots(record["match"]), record["label_scale"]) + int(record["offset_frames"])
    stretches = tracking_stretches(spans, video_fps(video_path))
    output_path = stretches_path(video_path, offsets_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stretches, columns=["start_frame", "end_frame"]).to_csv(output_path, index=False)
    return output_path, stretches


def read_stretch_list(video_path: Path, offsets_dir: Path = OFFSETS_DIR) -> list[tuple[int, int]]:
    listing = stretches_path(video_path, offsets_dir)
    if not listing.is_file():
        raise FileNotFoundError(f"no stretch list {listing} (run: python -m vision.label_offset {video_path} --stretches)")
    table = pd.read_csv(listing)
    return [(int(start), int(end)) for start, end in zip(table["start_frame"], table["end_frame"])]


def match_labels(video_path: Path, record: dict, offset: int) -> pd.DataFrame:
    homography = load_calibration(calibration_path(Path(video_path)))["pixel_to_court_homography"]
    pixel_scale = video_fps_and_width(video_path)[1] / LABEL_FRAME_WIDTH
    return labelled_hitters(match_shots(record["match"]), homography, record["label_scale"], offset, pixel_scale)


def find_fine_offset(video_path: Path, model_path: Path = MODEL_PATH, offsets_dir: Path = OFFSETS_DIR, search_seconds: float = FINE_SEARCH_SECONDS) -> tuple[OffsetResult, np.ndarray, np.ndarray]:
    record = read_offset_record(video_path, offsets_dir)
    fps, frame_width = video_fps_and_width(video_path)
    settings = settings_for_video(fps, frame_width)
    model = load_hit_model(model_path)
    scored = []
    for start_frame, end_frame in read_stretch_list(video_path, offsets_dir):
        tracks = load_player_tracks(video_path, start_frame, end_frame)
        scored.append((tracks.frames, sample_hit_scores(tracks, model, settings)))
    frames, scores = pooled_scores(scored)
    labels = match_labels(video_path, record, 0)
    centre, half_width = int(record["offset_frames"]), int(round(search_seconds * fps))
    offsets = np.arange(centre - half_width, centre + half_width + 1)
    profile = fine_offset_scores(labels["frame"].to_numpy(), labels["hitter"].to_numpy(), scores, frames, offsets)
    best, score, runner_up, clarity = peak_clarity(offsets, profile, FINE_EXCLUSION_FRAMES)
    return OffsetResult(best, score, runner_up, clarity, float(record["label_scale"])), offsets, profile


def find_offset(video_path: Path, match_folder: str, label_fps: float, stride: int = VIEW_SAMPLE_STRIDE, max_offset_seconds: float = MAX_OFFSET_SECONDS) -> tuple[OffsetResult, np.ndarray, np.ndarray]:
    calibration = load_calibration(calibration_path(Path(video_path)))
    fps = video_fps(video_path)
    frame_scale = fps / label_fps
    spans = rally_spans(match_shots(match_folder), frame_scale)
    frames, view = view_signal(video_path, calibration, stride)
    max_offset = int(round(max_offset_seconds * fps))
    offsets = np.arange(-max_offset, max_offset + 1, stride)
    scores = offset_scores(view, spans, frames, offsets)
    best, score, runner_up, clarity = peak_clarity(offsets, scores, int(round(CLARITY_EXCLUSION_SECONDS * fps)))
    return OffsetResult(best, score, runner_up, clarity, frame_scale), offsets, scores


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m vision.label_offset", description="Find the frame offset between a ShuttleSet match's labels and its video from the broadcast's camera cuts.")
    parser.add_argument("video", type=Path, help="the match video (calibrated with python -m vision.court <video> --shuttleset-match <folder>)")
    parser.add_argument("--match", help="ShuttleSet match folder (for the coarse offset)")
    parser.add_argument("--label-fps", type=float, help="the fps the labels were made at (matches.fps; for the coarse offset)")
    parser.add_argument("--stretches", action="store_true", help="from the coarse offset, write the labelled rallies to track (padded 2 s, merged when less than 1 s apart) to data/alignment/<video id>_stretches.csv")
    parser.add_argument("--fine", action="store_true", help="after tracking the stretches: the fine offset from the movement hit scores, added to the offset record")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="hit model for --fine (default data/models/movement_hits.joblib)")
    parser.add_argument("--stride", type=int, default=VIEW_SAMPLE_STRIDE, help="sample every Nth frame for the view check (default 3)")
    parser.add_argument("--max-offset-seconds", type=float, default=MAX_OFFSET_SECONDS, help="search offsets up to this far either way (default 30 s)")
    return parser.parse_args(arguments)


def stretches_command(parsed: argparse.Namespace) -> int:
    output_path, stretches = write_stretches(parsed.video)
    frames = sum(end - start + 1 for start, end in stretches)
    print(f"{len(stretches)} stretches, {frames} frames to track")
    print(f"saved {output_path}")
    print(f"next: python -m vision.players {parsed.video} --stretches {output_path}")
    return 0


def fine_command(parsed: argparse.Namespace) -> int:
    result, offsets, profile = find_fine_offset(parsed.video, parsed.model)
    record = read_offset_record(parsed.video)
    clear = is_clear(result, MIN_FINE_CLARITY)
    record.update({"fine_offset_frames": result.offset_frames, "fine_score": result.score, "fine_runner_up_score": result.runner_up_score, "fine_clarity": result.clarity, "fine_clear": clear})
    record_path = offset_record_path(parsed.video)
    record_path.write_text(json.dumps(record, indent=1))
    image_path = fine_check_image_path(parsed.video); image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), score_profile(profile, offsets, score_name="mean hitter score"))
    print(f"fine offset {result.offset_frames} frames (coarse {record['offset_frames']}), mean hitter score {result.score:.3f}, runner-up {result.runner_up_score:.3f}, clarity {result.clarity:.3f}, {'clear' if clear else 'UNCLEAR'}")
    print(f"saved {record_path}"); print(f"saved {image_path}")
    return 0


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    if parsed.stretches and parsed.fine:
        print("error: --stretches and --fine are separate steps (tracking comes in between); give one", file=sys.stderr)
        return 1
    if parsed.stretches or parsed.fine:
        try:
            return stretches_command(parsed) if parsed.stretches else fine_command(parsed)
        except (FileNotFoundError, OSError, ValueError, KeyError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    if parsed.match is None or parsed.label_fps is None:
        print("error: the coarse offset needs --match and --label-fps", file=sys.stderr)
        return 1
    try:
        result, offsets, scores = find_offset(parsed.video, parsed.match, parsed.label_fps, parsed.stride, parsed.max_offset_seconds)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    record = {"match": parsed.match, "offset_frames": result.offset_frames, "label_scale": result.label_scale, "score": result.score, "runner_up_score": result.runner_up_score, "clarity": result.clarity, "clear": is_clear(result), "stride": parsed.stride}
    record_path = offset_record_path(parsed.video); record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=1))
    image_path = check_image_path(parsed.video); image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), score_profile(scores, offsets))
    print(f"offset {result.offset_frames} frames (label frames x {result.label_scale:.4f} + offset), correlation {result.score:.3f}, runner-up {result.runner_up_score:.3f}, clarity {result.clarity:.3f}, {'clear' if is_clear(result) else 'UNCLEAR'}")
    print(f"saved {record_path}"); print(f"saved {image_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
