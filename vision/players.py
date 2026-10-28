import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vision.alignment import SHUTTLESET_SET_DIR, load_set_shots, open_video
from vision.court import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NET_Y_M,
    calibration_path,
    court_line_segments,
    line_offsets_to_bright_pixels,
    load_calibration,
    pixels_to_court,
)

PLAYERS = ("me", "opp")
PERSON_CLASS_ID = 0
COURT_MARGIN_M = 1.0
BEHIND_FAR_BASELINE_MARGIN_M = 2.0
DUPLICATE_BOX_MAX_IOU = 0.5
LINE_SAMPLES_PER_LINE = 40
MAIN_VIEW_MIN_LINE_SAMPLE_SHARE = 0.6
DEFAULT_MODEL_PATH = Path("data/models/yolo11s.pt")
PLAYERS_DIR = Path("data/players")
CHECKS_DIR = Path("data/checks/players")
POSITION_COLUMNS = ["frame", "player", "x_m", "y_m", "track_id", "confidence"]
PLAYER_COLOR_BGR = {"me": (255, 160, 0), "opp": (0, 0, 255)}
BYSTANDER_COLOR_BGR = (160, 160, 160)


@dataclass(frozen=True)
class FrameDetections:
    boxes_xyxy: np.ndarray
    track_ids: np.ndarray
    confidences: np.ndarray


@dataclass
class PlayerTrackingState:
    side_by_track_id: dict[int, str] = field(default_factory=dict)
    track_id_by_player: dict[str, int] = field(default_factory=dict)


def empty_detections() -> FrameDetections:
    return FrameDetections(np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0))


def foot_points(boxes_xyxy) -> np.ndarray:
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    return np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])


def court_positions_of_feet(homography: np.ndarray, boxes_xyxy) -> np.ndarray:
    feet = foot_points(boxes_xyxy)
    if len(feet) == 0:
        return np.zeros((0, 2))
    return pixels_to_court(homography, feet)


def on_court_mask(
    court_positions, margin_m: float = COURT_MARGIN_M, far_margin_m: float = BEHIND_FAR_BASELINE_MARGIN_M
) -> np.ndarray:
    positions = np.asarray(court_positions, dtype=float).reshape(-1, 2)
    inside_width = (positions[:, 0] >= -margin_m) & (positions[:, 0] <= COURT_WIDTH_M + margin_m)
    inside_length = (positions[:, 1] >= -margin_m) & (positions[:, 1] <= COURT_LENGTH_M + far_margin_m)
    return inside_width & inside_length


def side_of_court(y_m: float) -> str:
    return "me" if y_m < NET_Y_M else "opp"


def court_line_sample_share(frame: np.ndarray, homography: np.ndarray) -> float:
    floor_lines = [segment for name, segment in court_line_segments().items() if name != "net"]
    found_samples = sum(
        len(line_offsets_to_bright_pixels(frame, homography, start, end, sample_count=LINE_SAMPLES_PER_LINE))
        for start, end in floor_lines
    )
    return found_samples / (LINE_SAMPLES_PER_LINE * len(floor_lines))


def is_main_view(line_sample_share: float, minimum_share: float = MAIN_VIEW_MIN_LINE_SAMPLE_SHARE) -> bool:
    return line_sample_share >= minimum_share


def sides_with_new_tracks(side_by_track_id: dict[int, str], track_ids, court_positions) -> dict[int, str]:
    updated_sides = dict(side_by_track_id)
    for track_id, (_, y_m) in zip(np.asarray(track_ids, dtype=int), np.asarray(court_positions).reshape(-1, 2)):
        updated_sides.setdefault(int(track_id), side_of_court(float(y_m)))
    return updated_sides


def chosen_detection_index(
    candidate_indices: list[int], track_ids: np.ndarray, confidences: np.ndarray, previous_track_id: int | None
) -> int | None:
    if not candidate_indices:
        return None
    for index in candidate_indices:
        if int(track_ids[index]) == previous_track_id:
            return index
    return max(candidate_indices, key=lambda index: float(confidences[index]))


def label_players(
    frame_index: int,
    detections: FrameDetections,
    court_positions: np.ndarray,
    state: PlayerTrackingState,
    margin_m: float = COURT_MARGIN_M,
) -> tuple[list[dict], PlayerTrackingState]:
    track_ids = np.asarray(detections.track_ids, dtype=int)
    on_court = on_court_mask(court_positions, margin_m)
    side_by_track_id = sides_with_new_tracks(state.side_by_track_id, track_ids[on_court], court_positions[on_court])
    track_id_by_player = dict(state.track_id_by_player)
    rows = []
    for player in PLAYERS:
        candidates = [
            index for index in np.flatnonzero(on_court) if side_by_track_id[int(track_ids[index])] == player
        ]
        index = chosen_detection_index(candidates, track_ids, detections.confidences, track_id_by_player.get(player))
        if index is None:
            continue
        track_id_by_player[player] = int(track_ids[index])
        rows.append(
            {
                "frame": frame_index,
                "player": player,
                "x_m": round(float(court_positions[index, 0]), 3),
                "y_m": round(float(court_positions[index, 1]), 3),
                "track_id": int(track_ids[index]),
                "confidence": round(float(detections.confidences[index]), 3),
            }
        )
    return rows, PlayerTrackingState(side_by_track_id, track_id_by_player)


def id_continuity(positions: pd.DataFrame, player: str, frames) -> dict:
    frame_list = list(frames)
    player_rows = positions[(positions["player"] == player) & positions["frame"].isin(frame_list)]
    track_counts = Counter(player_rows["track_id"].astype(int))
    modal_track_id, frames_with_modal_id = track_counts.most_common(1)[0] if track_counts else (None, 0)
    frame_count = len(frame_list)
    return {
        "player": player,
        "frames": frame_count,
        "frames_detected": len(player_rows),
        "track_ids": len(track_counts),
        "modal_track_id": modal_track_id,
        "frames_with_modal_id": frames_with_modal_id,
        "share_with_one_id": frames_with_modal_id / frame_count if frame_count else 0.0,
    }


def share_in_own_half(positions: pd.DataFrame, player: str) -> float:
    player_rows = positions[positions["player"] == player]
    if player_rows.empty:
        return 0.0
    own_half = player_rows["y_m"] < NET_Y_M if player == "me" else player_rows["y_m"] >= NET_Y_M
    return float(own_half.mean())


def rally_frame_range(shots: pd.DataFrame, rally: int) -> tuple[int, int]:
    rally_hit_frames = shots.loc[shots["rally"] == rally, "frame_num"].dropna()
    if rally_hit_frames.empty:
        raise ValueError(f"rally {rally} has no labeled shots")
    return int(rally_hit_frames.min()), int(rally_hit_frames.max())


def draw_players(frame: np.ndarray, detections: FrameDetections, frame_rows: list[dict]) -> np.ndarray:
    drawn_frame = frame.copy()
    player_by_track_id = {row["track_id"]: row for row in frame_rows}
    for box, track_id in zip(np.asarray(detections.boxes_xyxy).reshape(-1, 4), detections.track_ids):
        left, top, right, bottom = np.round(box).astype(int)
        row = player_by_track_id.get(int(track_id))
        color = PLAYER_COLOR_BGR[row["player"]] if row else BYSTANDER_COLOR_BGR
        cv2.rectangle(drawn_frame, (left, top), (right, bottom), color, 2)
        cv2.circle(drawn_frame, ((left + right) // 2, bottom), 5, color, -1)
        label = f"{row['player']} #{track_id} ({row['x_m']:.1f}, {row['y_m']:.1f}) m" if row else f"#{track_id}"
        cv2.putText(drawn_frame, label, (left, max(top - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return drawn_frame


def require_video_file(video_path: Path) -> None:
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"video not found: {video_path}")


def check_frame_range(start_frame: int, end_frame: int) -> None:
    if start_frame < 0:
        raise ValueError(f"start frame must be >= 0, got {start_frame}")
    if end_frame < start_frame:
        raise ValueError(f"end frame {end_frame} is before start frame {start_frame}")


def frames_in_range(capture: cv2.VideoCapture, start_frame: int, end_frame: int):
    check_frame_range(start_frame, end_frame)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_read = 0
    for frame_index in range(start_frame, end_frame + 1):
        frame_was_read, frame = capture.read()
        if not frame_was_read:
            break
        frames_read += 1
        yield frame_index, frame
    if frames_read == 0:
        video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        raise ValueError(
            f"no frames could be read from {start_frame} to {end_frame} (the video has {video_frame_count} frames)"
        )


def load_person_tracker(model_path: Path):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"model weights not found: {model_path} "
            f"(download with: cd {model_path.parent} && python -c \"from ultralytics import YOLO; YOLO('{model_path.name}')\")"
        )
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    numpy_print_options = np.get_printoptions()
    try:
        from ultralytics import YOLO

        return YOLO(str(model_path))
    finally:
        np.set_printoptions(**numpy_print_options)


def track_people(tracker, frame: np.ndarray) -> FrameDetections:
    result = tracker.track(
        frame,
        persist=True,
        classes=[PERSON_CLASS_ID],
        iou=DUPLICATE_BOX_MAX_IOU,
        tracker="bytetrack.yaml",
        imgsz=frame.shape[1],
        verbose=False,
    )[0]
    if result.boxes is None or result.boxes.id is None:
        return empty_detections()
    return FrameDetections(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.id.cpu().numpy().astype(int),
        result.boxes.conf.cpu().numpy(),
    )


@dataclass
class TrackingRun:
    positions: pd.DataFrame
    main_view_frames: list[int]
    skipped_frames: list[int]
    overlay_paths: list[Path]


def track_players(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    calibration_file: Path | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    overlay_frames: tuple[int, ...] = (),
    checks_dir: Path = CHECKS_DIR,
) -> TrackingRun:
    check_frame_range(start_frame, end_frame)
    require_video_file(video_path)
    calibration = load_calibration(calibration_file or calibration_path(video_path))
    homography = calibration["pixel_to_court_homography"]
    tracker = load_person_tracker(model_path)
    capture = open_video(Path(video_path))
    state = PlayerTrackingState()
    rows, main_view_frames, skipped_frames, overlay_paths = [], [], [], []
    try:
        for frame_index, frame in frames_in_range(capture, start_frame, end_frame):
            if not is_main_view(court_line_sample_share(frame, homography)):
                skipped_frames.append(frame_index)
                continue
            main_view_frames.append(frame_index)
            detections = track_people(tracker, frame)
            court_positions = court_positions_of_feet(homography, detections.boxes_xyxy)
            frame_rows, state = label_players(frame_index, detections, court_positions, state)
            rows.extend(frame_rows)
            if frame_index in overlay_frames:
                checks_dir.mkdir(parents=True, exist_ok=True)
                overlay_path = checks_dir / f"{Path(video_path).stem}_players_frame{frame_index}.jpg"
                cv2.imwrite(str(overlay_path), draw_players(frame, detections, frame_rows))
                overlay_paths.append(overlay_path)
    finally:
        capture.release()
    return TrackingRun(pd.DataFrame(rows, columns=POSITION_COLUMNS), main_view_frames, skipped_frames, overlay_paths)


def split_frames_by_view(
    video_path: Path, start_frame: int, end_frame: int, calibration_file: Path | None = None
) -> tuple[list[int], list[int]]:
    check_frame_range(start_frame, end_frame)
    require_video_file(video_path)
    homography = load_calibration(calibration_file or calibration_path(video_path))["pixel_to_court_homography"]
    capture = open_video(Path(video_path))
    main_view_frames, skipped_frames = [], []
    try:
        for frame_index, frame in frames_in_range(capture, start_frame, end_frame):
            if is_main_view(court_line_sample_share(frame, homography)):
                main_view_frames.append(frame_index)
            else:
                skipped_frames.append(frame_index)
    finally:
        capture.release()
    return main_view_frames, skipped_frames


def positions_path(video_path: Path, start_frame: int, end_frame: int, players_dir: Path = PLAYERS_DIR) -> Path:
    return players_dir / Path(video_path).stem / f"positions_{start_frame}_{end_frame}.csv"


def save_positions(positions: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions.to_csv(output_path, index=False)
    return output_path


def summary_lines(run: TrackingRun, measured_frames: list[int]) -> list[str]:
    total_frames = len(run.main_view_frames) + len(run.skipped_frames)
    lines = [f"frames: {total_frames}, main view: {len(run.main_view_frames)}, skipped (not main view): {len(run.skipped_frames)}"]
    for player in PLAYERS:
        continuity = id_continuity(run.positions, player, measured_frames)
        lines.append(
            f"{player}: detected {continuity['frames_detected']}/{continuity['frames']} frames, "
            f"{continuity['track_ids']} track ids, one id (#{continuity['modal_track_id']}) "
            f"{continuity['frames_with_modal_id']}/{continuity['frames']} = {continuity['share_with_one_id']:.1%}, "
            f"in own half {share_in_own_half(run.positions, player):.1%}"
        )
    return lines


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vision.players",
        description="Track both players in a video and write their court positions in meters.",
    )
    parser.add_argument("video", type=Path, help="path to the match video")
    parser.add_argument("--start", type=int, help="first frame to track")
    parser.add_argument("--end", type=int, help="last frame to track (inclusive)")
    parser.add_argument(
        "--shuttleset-rally",
        nargs=3,
        metavar=("MATCH_FOLDER", "SET_NO", "RALLY"),
        help="track one labeled rally: its first to last hit frame, plus --padding frames each side",
    )
    parser.add_argument("--padding", type=int, default=30, help="frames tracked before and after a ShuttleSet rally")
    parser.add_argument("--calibration", type=Path, help="court.json (default: data/calibration/<video stem>/court.json)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="YOLO weights under data/models/")
    parser.add_argument("--views-only", action="store_true", help="only count main-view and skipped frames (no YOLO)")
    parser.add_argument("--overlay-frame", type=int, action="append", default=[], help="save an overlay image of this frame")
    return parser.parse_args(arguments)


def frames_to_track(parsed: argparse.Namespace) -> tuple[int, int, list[int]]:
    if parsed.shuttleset_rally:
        match_folder, set_no, rally = parsed.shuttleset_rally
        first_hit, last_hit = rally_frame_range(load_set_shots(SHUTTLESET_SET_DIR, match_folder, int(set_no)), int(rally))
        return max(first_hit - parsed.padding, 0), last_hit + parsed.padding, list(range(first_hit, last_hit + 1))
    if parsed.start is None or parsed.end is None:
        raise ValueError("give --start and --end, or --shuttleset-rally")
    return parsed.start, parsed.end, list(range(parsed.start, parsed.end + 1))


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        start_frame, end_frame, rally_frames = frames_to_track(parsed)
        if parsed.views_only:
            main_view_frames, skipped_frames = split_frames_by_view(parsed.video, start_frame, end_frame, parsed.calibration)
            print(f"frames: {len(main_view_frames) + len(skipped_frames)}, main view: {len(main_view_frames)}, skipped (not main view): {len(skipped_frames)}")
            return 0
        run = track_players(
            parsed.video, start_frame, end_frame, parsed.calibration, parsed.model, tuple(parsed.overlay_frame)
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    output_path = save_positions(run.positions, positions_path(parsed.video, start_frame, end_frame))
    main_view_frames = set(run.main_view_frames)
    measured_frames = [frame for frame in rally_frames if frame in main_view_frames]
    for line in summary_lines(run, measured_frames):
        print(line)
    print(f"saved {output_path}")
    for overlay_path in run.overlay_paths:
        print(f"saved {overlay_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
