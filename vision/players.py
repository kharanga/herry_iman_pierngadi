import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Iterable, Iterator

import cv2
import numpy as np
import pandas as pd
import torch

from vision.alignment import SHUTTLESET_SET_DIR, load_set_shots, open_video, read_frame
from vision.court import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NET_Y_M,
    calibration_path,
    court_line_segments,
    line_offsets_to_bright_pixels,
    line_sample_counts,
    load_calibration,
    pixels_to_court,
    video_id,
)

PLAYERS = ("me", "opp")
PERSON_CLASS_ID = 0
COURT_MARGIN_M = 1.0
BEHIND_FAR_BASELINE_MARGIN_M = 2.0
INSIDE_LINES_MARGIN_M = 0.3
NET_HYSTERESIS_M = 0.5
STILL_RADIUS_M = 0.4
STILL_LIMIT_SECONDS = 2.0
STILL_LIMIT_FRAMES_AT_30_FPS = 60
TRACK_MEMORY_SECONDS = 0.5
TRACK_MEMORY_FRAMES_AT_30_FPS = 15
SIDE_EDGE_PX = 4
PASSER_HEIGHT_RATIO = 1.6
HEIGHT_MEMORY = 0.05
DUPLICATE_BOX_MAX_IOU = 0.5
MAX_DETECTION_WIDTH_PX = 1280
DETECTION_BATCH_FRAMES = 8
PREFETCHED_CHUNKS = 2
GPU_DEVICE = "mps"
CPU_DEVICE = "cpu"
LINE_SAMPLES_PER_LINE = 40
MAIN_VIEW_MIN_LINE_SAMPLE_SHARE = 0.6
MAIN_VIEW_SHARE_OF_CALIBRATION_FRAME = 0.6
DEFAULT_MODEL_PATH = Path("data/models/yolo11s.pt")
PLAYERS_DIR = Path("data/players")
TRACKING_RECORD_SUFFIX = ".tracking.json"
CHECKS_DIR = Path("data/checks/players")
POSITION_COLUMNS = ["frame", "player", "x_m", "y_m", "track_id", "confidence", "doubtful"]
DETECTION_COLUMNS = ["frame", "track_id", "confidence", "left", "top", "right", "bottom", "x_m", "y_m"]
BOX_COLUMNS = ["frame", "player", "left", "top", "right", "bottom"]
PLAYER_COLOR_BGR = {"me": (255, 160, 0), "opp": (0, 0, 255)}
BYSTANDER_COLOR_BGR = (160, 160, 160)


@dataclass(frozen=True)
class FrameDetections:
    boxes_xyxy: np.ndarray
    track_ids: np.ndarray
    confidences: np.ndarray


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
    counts = [line_sample_counts(frame, homography, start, end, sample_count=LINE_SAMPLES_PER_LINE) for start, end in floor_lines]
    samples_in_frame = sum(inside for _, inside in counts)
    return sum(found for found, _ in counts) / samples_in_frame if samples_in_frame else 0.0


def main_view_min_share(calibration_frame_share: float, share_of_calibration_frame: float = MAIN_VIEW_SHARE_OF_CALIBRATION_FRAME) -> float:
    return share_of_calibration_frame * calibration_frame_share


def calibration_frame_share(video_path: Path, calibration: dict) -> float:
    capture = open_video(Path(video_path))
    try:
        frame = read_frame(capture, int(calibration["frame_index"]))
    finally:
        capture.release()
    if frame is None:
        raise ValueError(f"could not read the calibration frame {calibration['frame_index']} of {video_path}")
    return court_line_sample_share(frame, calibration["pixel_to_court_homography"])


def is_main_view(line_sample_share: float, minimum_share: float = MAIN_VIEW_MIN_LINE_SAMPLE_SHARE) -> bool:
    return line_sample_share >= minimum_share


def frames_for_seconds(seconds: float, fps: float) -> int:
    return max(1, int(round(seconds * fps)))


def inside_lines_mask(court_positions, margin_m: float = INSIDE_LINES_MARGIN_M) -> np.ndarray:
    positions = np.asarray(court_positions, dtype=float).reshape(-1, 2)
    return (positions[:, 0] >= -margin_m) & (positions[:, 0] <= COURT_WIDTH_M + margin_m) & (positions[:, 1] >= -margin_m) & (positions[:, 1] <= COURT_LENGTH_M + margin_m)


def side_with_hysteresis(y_m: float, previous_side: str | None, hysteresis_m: float = NET_HYSTERESIS_M) -> str:
    if previous_side is not None and abs(y_m - NET_Y_M) < hysteresis_m:
        return previous_side
    return side_of_court(y_m)


def updated_anchors(anchors: dict[int, tuple[float, float, int]], track_ids, court_positions, frame_index: int, radius_m: float = STILL_RADIUS_M) -> dict[int, tuple[float, float, int]]:
    updated = {}
    for track_id, (x_m, y_m) in zip(np.asarray(track_ids, dtype=int), np.asarray(court_positions, dtype=float).reshape(-1, 2)):
        anchor = anchors.get(int(track_id))
        if anchor is None or np.hypot(x_m - anchor[0], y_m - anchor[1]) > radius_m:
            anchor = (float(x_m), float(y_m), int(frame_index))
        updated[int(track_id)] = anchor
    return updated


def still_frame_counts(anchors: dict[int, tuple[float, float, int]], track_ids, frame_index: int) -> np.ndarray:
    return np.array([frame_index - anchors[int(track_id)][2] for track_id in np.asarray(track_ids, dtype=int)], dtype=int)


def box_heights(boxes_xyxy) -> np.ndarray:
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    return boxes[:, 3] - boxes[:, 1]


def at_image_side(boxes_xyxy, frame_width: float | None, edge_px: float = SIDE_EDGE_PX) -> np.ndarray:
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    if frame_width is None:
        return np.zeros(len(boxes), dtype=bool)
    return (boxes[:, 0] <= edge_px) | (boxes[:, 2] >= frame_width - edge_px)


def candidate_rank(inside: bool, still: bool, confidence: float) -> tuple:
    return (bool(inside), not still, float(confidence))


@dataclass(frozen=True)
class PlayerChoice:
    index: int | None
    doubtful: bool


def chosen_candidate(candidates: list[int], previous_index: int | None, inside: np.ndarray, still: np.ndarray, confidences: np.ndarray) -> PlayerChoice:
    if not candidates:
        return PlayerChoice(None, False)
    best = max(candidates, key=lambda index: candidate_rank(inside[index], still[index], confidences[index]))
    someone_inside = any(inside[index] for index in candidates)
    if previous_index is not None and previous_index in candidates:
        stale = not inside[previous_index] and still[previous_index]
        if not (stale and inside[best] and not still[best]):
            return PlayerChoice(previous_index, stale or (not inside[previous_index] and someone_inside))
    return PlayerChoice(best, not inside[best] and still[best])


@dataclass
class PlayerTrackingState:
    side_by_track_id: dict[int, str] = field(default_factory=dict)
    track_id_by_player: dict[str, int] = field(default_factory=dict)
    anchor_by_track_id: dict[int, tuple[float, float, int]] = field(default_factory=dict)
    typical_height_by_player: dict[str, float] = field(default_factory=dict)
    last_seen_by_track_id: dict[int, int] = field(default_factory=dict)


def remembered_tracks(last_seen: dict[int, int], frame_index: int, memory_frames: int) -> set[int]:
    return {track_id for track_id, seen in last_seen.items() if frame_index - seen <= memory_frames}


def label_players(
    frame_index: int,
    detections: FrameDetections,
    court_positions: np.ndarray,
    state: PlayerTrackingState,
    margin_m: float = COURT_MARGIN_M,
    still_limit_frames: int = STILL_LIMIT_FRAMES_AT_30_FPS,
    frame_width: float | None = None,
    memory_frames: int = TRACK_MEMORY_FRAMES_AT_30_FPS,
) -> tuple[list[dict], PlayerTrackingState]:
    track_ids = np.asarray(detections.track_ids, dtype=int)
    court_positions = np.asarray(court_positions, dtype=float).reshape(-1, 2)
    on_court = on_court_mask(court_positions, margin_m)
    inside = inside_lines_mask(court_positions)
    remembered = remembered_tracks(state.last_seen_by_track_id, frame_index, memory_frames)
    kept_anchors = {track_id: anchor for track_id, anchor in state.anchor_by_track_id.items() if track_id in remembered}
    kept_sides = {track_id: side for track_id, side in state.side_by_track_id.items() if track_id in remembered}
    anchors = {**kept_anchors, **updated_anchors(kept_anchors, track_ids[on_court], court_positions[on_court], frame_index)}
    still = np.zeros(len(track_ids), dtype=bool)
    still[on_court] = still_frame_counts(anchors, track_ids[on_court], frame_index) >= still_limit_frames
    sides = {**kept_sides, **{int(track_ids[index]): side_with_hysteresis(float(court_positions[index, 1]), kept_sides.get(int(track_ids[index]))) for index in np.flatnonzero(on_court)}}
    last_seen = {track_id: seen for track_id, seen in state.last_seen_by_track_id.items() if track_id in remembered}
    last_seen.update({int(track_id): int(frame_index) for track_id in track_ids[on_court]})
    heights = box_heights(detections.boxes_xyxy)
    edge = at_image_side(detections.boxes_xyxy, frame_width)
    track_id_by_player = dict(state.track_id_by_player)
    typical_heights = dict(state.typical_height_by_player)
    rows = []
    for player in PLAYERS:
        typical = typical_heights.get(player)
        oversized = [index for index in np.flatnonzero(on_court) if sides[int(track_ids[index])] == player and typical is not None and edge[index] and heights[index] > PASSER_HEIGHT_RATIO * typical]
        candidates = [index for index in np.flatnonzero(on_court) if sides[int(track_ids[index])] == player and index not in oversized]
        previous = next((index for index in candidates if int(track_ids[index]) == track_id_by_player.get(player)), None)
        choice = chosen_candidate(candidates, previous, inside, still, detections.confidences)
        if choice.index is None:
            continue
        index = choice.index
        track_id_by_player[player] = int(track_ids[index])
        typical_heights[player] = float(heights[index]) if typical is None else (1 - HEIGHT_MEMORY) * typical + HEIGHT_MEMORY * float(heights[index])
        rows.append(
            {
                "frame": frame_index,
                "player": player,
                "x_m": round(float(court_positions[index, 0]), 3),
                "y_m": round(float(court_positions[index, 1]), 3),
                "track_id": int(track_ids[index]),
                "confidence": round(float(detections.confidences[index]), 3),
                "doubtful": bool(choice.doubtful or oversized),
            }
        )
    return rows, PlayerTrackingState(sides, track_id_by_player, anchors, typical_heights, last_seen)


def player_box_rows(detections: FrameDetections, frame_rows: list[dict]) -> list[dict]:
    box_by_track_id = {
        int(track_id): box for track_id, box in zip(detections.track_ids, np.asarray(detections.boxes_xyxy).reshape(-1, 4))
    }
    return [
        {
            "frame": row["frame"],
            "player": row["player"],
            **dict(zip(BOX_COLUMNS[2:], (round(float(value), 1) for value in box_by_track_id[row["track_id"]]))),
        }
        for row in frame_rows
    ]


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


def check_frame_stride(frame_stride: int) -> None:
    if frame_stride < 1:
        raise ValueError(f"frame stride must be >= 1, got {frame_stride}")


def sampled_frames(start_frame: int, end_frame: int, frame_stride: int = 1) -> np.ndarray:
    check_frame_stride(frame_stride)
    return np.arange(start_frame, end_frame + 1, frame_stride)


def frames_in_range(capture: cv2.VideoCapture, start_frame: int, end_frame: int, frame_stride: int = 1):
    check_frame_range(start_frame, end_frame)
    check_frame_stride(frame_stride)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_read = 0
    for frame_index in range(start_frame, end_frame + 1):
        if (frame_index - start_frame) % frame_stride:
            if not capture.grab():
                break
            continue
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


def default_device() -> str:
    return GPU_DEVICE if torch.backends.mps.is_available() else CPU_DEVICE


def chunked(items: Iterable, size: int) -> Iterator[list]:
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    chunk = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def prefetched(items: Iterable, depth: int = PREFETCHED_CHUNKS) -> Iterator:
    queue: Queue = Queue(maxsize=depth)
    stop_producing = Event()
    end_of_items = object()

    def produce():
        try:
            for item in items:
                if stop_producing.is_set():
                    break
                queue.put(item)
        except BaseException as error:
            queue.put(error)
        finally:
            queue.put(end_of_items)

    producer = Thread(target=produce, daemon=True)
    producer.start()
    end_seen = False
    try:
        while True:
            item = queue.get()
            if item is end_of_items:
                end_seen = True
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop_producing.set()
        while not end_seen:
            end_seen = queue.get() is end_of_items
        producer.join()


def frames_with_view(frames: Iterable[tuple[int, np.ndarray]], homography: np.ndarray, minimum_share: float) -> Iterator[tuple[int, np.ndarray, bool]]:
    for frame_index, frame in frames:
        yield frame_index, frame, is_main_view(court_line_sample_share(frame, homography), minimum_share)


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


def frame_shape_of_capture(capture: cv2.VideoCapture) -> tuple[int, int, int]:
    return int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 3


def warm_up_tracker(tracker, frame_shape: tuple[int, int, int], device: str = CPU_DEVICE) -> None:
    tracker.predict(
        np.zeros(frame_shape, dtype=np.uint8),
        classes=[PERSON_CLASS_ID],
        imgsz=min(frame_shape[1], MAX_DETECTION_WIDTH_PX),
        device=device,
        verbose=False,
    )


def detections_of_result(result) -> FrameDetections:
    if result.boxes is None or result.boxes.id is None:
        return empty_detections()
    return FrameDetections(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.id.cpu().numpy().astype(int),
        result.boxes.conf.cpu().numpy(),
    )


def track_people(tracker, frames: list[np.ndarray], device: str = CPU_DEVICE) -> list[FrameDetections]:
    if not frames:
        return []
    results = tracker.track(
        list(frames),
        persist=True,
        classes=[PERSON_CLASS_ID],
        iou=DUPLICATE_BOX_MAX_IOU,
        tracker="bytetrack.yaml",
        imgsz=min(frames[0].shape[1], MAX_DETECTION_WIDTH_PX),
        device=device,
        verbose=False,
    )
    return [detections_of_result(result) for result in results]


@dataclass
class TrackingRun:
    positions: pd.DataFrame
    main_view_frames: list[int]
    skipped_frames: list[int]
    overlay_paths: list[Path]
    boxes: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=BOX_COLUMNS))
    detections: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=DETECTION_COLUMNS))


def detection_rows(frame_index: int, detections: FrameDetections, court_positions: np.ndarray) -> list[dict]:
    boxes = np.asarray(detections.boxes_xyxy, dtype=float).reshape(-1, 4)
    positions = np.asarray(court_positions, dtype=float).reshape(-1, 2)
    return [
        {
            "frame": frame_index,
            "track_id": int(track_id),
            "confidence": round(float(confidence), 3),
            **dict(zip(BOX_COLUMNS[2:], (round(float(value), 1) for value in box))),
            "x_m": round(float(position[0]), 3),
            "y_m": round(float(position[1]), 3),
        }
        for track_id, confidence, box, position in zip(detections.track_ids, detections.confidences, boxes, positions)
    ]


def detections_of_frame(table: pd.DataFrame) -> tuple[FrameDetections, np.ndarray]:
    boxes = table[BOX_COLUMNS[2:]].to_numpy(dtype=float).reshape(-1, 4)
    return FrameDetections(boxes, table["track_id"].to_numpy(dtype=int), table["confidence"].to_numpy(dtype=float)), table[["x_m", "y_m"]].to_numpy(dtype=float).reshape(-1, 2)


def players_from_detections(table: pd.DataFrame, fps: float, frame_width: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = PlayerTrackingState()
    still_limit = frames_for_seconds(STILL_LIMIT_SECONDS, fps)
    memory = frames_for_seconds(TRACK_MEMORY_SECONDS, fps)
    rows, box_rows = [], []
    for frame_index, frame_table in table.sort_values(["frame", "track_id"]).groupby("frame", sort=True):
        detections, court_positions = detections_of_frame(frame_table)
        frame_rows, state = label_players(int(frame_index), detections, court_positions, state, still_limit_frames=still_limit, frame_width=frame_width, memory_frames=memory)
        rows.extend(frame_rows)
        box_rows.extend(player_box_rows(detections, frame_rows))
    return pd.DataFrame(rows, columns=POSITION_COLUMNS), pd.DataFrame(box_rows, columns=BOX_COLUMNS)


def track_players(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    calibration_file: Path | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    overlay_frames: tuple[int, ...] = (),
    checks_dir: Path = CHECKS_DIR,
    device: str | None = None,
    batch_frames: int = DETECTION_BATCH_FRAMES,
    frame_stride: int = 1,
    tracker=None,
) -> TrackingRun:
    check_frame_range(start_frame, end_frame)
    check_frame_stride(frame_stride)
    require_video_file(video_path)
    calibration = load_calibration(calibration_file or calibration_path(video_path))
    homography = calibration["pixel_to_court_homography"]
    minimum_share = main_view_min_share(calibration_frame_share(video_path, calibration))
    tracker = tracker if tracker is not None else load_person_tracker(model_path)
    device = device or default_device()
    capture = open_video(Path(video_path))
    warm_up_tracker(tracker, frame_shape_of_capture(capture), device)
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    still_limit = frames_for_seconds(STILL_LIMIT_SECONDS, video_fps)
    memory = frames_for_seconds(TRACK_MEMORY_SECONDS, video_fps)
    frame_width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    state = PlayerTrackingState()
    rows, box_rows, detection_table_rows, main_view_frames, skipped_frames, overlay_paths = [], [], [], [], [], []
    try:
        viewed_frames = frames_with_view(frames_in_range(capture, start_frame, end_frame, frame_stride), homography, minimum_share)
        for chunk in prefetched(chunked(viewed_frames, batch_frames)):
            skipped_frames.extend(frame_index for frame_index, _, in_main_view in chunk if not in_main_view)
            main_view_chunk = [(frame_index, frame) for frame_index, frame, in_main_view in chunk if in_main_view]
            main_view_frames.extend(frame_index for frame_index, _ in main_view_chunk)
            detections_per_frame = track_people(tracker, [frame for _, frame in main_view_chunk], device)
            for (frame_index, frame), detections in zip(main_view_chunk, detections_per_frame):
                court_positions = court_positions_of_feet(homography, detections.boxes_xyxy)
                frame_rows, state = label_players(frame_index, detections, court_positions, state, still_limit_frames=still_limit, frame_width=frame_width, memory_frames=memory)
                detection_table_rows.extend(detection_rows(frame_index, detections, court_positions))
                rows.extend(frame_rows)
                box_rows.extend(player_box_rows(detections, frame_rows))
                if frame_index in overlay_frames:
                    checks_dir.mkdir(parents=True, exist_ok=True)
                    overlay_path = checks_dir / f"{video_id(video_path)}_players_frame{frame_index}.jpg"
                    cv2.imwrite(str(overlay_path), draw_players(frame, detections, frame_rows))
                    overlay_paths.append(overlay_path)
    finally:
        capture.release()
    return TrackingRun(
        pd.DataFrame(rows, columns=POSITION_COLUMNS),
        main_view_frames,
        skipped_frames,
        overlay_paths,
        pd.DataFrame(box_rows, columns=BOX_COLUMNS),
        pd.DataFrame(detection_table_rows, columns=DETECTION_COLUMNS),
    )


def split_frames_by_view(
    video_path: Path, start_frame: int, end_frame: int, calibration_file: Path | None = None
) -> tuple[list[int], list[int]]:
    check_frame_range(start_frame, end_frame)
    require_video_file(video_path)
    calibration = load_calibration(calibration_file or calibration_path(video_path))
    homography = calibration["pixel_to_court_homography"]
    minimum_share = main_view_min_share(calibration_frame_share(video_path, calibration))
    capture = open_video(Path(video_path))
    main_view_frames, skipped_frames = [], []
    try:
        for frame_index, frame in frames_in_range(capture, start_frame, end_frame):
            if is_main_view(court_line_sample_share(frame, homography), minimum_share):
                main_view_frames.append(frame_index)
            else:
                skipped_frames.append(frame_index)
    finally:
        capture.release()
    return main_view_frames, skipped_frames


def positions_path(video_path: Path, start_frame: int, end_frame: int, players_dir: Path = PLAYERS_DIR) -> Path:
    return players_dir / video_id(video_path) / f"positions_{start_frame}_{end_frame}.csv"


def boxes_path(video_path: Path, start_frame: int, end_frame: int, players_dir: Path = PLAYERS_DIR) -> Path:
    return players_dir / video_id(video_path) / f"boxes_{start_frame}_{end_frame}.csv"


def detections_path(video_path: Path, start_frame: int, end_frame: int, players_dir: Path = PLAYERS_DIR) -> Path:
    return players_dir / video_id(video_path) / f"detections_{start_frame}_{end_frame}.csv"


def is_default_dir(directory: Path | None, default: Path) -> bool:
    return directory is None or Path(directory) == Path(default)


def save_positions(positions: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions.to_csv(output_path, index=False)
    return output_path


def tracking_record_path(positions_file: Path) -> Path:
    return Path(positions_file).with_name(Path(positions_file).name + TRACKING_RECORD_SUFFIX)


def tracking_record(video_fps: float, frame_width: float, frame_stride: int, model_path: Path, device: str, batch_frames: int) -> dict:
    return {
        "frame_stride": int(frame_stride),
        "video_fps": float(video_fps),
        "effective_fps": float(video_fps) / int(frame_stride),
        "frame_width": float(frame_width),
        "model": Path(model_path).name,
        "device": device,
        "batch_frames": int(batch_frames),
    }


def write_tracking_record(positions_file: Path, record: dict) -> Path:
    record_path = tracking_record_path(positions_file)
    record_path.write_text(json.dumps(record, indent=1))
    return record_path


def read_tracking_record(positions_file: Path) -> dict:
    record_path = tracking_record_path(positions_file)
    if not record_path.is_file():
        return {"frame_stride": 1}
    return json.loads(record_path.read_text())


def frame_stride_of(positions_file: Path) -> int:
    return int(read_tracking_record(positions_file).get("frame_stride", 1))


def effective_fps(video_fps: float, frame_stride: int) -> float:
    check_frame_stride(frame_stride)
    return float(video_fps) / frame_stride


def video_fps_and_width(video_path: Path) -> tuple[float, float]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        return float(capture.get(cv2.CAP_PROP_FPS)), float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        capture.release()


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


def read_stretches(stretches_file: Path) -> list[tuple[int, int]]:
    table = pd.read_csv(stretches_file)
    if not {"start_frame", "end_frame"} <= set(table.columns):
        raise ValueError(f"{stretches_file} needs start_frame and end_frame columns")
    return [(int(start), int(end)) for start, end in zip(table["start_frame"], table["end_frame"])]


def stretch_is_tracked(video_path: Path, start_frame: int, end_frame: int, players_dir: Path = PLAYERS_DIR) -> bool:
    return positions_path(video_path, start_frame, end_frame, players_dir).is_file() and boxes_path(video_path, start_frame, end_frame, players_dir).is_file()


def track_stretches(
    video_path: Path,
    stretches: list[tuple[int, int]],
    calibration_file: Path | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    device: str | None = None,
    batch_frames: int = DETECTION_BATCH_FRAMES,
    players_dir: Path = PLAYERS_DIR,
) -> tuple[int, int, int]:
    pending = [(start, end) for start, end in stretches if not stretch_is_tracked(video_path, start, end, players_dir)]
    if not pending:
        return 0, len(stretches), 0
    tracker = load_person_tracker(model_path)
    video_fps, frame_width = video_fps_and_width(video_path)
    record = tracking_record(video_fps, frame_width, 1, model_path, device or default_device(), batch_frames)
    frames = 0
    for start_frame, end_frame in pending:
        run = track_players(video_path, start_frame, end_frame, calibration_file, model_path, device=device, batch_frames=batch_frames, tracker=tracker)
        save_positions(run.detections, detections_path(video_path, start_frame, end_frame, players_dir))
        save_positions(run.boxes, boxes_path(video_path, start_frame, end_frame, players_dir))
        write_tracking_record(save_positions(run.positions, positions_path(video_path, start_frame, end_frame, players_dir)), record)
        frames += end_frame - start_frame + 1
    return len(pending), len(stretches) - len(pending), frames


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
    parser.add_argument("--stretches", type=Path, metavar="CSV", help="track every start_frame-end_frame row of this file with one loaded model, skipping rows already tracked (python -m vision.label_offset --stretches writes it)")
    parser.add_argument("--from-detections", action="store_true", help="don't run YOLO: choose the players again from the saved raw detections_<start>_<end>.csv and rewrite the positions and boxes files")
    parser.add_argument("--merge", action="store_true", help="with --stretches: also join the stretches' files into positions_0_<last>.csv and boxes_0_<last>.csv for the whole video (what vision.hits --start 0 and vision.label_ui read)")
    parser.add_argument("--plan-stretches", type=Path, metavar="CSV", help="write a stretches file for this video and exit: back-to-back stretches of --stretch-seconds, or --stretch-count of them spread evenly")
    parser.add_argument("--stretch-seconds", type=float, default=180.0, help="with --plan-stretches: length of each stretch (default 180)")
    parser.add_argument("--stretch-count", type=int, help="with --plan-stretches: this many stretches, one centred in each equal slice of the video")
    parser.add_argument("--padding", type=int, default=30, help="frames tracked before and after a ShuttleSet rally")
    parser.add_argument("--calibration", type=Path, help="court.json (default: data/calibration/<video stem>/court.json)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="YOLO weights under data/models/")
    parser.add_argument("--views-only", action="store_true", help="only count main-view and skipped frames (no YOLO)")
    parser.add_argument("--overlay-frame", type=int, action="append", default=[], help="save an overlay image of this frame")
    parser.add_argument("--device", choices=(GPU_DEVICE, CPU_DEVICE), help="where YOLO runs (default: mps when available, else cpu)")
    parser.add_argument("--batch-frames", type=int, default=DETECTION_BATCH_FRAMES, help="frames per YOLO call (1 = one frame at a time)")
    parser.add_argument("--frame-stride", type=int, default=1, help="track every Nth frame (default 1 = every frame); the files keep the video's frame numbers")
    parser.add_argument("--players-dir", type=Path, default=PLAYERS_DIR, help="root of the output files (default data/players)")
    return parser.parse_args(arguments)


def frames_to_track(parsed: argparse.Namespace) -> tuple[int, int, list[int]]:
    if parsed.shuttleset_rally:
        match_folder, set_no, rally = parsed.shuttleset_rally
        first_hit, last_hit = rally_frame_range(load_set_shots(SHUTTLESET_SET_DIR, match_folder, int(set_no)), int(rally))
        return max(first_hit - parsed.padding, 0), last_hit + parsed.padding, list(range(first_hit, last_hit + 1))
    if parsed.start is None or parsed.end is None:
        raise ValueError("give --start and --end, or --shuttleset-rally")
    return parsed.start, parsed.end, list(range(parsed.start, parsed.end + 1))


def even_stretches(frame_count: int, stretch_frames: int, stretch_count: int | None = None) -> list[tuple[int, int]]:
    if frame_count < 1 or stretch_frames < 1:
        raise ValueError(f"need a positive frame count and stretch length, got {frame_count} and {stretch_frames}")
    if stretch_count is None:
        return [(start, min(start + stretch_frames, frame_count) - 1) for start in range(0, frame_count, stretch_frames)]
    if stretch_count < 1 or stretch_count * stretch_frames > frame_count:
        raise ValueError(f"{stretch_count} stretches of {stretch_frames} frames don't fit in {frame_count} frames")
    slot_frames = frame_count / stretch_count
    starts = [int(round(slot * slot_frames + (slot_frames - stretch_frames) / 2)) for slot in range(stretch_count)]
    return [(start, start + stretch_frames - 1) for start in starts]


def write_stretches(stretches: list[tuple[int, int]], stretches_file: Path) -> Path:
    stretches_file = Path(stretches_file)
    stretches_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stretches, columns=["start_frame", "end_frame"]).to_csv(stretches_file, index=False)
    return stretches_file


def video_last_frame(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    finally:
        capture.release()


def merged_tables(tables: list[pd.DataFrame], keys: tuple[str, str] = ("frame", "player")) -> pd.DataFrame:
    merged = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if merged.empty:
        return merged
    return merged.drop_duplicates(subset=list(keys), keep="first").sort_values(list(keys), kind="stable").reset_index(drop=True)


def merge_stretch_tracks(video_path: Path, stretches: list[tuple[int, int]], last_frame: int, players_dir: Path = PLAYERS_DIR) -> tuple[Path, Path]:
    missing = [(start, end) for start, end in stretches if not stretch_is_tracked(video_path, start, end, players_dir)]
    if missing:
        raise ValueError(f"{len(missing)} stretches are not tracked yet, first {missing[0][0]}-{missing[0][1]}")
    records = [read_tracking_record(positions_path(video_path, start, end, players_dir)) for start, end in stretches]
    rates = sorted({(int(record.get("frame_stride", 1)), record.get("effective_fps")) for record in records}, key=str)
    if len(rates) > 1:
        raise ValueError(f"the stretches were tracked at different frame strides / effective fps {rates}; retrack the odd ones before merging")
    positions = merged_tables([pd.read_csv(positions_path(video_path, start, end, players_dir)) for start, end in stretches])
    boxes = merged_tables([pd.read_csv(boxes_path(video_path, start, end, players_dir)) for start, end in stretches])
    merged_positions = save_positions(positions, positions_path(video_path, 0, last_frame, players_dir))
    merged_boxes = save_positions(boxes, boxes_path(video_path, 0, last_frame, players_dir))
    write_tracking_record(merged_positions, {**records[0], "stretches": [[int(start), int(end)] for start, end in stretches]})
    detection_files = [detections_path(video_path, start, end, players_dir) for start, end in stretches]
    if all(path.is_file() for path in detection_files):
        save_positions(merged_tables([pd.read_csv(path) for path in detection_files], ("frame", "track_id")), detections_path(video_path, 0, last_frame, players_dir))
    return merged_positions, merged_boxes


def rechoose_command(parsed: argparse.Namespace) -> int:
    try:
        if parsed.start is None or parsed.end is None:
            raise ValueError("--from-detections needs --start and --end")
        source = detections_path(parsed.video, parsed.start, parsed.end, parsed.players_dir)
        if not source.is_file():
            raise FileNotFoundError(f"detections file not found: {source} (tracked before raw detections were saved; track again)")
        fps, frame_width = video_fps_and_width(parsed.video)
        positions, boxes = players_from_detections(pd.read_csv(source), fps, frame_width)
        positions_file = save_positions(positions, positions_path(parsed.video, parsed.start, parsed.end, parsed.players_dir))
        print(f"saved {positions_file} ({int(positions['doubtful'].sum())} doubtful rows of {len(positions)})")
        print(f"saved {save_positions(boxes, boxes_path(parsed.video, parsed.start, parsed.end, parsed.players_dir))}")
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def stretches_command(parsed: argparse.Namespace) -> int:
    try:
        stretches = read_stretches(parsed.stretches)
        tracked, skipped, frames = track_stretches(parsed.video, stretches, parsed.calibration, parsed.model, parsed.device, parsed.batch_frames, parsed.players_dir)
        print(f"{len(stretches)} stretches: {tracked} tracked ({frames} frames), {skipped} already tracked")
        if parsed.merge:
            for saved in merge_stretch_tracks(parsed.video, stretches, video_last_frame(parsed.video), parsed.players_dir):
                print(f"saved {saved}")
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def plan_command(parsed: argparse.Namespace) -> int:
    try:
        require_video_file(parsed.video)
        fps, _ = video_fps_and_width(parsed.video)
        stretch_frames = int(round(parsed.stretch_seconds * fps))
        stretches = even_stretches(video_last_frame(parsed.video) + 1, stretch_frames, parsed.stretch_count)
        print(f"saved {write_stretches(stretches, parsed.plan_stretches)} ({len(stretches)} stretches of up to {stretch_frames} frames)")
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    if parsed.merge and not parsed.stretches:
        print("error: --merge only works with --stretches CSV", file=sys.stderr)
        return 2
    if parsed.plan_stretches:
        return plan_command(parsed)
    if parsed.from_detections:
        return rechoose_command(parsed)
    if parsed.stretches:
        return stretches_command(parsed)
    try:
        start_frame, end_frame, rally_frames = frames_to_track(parsed)
        if parsed.views_only:
            main_view_frames, skipped_frames = split_frames_by_view(parsed.video, start_frame, end_frame, parsed.calibration)
            print(f"frames: {len(main_view_frames) + len(skipped_frames)}, main view: {len(main_view_frames)}, skipped (not main view): {len(skipped_frames)}")
            return 0
        run = track_players(
            parsed.video,
            start_frame,
            end_frame,
            parsed.calibration,
            parsed.model,
            tuple(parsed.overlay_frame),
            device=parsed.device,
            batch_frames=parsed.batch_frames,
            frame_stride=parsed.frame_stride,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    custom_dir = () if is_default_dir(parsed.players_dir, PLAYERS_DIR) else (parsed.players_dir,)
    output_path = save_positions(run.positions, positions_path(parsed.video, start_frame, end_frame, *custom_dir))
    main_view_frames = set(run.main_view_frames)
    measured_frames = [frame for frame in rally_frames if frame in main_view_frames]
    for line in summary_lines(run, measured_frames):
        print(line)
    print(f"saved {output_path}")
    print(f"saved {save_positions(run.boxes, boxes_path(parsed.video, start_frame, end_frame, *custom_dir))}")
    print(f"saved {save_positions(run.detections, detections_path(parsed.video, start_frame, end_frame, *custom_dir))}")
    video_fps, frame_width = video_fps_and_width(parsed.video)
    record = tracking_record(video_fps, frame_width, parsed.frame_stride, parsed.model, parsed.device or default_device(), parsed.batch_frames)
    print(f"saved {write_tracking_record(output_path, record)} (frame stride {parsed.frame_stride}, effective fps {record['effective_fps']:g})")
    for overlay_path in run.overlay_paths:
        print(f"saved {overlay_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
