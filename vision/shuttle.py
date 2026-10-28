import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from vision.alignment import SHUTTLESET_SET_DIR, load_set_shots, open_video
from vision.players import check_frame_range, frames_in_range, rally_frame_range, require_video_file
from vision.tracknet import load_inpaintnet, load_tracknet

CHECKPOINT_DIR = Path("data/models/tracknetv3")
TRACKNET_CHECKPOINT = CHECKPOINT_DIR / "TrackNet_best.pt"
INPAINTNET_CHECKPOINT = CHECKPOINT_DIR / "InpaintNet_best.pt"
MODEL_INPUT_WIDTH = 512
MODEL_INPUT_HEIGHT = 288
HEATMAP_THRESHOLD = 0.5
INVISIBLE_COORDINATE_THRESHOLD = 50 / math.sqrt(MODEL_INPUT_HEIGHT**2 + MODEL_INPUT_WIDTH**2)
INPAINT_MIN_HEIGHT_SHARE = 0.05
BACKGROUND_SAMPLE_COUNT = 300
WINDOW_BATCH_SIZE = 8
ENSEMBLE_MODES = ("weighted", "nonoverlap")
SHUTTLE_DIR = Path("data/shuttle")
CHECKS_DIR = Path("data/checks/shuttle")
POSITION_COLUMNS = ["frame", "x", "y", "visibility", "inpainted"]
TRAIL_COLOR_BGR = (0, 255, 255)
CURRENT_COLOR_BGR = (0, 0, 255)


def model_input_image(frame_bgr: np.ndarray) -> np.ndarray:
    rgb_image = Image.fromarray(np.ascontiguousarray(frame_bgr[..., ::-1]))
    resized = np.asarray(rgb_image.resize((MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)), dtype=np.uint8)
    return np.ascontiguousarray(np.moveaxis(resized, -1, 0))


def median_background(frames_bgr: list[np.ndarray]) -> np.ndarray:
    median_frame = np.median(np.stack(frames_bgr), axis=0).astype(np.uint8)
    return model_input_image(median_frame)


def tracknet_window_input(background: np.ndarray, window_images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([background, *window_images], axis=0).astype(np.float32) / 255.0


def ensemble_weights(sequence_length: int) -> np.ndarray:
    position = np.arange(sequence_length)
    weights = np.minimum(position, sequence_length - 1 - position) + 1.0
    return weights / weights.sum()


def check_ensemble_mode(ensemble_mode: str) -> None:
    if ensemble_mode not in ENSEMBLE_MODES:
        raise ValueError(f"ensemble mode must be one of {ENSEMBLE_MODES}, got {ensemble_mode!r}")


def window_starts(frame_count: int, sequence_length: int, ensemble_mode: str) -> list[int]:
    check_ensemble_mode(ensemble_mode)
    last_start = max(frame_count - sequence_length, 0)
    if ensemble_mode == "weighted":
        return list(range(last_start + 1))
    starts = list(range(0, last_start + 1, sequence_length))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def padded_window_frame_indices(start: int, sequence_length: int, frame_count: int) -> list[int]:
    return [min(start + offset, frame_count - 1) for offset in range(sequence_length)]


def heatmap_position(heatmap: np.ndarray, frame_width: int, frame_height: int) -> tuple[int, int, int]:
    binary = (np.asarray(heatmap) > HEATMAP_THRESHOLD).astype(np.uint8) * 255
    if not binary.any():
        return 0, 0, 0
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(contour) for contour in contours]
    left, top, width, height = max(boxes, key=lambda box: box[2] * box[3])
    centre_x, centre_y = int(left + width / 2), int(top + height / 2)
    x = int(centre_x * frame_width / heatmap.shape[1])
    y = int(centre_y * frame_height / heatmap.shape[0])
    return x, y, int(not (x == 0 and y == 0))


def inpaint_mask(y_values, visibility, height_threshold: float) -> np.ndarray:
    y_array, visible = np.asarray(y_values, dtype=float), np.asarray(visibility, dtype=int)
    mask = np.zeros(len(visible), dtype=int)
    gap_start = 0
    while gap_start < len(visible):
        while gap_start < len(visible) - 1 and visible[gap_start] == 1:
            gap_start += 1
        gap_end = gap_start
        while gap_end < len(visible) - 1 and visible[gap_end] == 0:
            gap_end += 1
        if gap_end == gap_start:
            break
        if gap_start == 0 and y_array[gap_end] > height_threshold:
            mask[:gap_end] = 1
        elif gap_start > 1 and y_array[gap_start - 1] > height_threshold and y_array[gap_end] > height_threshold:
            mask[gap_start:gap_end] = 1
        gap_start = gap_end
    return mask


def coordinates_below_threshold_to_zero(normalized_coordinates: np.ndarray) -> np.ndarray:
    cleaned = np.array(normalized_coordinates, dtype=np.float32)
    invisible = (cleaned[..., 0] < INVISIBLE_COORDINATE_THRESHOLD) & (cleaned[..., 1] < INVISIBLE_COORDINATE_THRESHOLD)
    cleaned[invisible] = 0.0
    return cleaned


def weighted_frame_average(window_values: np.ndarray, starts: list[int], frame_count: int, weights: np.ndarray) -> np.ndarray:
    sequence_length = window_values.shape[1]
    totals = np.zeros((frame_count, *window_values.shape[2:]), dtype=np.float64)
    weight_sums = np.zeros(frame_count, dtype=np.float64)
    for window_index, start in enumerate(starts):
        for offset in range(sequence_length):
            frame = start + offset
            if frame < frame_count:
                totals[frame] += weights[offset] * window_values[window_index, offset]
                weight_sums[frame] += weights[offset]
    weight_sums = weight_sums.reshape(-1, *([1] * (totals.ndim - 1)))
    return (totals / np.maximum(weight_sums, 1e-12)).astype(np.float32)


class HeatmapEnsemble:
    def __init__(self, frame_count: int, sequence_length: int, ensemble_mode: str):
        self.frame_count = frame_count
        self.weights = ensemble_weights(sequence_length) if ensemble_mode == "weighted" else np.ones(sequence_length)
        self.totals: dict[int, np.ndarray] = {}
        self.weight_sums: dict[int, float] = {}

    def add_window(self, start: int, heatmaps: np.ndarray) -> None:
        for offset, heatmap in enumerate(heatmaps):
            frame = start + offset
            if frame >= self.frame_count:
                continue
            self.totals[frame] = self.totals.get(frame, 0.0) + self.weights[offset] * heatmap
            self.weight_sums[frame] = self.weight_sums.get(frame, 0.0) + self.weights[offset]

    def pop_frames_before(self, frame_limit: int) -> list[tuple[int, np.ndarray]]:
        finished = sorted(frame for frame in self.totals if frame < frame_limit)
        return [(frame, self.totals.pop(frame) / self.weight_sums.pop(frame)) for frame in finished]


def sampled_frame_indices(start_frame: int, end_frame: int, sample_count: int) -> list[int]:
    return sorted(set(np.linspace(start_frame, end_frame, num=min(sample_count, end_frame - start_frame + 1)).round().astype(int).tolist()))


def read_model_input_images(video_path: Path, start_frame: int, end_frame: int) -> tuple[list[np.ndarray], tuple[int, int]]:
    capture = open_video(Path(video_path))
    try:
        images = [model_input_image(frame) for _, frame in frames_in_range(capture, start_frame, end_frame)]
        frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    return images, frame_size


def read_background(video_path: Path, start_frame: int, end_frame: int) -> np.ndarray:
    capture = open_video(Path(video_path))
    sampled_frames = []
    try:
        for frame_index in sampled_frame_indices(start_frame, end_frame, BACKGROUND_SAMPLE_COUNT):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            frame_was_read, frame = capture.read()
            if frame_was_read:
                sampled_frames.append(frame)
    finally:
        capture.release()
    return median_background(sampled_frames)


def tracknet_positions(
    images: list[np.ndarray],
    frame_size: tuple[int, int],
    background: np.ndarray,
    tracknet,
    sequence_length: int,
    ensemble_mode: str,
) -> pd.DataFrame:
    frame_width, frame_height = frame_size
    starts = window_starts(len(images), sequence_length, ensemble_mode)
    ensemble = HeatmapEnsemble(len(images), sequence_length, ensemble_mode)
    rows = []
    for batch_begin in range(0, len(starts), WINDOW_BATCH_SIZE):
        batch_starts = starts[batch_begin : batch_begin + WINDOW_BATCH_SIZE]
        batch_input = np.stack(
            [
                tracknet_window_input(
                    background, [images[index] for index in padded_window_frame_indices(start, sequence_length, len(images))]
                )
                for start in batch_starts
            ]
        )
        with torch.no_grad():
            batch_heatmaps = tracknet(torch.from_numpy(batch_input)).numpy()
        for start, heatmaps in zip(batch_starts, batch_heatmaps):
            ensemble.add_window(start, heatmaps)
        next_start = starts[batch_begin + WINDOW_BATCH_SIZE] if batch_begin + WINDOW_BATCH_SIZE < len(starts) else len(images)
        for frame, heatmap in ensemble.pop_frames_before(next_start):
            x, y, visibility = heatmap_position(heatmap, frame_width, frame_height)
            rows.append({"frame": frame, "x": x, "y": y, "visibility": visibility})
    return pd.DataFrame(rows, columns=["frame", "x", "y", "visibility"])


def inpainted_positions(
    positions: pd.DataFrame, frame_width: int, frame_height: int, inpaintnet, sequence_length: int
) -> pd.DataFrame:
    frame_count = len(positions)
    mask = inpaint_mask(positions["y"], positions["visibility"], frame_height * INPAINT_MIN_HEIGHT_SHARE).astype(np.float32)
    normalized = np.column_stack([positions["x"] / frame_width, positions["y"] / frame_height]).astype(np.float32)
    starts = window_starts(frame_count, sequence_length, "weighted")
    window_indices = [padded_window_frame_indices(start, sequence_length, frame_count) for start in starts]
    window_coordinates = torch.from_numpy(np.stack([normalized[indices] for indices in window_indices]))
    window_masks = torch.from_numpy(np.stack([mask[indices] for indices in window_indices])[..., None])
    with torch.no_grad():
        predicted = inpaintnet(window_coordinates, window_masks)
    mixed = (predicted * window_masks + window_coordinates * (1 - window_masks)).numpy()
    averaged = weighted_frame_average(
        coordinates_below_threshold_to_zero(mixed), starts, frame_count, ensemble_weights(sequence_length)
    )
    averaged = coordinates_below_threshold_to_zero(averaged)
    x = (averaged[:, 0] * frame_width).astype(int)
    y = (averaged[:, 1] * frame_height).astype(int)
    return pd.DataFrame(
        {
            "frame": positions["frame"].to_numpy(),
            "x": x,
            "y": y,
            "visibility": ((x != 0) | (y != 0)).astype(int),
            "inpainted": mask.astype(int),
        },
        columns=POSITION_COLUMNS,
    )


def track_shuttle(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    ensemble_mode: str = "weighted",
    tracknet_checkpoint: Path = TRACKNET_CHECKPOINT,
    inpaintnet_checkpoint: Path = INPAINTNET_CHECKPOINT,
) -> pd.DataFrame:
    check_frame_range(start_frame, end_frame)
    check_ensemble_mode(ensemble_mode)
    require_video_file(video_path)
    tracknet, tracknet_sequence_length = load_tracknet(tracknet_checkpoint)
    inpaintnet, inpaintnet_sequence_length = load_inpaintnet(inpaintnet_checkpoint)
    images, frame_size = read_model_input_images(video_path, start_frame, end_frame)
    background = read_background(video_path, start_frame, start_frame + len(images) - 1)
    positions = tracknet_positions(images, frame_size, background, tracknet, tracknet_sequence_length, ensemble_mode)
    positions = inpainted_positions(positions, *frame_size, inpaintnet, inpaintnet_sequence_length)
    positions["frame"] += start_frame
    return positions


def draw_shuttle_trail(frame: np.ndarray, trail_points: list[tuple[int, int]]) -> np.ndarray:
    drawn_frame = frame.copy()
    for start_point, end_point in zip(trail_points, trail_points[1:]):
        cv2.line(drawn_frame, start_point, end_point, TRAIL_COLOR_BGR, 2, cv2.LINE_AA)
    if trail_points:
        cv2.circle(drawn_frame, trail_points[-1], 8, CURRENT_COLOR_BGR, 2, cv2.LINE_AA)
    return drawn_frame


def visible_trail(positions: pd.DataFrame, frame_index: int, trail_length: int) -> list[tuple[int, int]]:
    recent = positions[(positions["frame"] > frame_index - trail_length) & (positions["frame"] <= frame_index)]
    recent = recent[recent["visibility"] == 1]
    return [(int(x), int(y)) for x, y in zip(recent["x"], recent["y"])]


def write_overlay_video(video_path: Path, positions: pd.DataFrame, output_path: Path, trail_length: int = 8) -> Path:
    capture = open_video(Path(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    try:
        for frame_index, frame in frames_in_range(capture, int(positions["frame"].min()), int(positions["frame"].max())):
            writer.write(draw_shuttle_trail(frame, visible_trail(positions, frame_index, trail_length)))
    finally:
        capture.release()
        writer.release()
    return output_path


def positions_path(video_path: Path, start_frame: int, end_frame: int, shuttle_dir: Path = SHUTTLE_DIR) -> Path:
    return shuttle_dir / Path(video_path).stem / f"shuttle_{start_frame}_{end_frame}.csv"


def overlay_path(video_path: Path, start_frame: int, end_frame: int, checks_dir: Path = CHECKS_DIR) -> Path:
    return checks_dir / f"{Path(video_path).stem}_shuttle_{start_frame}_{end_frame}.mp4"


def save_positions(positions: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions.to_csv(output_path, index=False)
    return output_path


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vision.shuttle",
        description="Track the shuttle in a video with TrackNetV3 and write its pixel position per frame.",
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
    parser.add_argument("--ensemble", choices=ENSEMBLE_MODES, default="weighted", help="weighted: every frame from 8 windows (slow); nonoverlap: 8x faster")
    parser.add_argument("--overlay", action="store_true", help="also write an overlay video to data/checks/shuttle/")
    return parser.parse_args(arguments)


def frames_to_track(parsed: argparse.Namespace) -> tuple[int, int]:
    if parsed.shuttleset_rally:
        match_folder, set_no, rally = parsed.shuttleset_rally
        first_hit, last_hit = rally_frame_range(load_set_shots(SHUTTLESET_SET_DIR, match_folder, int(set_no)), int(rally))
        return max(first_hit - parsed.padding, 0), last_hit + parsed.padding
    if parsed.start is None or parsed.end is None:
        raise ValueError("give --start and --end, or --shuttleset-rally")
    return parsed.start, parsed.end


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        start_frame, end_frame = frames_to_track(parsed)
        positions = track_shuttle(parsed.video, start_frame, end_frame, parsed.ensemble)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    output_path = save_positions(positions, positions_path(parsed.video, start_frame, end_frame))
    print(
        f"frames: {len(positions)}, visible: {int(positions['visibility'].sum())}, "
        f"inpainted: {int(positions['inpainted'].sum())}"
    )
    print(f"saved {output_path}")
    if parsed.overlay:
        print(f"saved {write_overlay_video(parsed.video, positions, overlay_path(parsed.video, start_frame, end_frame))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
