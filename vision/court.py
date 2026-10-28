import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vision.alignment import label_scale, open_video, read_frame

COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
NET_Y_M = COURT_LENGTH_M / 2
SINGLES_SIDELINE_INSET_M = 0.46
SHORT_SERVICE_LINE_FROM_NET_M = 1.98
DOUBLES_LONG_SERVICE_LINE_FROM_BASELINE_M = 0.76
CORNER_NAMES = ("near_left", "near_right", "far_right", "far_left")
CALIBRATION_DIR = Path("data/calibration")
SHUTTLESET_HOMOGRAPHY_CSV = Path("data/CoachAI-Projects/ShuttleSet/set/homography.csv")
LINE_COLOR_BGR = (0, 0, 255)
CORNER_COLOR_BGR = (0, 255, 255)


def court_corner_meters() -> np.ndarray:
    return np.array(
        [[0.0, 0.0], [COURT_WIDTH_M, 0.0], [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]]
    )


def as_corner_array(corner_pixels) -> np.ndarray:
    corners = np.asarray(corner_pixels, dtype=float)
    if corners.shape != (4, 2):
        raise ValueError(f"expected 4 corners as (x, y) pixels, got shape {corners.shape}")
    if not np.all(np.isfinite(corners)):
        raise ValueError("corner pixels must be finite numbers")
    return corners


def homography_between(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    equations = []
    targets = []
    for (source_x, source_y), (target_x, target_y) in zip(source_points, target_points):
        equations.append([source_x, source_y, 1, 0, 0, 0, -target_x * source_x, -target_x * source_y])
        equations.append([0, 0, 0, source_x, source_y, 1, -target_y * source_x, -target_y * source_y])
        targets.extend([target_x, target_y])
    solution = np.linalg.solve(np.array(equations, dtype=float), np.array(targets, dtype=float))
    return np.append(solution, 1.0).reshape(3, 3)


def turn_directions(corners: np.ndarray) -> np.ndarray:
    edges = np.roll(corners, -1, axis=0) - corners
    next_edges = np.roll(edges, -1, axis=0)
    return edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]


def check_corners_form_convex_quadrilateral(corners: np.ndarray) -> None:
    turns = turn_directions(corners)
    smallest_allowed_turn = 1e-6 * max(float(np.ptp(corners, axis=0).max()), 1.0) ** 2
    if np.any(np.abs(turns) <= smallest_allowed_turn):
        raise ValueError("corners are degenerate: three of them lie on one line or two are the same point")
    if not (np.all(turns > 0) or np.all(turns < 0)):
        raise ValueError(f"corners do not form a convex quadrilateral; give them in order {', '.join(CORNER_NAMES)}")


def check_corner_order_seen_from_behind_near_baseline(corners: np.ndarray) -> None:
    near_left, near_right, far_right, far_left = corners
    near_side_is_lower_in_image = near_left[1] + near_right[1] > far_left[1] + far_right[1]
    left_side_is_left_in_image = near_left[0] < near_right[0] and far_left[0] < far_right[0]
    if not (near_side_is_lower_in_image and left_side_is_left_in_image):
        raise ValueError(
            f"corner order looks wrong for a camera behind the near baseline; give them in order {', '.join(CORNER_NAMES)}"
        )


def homography_from_corners(corner_pixels) -> np.ndarray:
    corners = as_corner_array(corner_pixels)
    check_corners_form_convex_quadrilateral(corners)
    check_corner_order_seen_from_behind_near_baseline(corners)
    return homography_between(corners, court_corner_meters())


def apply_homography(homography: np.ndarray, points) -> np.ndarray:
    point_array = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack([point_array, np.ones(len(point_array))]) @ np.asarray(homography).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def pixels_to_court(homography: np.ndarray, pixel_points) -> np.ndarray:
    return apply_homography(homography, pixel_points)


def court_to_pixels(homography: np.ndarray, court_points) -> np.ndarray:
    return apply_homography(np.linalg.inv(homography), court_points)


def reprojection_errors(homography: np.ndarray, pixel_points, court_points) -> np.ndarray:
    projected_pixels = court_to_pixels(homography, court_points)
    return np.linalg.norm(projected_pixels - np.asarray(pixel_points, dtype=float).reshape(-1, 2), axis=1)


def court_line_segments() -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    near_short_service_y = NET_Y_M - SHORT_SERVICE_LINE_FROM_NET_M
    far_short_service_y = NET_Y_M + SHORT_SERVICE_LINE_FROM_NET_M
    near_long_service_y = DOUBLES_LONG_SERVICE_LINE_FROM_BASELINE_M
    far_long_service_y = COURT_LENGTH_M - DOUBLES_LONG_SERVICE_LINE_FROM_BASELINE_M
    left_singles_x = SINGLES_SIDELINE_INSET_M
    right_singles_x = COURT_WIDTH_M - SINGLES_SIDELINE_INSET_M
    centre_x = COURT_WIDTH_M / 2
    return {
        "near_baseline": ((0.0, 0.0), (COURT_WIDTH_M, 0.0)),
        "far_baseline": ((0.0, COURT_LENGTH_M), (COURT_WIDTH_M, COURT_LENGTH_M)),
        "left_doubles_sideline": ((0.0, 0.0), (0.0, COURT_LENGTH_M)),
        "right_doubles_sideline": ((COURT_WIDTH_M, 0.0), (COURT_WIDTH_M, COURT_LENGTH_M)),
        "left_singles_sideline": ((left_singles_x, 0.0), (left_singles_x, COURT_LENGTH_M)),
        "right_singles_sideline": ((right_singles_x, 0.0), (right_singles_x, COURT_LENGTH_M)),
        "near_doubles_long_service_line": ((0.0, near_long_service_y), (COURT_WIDTH_M, near_long_service_y)),
        "far_doubles_long_service_line": ((0.0, far_long_service_y), (COURT_WIDTH_M, far_long_service_y)),
        "near_short_service_line": ((0.0, near_short_service_y), (COURT_WIDTH_M, near_short_service_y)),
        "far_short_service_line": ((0.0, far_short_service_y), (COURT_WIDTH_M, far_short_service_y)),
        "near_centre_line": ((centre_x, 0.0), (centre_x, near_short_service_y)),
        "far_centre_line": ((centre_x, far_short_service_y), (centre_x, COURT_LENGTH_M)),
        "net": ((0.0, NET_Y_M), (COURT_WIDTH_M, NET_Y_M)),
    }


def points_along_segment(start, end, point_count: int) -> np.ndarray:
    fractions = np.linspace(0.0, 1.0, point_count)[:, None]
    return np.asarray(start, dtype=float) * (1 - fractions) + np.asarray(end, dtype=float) * fractions


def draw_court_lines(
    frame: np.ndarray,
    homography: np.ndarray,
    color: tuple[int, int, int] = LINE_COLOR_BGR,
    thickness: int = 1,
) -> np.ndarray:
    drawn_frame = frame.copy()
    for start, end in court_line_segments().values():
        start_pixel, end_pixel = np.round(court_to_pixels(homography, [start, end])).astype(int)
        cv2.line(drawn_frame, tuple(map(int, start_pixel)), tuple(map(int, end_pixel)), color, thickness, cv2.LINE_AA)
    return drawn_frame


def draw_corner_markers(frame: np.ndarray, corner_pixels, color: tuple[int, int, int] = CORNER_COLOR_BGR) -> np.ndarray:
    marked_frame = frame.copy()
    for corner_name, (x, y) in zip(CORNER_NAMES, as_corner_array(corner_pixels)):
        center = (int(round(x)), int(round(y)))
        cv2.circle(marked_frame, center, 6, color, 2)
        cv2.putText(marked_frame, corner_name, (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return marked_frame


def brightness_profile_peak_offset(gray_frame: np.ndarray, point: np.ndarray, normal: np.ndarray, search_radius: int) -> float | None:
    offsets = np.arange(-search_radius, search_radius + 1)
    sample_points = np.round(point + offsets[:, None] * normal).astype(int)
    height, width = gray_frame.shape
    inside = (
        (sample_points[:, 0] >= 0) & (sample_points[:, 0] < width) & (sample_points[:, 1] >= 0) & (sample_points[:, 1] < height)
    )
    if not np.all(inside):
        return None
    brightness = gray_frame[sample_points[:, 1], sample_points[:, 0]].astype(float)
    if brightness.max() - np.median(brightness) < 40:
        return None
    brightest = np.flatnonzero(brightness == brightness.max())
    return float(offsets[brightest].mean())


def line_offsets_to_bright_pixels(
    frame: np.ndarray,
    homography: np.ndarray,
    start_meters,
    end_meters,
    sample_count: int = 40,
    search_radius: int = 8,
) -> np.ndarray:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    start_pixel, end_pixel = court_to_pixels(homography, [start_meters, end_meters])
    direction = end_pixel - start_pixel
    normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    sample_fractions = np.linspace(0.05, 0.95, sample_count)
    offsets = []
    for fraction in sample_fractions:
        peak_offset = brightness_profile_peak_offset(
            gray_frame, start_pixel + fraction * direction, normal, search_radius
        )
        if peak_offset is not None:
            offsets.append(peak_offset)
    return np.array(offsets)


def calibration_path(video_path: Path, calibration_dir: Path = CALIBRATION_DIR) -> Path:
    return calibration_dir / Path(video_path).stem / "court.json"


def save_calibration(
    output_path: Path,
    video_path: Path,
    frame_index: int,
    frame_size: tuple[int, int],
    corner_pixels,
) -> dict:
    corners = as_corner_array(corner_pixels)
    homography = homography_from_corners(corners)
    calibration = {
        "video": str(video_path),
        "frame_index": int(frame_index),
        "frame_width": int(frame_size[0]),
        "frame_height": int(frame_size[1]),
        "corner_order": list(CORNER_NAMES),
        "corner_pixels": corners.tolist(),
        "court_corner_meters": court_corner_meters().tolist(),
        "pixel_to_court_homography": homography.tolist(),
        "max_corner_reprojection_error_px": float(reprojection_errors(homography, corners, court_corner_meters()).max()),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calibration, indent=2))
    return calibration


def load_calibration(calibration_file: Path) -> dict:
    calibration_file = Path(calibration_file)
    if not calibration_file.is_file():
        raise FileNotFoundError(f"calibration not found: {calibration_file} (run python -m vision.court first)")
    calibration = json.loads(calibration_file.read_text())
    calibration["pixel_to_court_homography"] = np.array(calibration["pixel_to_court_homography"])
    calibration["corner_pixels"] = np.array(calibration["corner_pixels"])
    return calibration


def shuttleset_corner_pixels(match_folder: str, homography_csv: Path = SHUTTLESET_HOMOGRAPHY_CSV) -> np.ndarray:
    if not Path(homography_csv).is_file():
        raise FileNotFoundError(f"homography CSV not found: {homography_csv}")
    homography_rows = pd.read_csv(homography_csv)
    match_rows = homography_rows[homography_rows["video"] == match_folder]
    if match_rows.empty:
        raise ValueError(f"match not in {homography_csv}: {match_folder}")
    row = match_rows.iloc[0]
    return np.array(
        [
            [row["downleft_x"], row["downleft_y"]],
            [row["downright_x"], row["downright_y"]],
            [row["upright_x"], row["upright_y"]],
            [row["upleft_x"], row["upleft_y"]],
        ],
        dtype=float,
    )


def parse_corner_argument(corner_text: str) -> tuple[float, float]:
    try:
        x_text, y_text = corner_text.split(",")
        return float(x_text), float(y_text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"corner must be x,y pixels, got {corner_text!r}")


def window_was_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def click_corners(frame: np.ndarray, window_name: str = "court calibration") -> np.ndarray | None:
    clicked_points: list[tuple[int, int]] = []

    def remember_click(event, x, y, flags, parameters):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            clicked_points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, remember_click)
    try:
        while True:
            shown_frame = frame.copy()
            for point in clicked_points:
                cv2.circle(shown_frame, point, 6, CORNER_COLOR_BGR, 2)
            prompt = (
                f"click {CORNER_NAMES[len(clicked_points)]} corner (u = undo, esc = cancel)"
                if len(clicked_points) < 4
                else "enter = save, u = undo, esc = cancel"
            )
            cv2.putText(shown_frame, prompt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CORNER_COLOR_BGR, 2)
            cv2.imshow(window_name, shown_frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or window_was_closed(window_name):
                return None
            if key == ord("u") and clicked_points:
                clicked_points.pop()
            if key in (10, 13) and len(clicked_points) == 4:
                return np.array(clicked_points, dtype=float)
    finally:
        if not window_was_closed(window_name):
            cv2.destroyWindow(window_name)


def overlay_path_for(calibration_file: Path) -> Path:
    return Path(calibration_file).with_name("court_overlay.jpg")


def calibrate_video(
    video_path: Path,
    frame_index: int = 0,
    corner_pixels=None,
    calibration_dir: Path = CALIBRATION_DIR,
    corners_in_shuttleset_label_frame: bool = False,
) -> Path | None:
    capture = open_video(Path(video_path))
    try:
        frame = read_frame(capture, frame_index)
    finally:
        capture.release()
    if frame is None:
        raise ValueError(f"could not read frame {frame_index} of {video_path}")
    if corner_pixels is not None and corners_in_shuttleset_label_frame:
        corner_pixels = as_corner_array(corner_pixels) * label_scale(frame.shape[1])
    if corner_pixels is None:
        corner_pixels = click_corners(frame)
        if corner_pixels is None:
            return None
    output_path = calibration_path(Path(video_path), calibration_dir)
    calibration = save_calibration(output_path, Path(video_path), frame_index, (frame.shape[1], frame.shape[0]), corner_pixels)
    overlay = draw_corner_markers(
        draw_court_lines(frame, np.array(calibration["pixel_to_court_homography"])), corner_pixels
    )
    cv2.imwrite(str(overlay_path_for(output_path)), overlay)
    return output_path


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vision.court",
        description="Click the 4 court corners on one frame and save the pixel -> court-meter homography.",
    )
    parser.add_argument("video", type=Path, help="path to the match video")
    parser.add_argument("--frame", type=int, default=0, help="frame index to calibrate on")
    corner_source = parser.add_mutually_exclusive_group()
    corner_source.add_argument(
        "--corners",
        type=parse_corner_argument,
        nargs=4,
        metavar="X,Y",
        help="corner pixels in order near_left near_right far_right far_left (skips the click window)",
    )
    corner_source.add_argument(
        "--shuttleset-match",
        help="take the corners from ShuttleSet homography.csv for this match folder (skips the click window)",
    )
    parser.add_argument("--output-dir", type=Path, default=CALIBRATION_DIR, help="root folder for calibrations")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        corner_pixels = parsed.corners
        if parsed.shuttleset_match:
            corner_pixels = shuttleset_corner_pixels(parsed.shuttleset_match)
        output_path = calibrate_video(
            parsed.video,
            parsed.frame,
            corner_pixels,
            parsed.output_dir,
            corners_in_shuttleset_label_frame=bool(parsed.shuttleset_match),
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if output_path is None:
        print("cancelled: no calibration saved", file=sys.stderr)
        return 1
    print(f"saved {output_path}")
    print(f"saved {overlay_path_for(output_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
