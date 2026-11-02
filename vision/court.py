import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from analytics.court import CENTRE_X_M, DEPTH_EDGES_M, DEPTH_NAMES, depth_band_from_net_m, rotate_half_turn, zone_in_near_half
from vision.alignment import label_scale, open_video, read_frame

COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
NET_Y_M = COURT_LENGTH_M / 2
SINGLES_SIDELINE_INSET_M = 0.46
SHORT_SERVICE_LINE_FROM_NET_M = 1.98
DOUBLES_LONG_SERVICE_LINE_FROM_BASELINE_M = 0.76
PAINTED_LINE_WIDTH_M = 0.04
SEARCH_RADIUS_LINE_WIDTHS = 1.5
CORNER_NAMES = ("near_left", "near_right", "far_right", "far_left")
DATA_DIR = Path("data")
CALIBRATION_DIR = Path("data/calibration")
SHUTTLESET_HOMOGRAPHY_CSV = Path("data/CoachAI-Projects/ShuttleSet/set/homography.csv")
LINE_COLOR_BGR = (0, 0, 255)
CORNER_COLOR_BGR = (0, 255, 255)
ZONE_LINE_COLOR_BGR = (0, 255, 255)
ZONE_TEXT_COLOR_BGR = (255, 255, 255)
ZONES_CHECK_DIR = Path("data/checks/court")


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


def court_lines_across() -> dict[str, float]:
    return {
        "near_baseline": 0.0,
        "near_long_service_line": DOUBLES_LONG_SERVICE_LINE_FROM_BASELINE_M,
        "near_short_service_line": NET_Y_M - SHORT_SERVICE_LINE_FROM_NET_M,
        "far_short_service_line": NET_Y_M + SHORT_SERVICE_LINE_FROM_NET_M,
        "far_long_service_line": COURT_LENGTH_M - DOUBLES_LONG_SERVICE_LINE_FROM_BASELINE_M,
        "far_baseline": COURT_LENGTH_M,
    }


def court_lines_along() -> dict[str, float]:
    return {
        "left_doubles_sideline": 0.0,
        "left_singles_sideline": SINGLES_SIDELINE_INSET_M,
        "centre_line": COURT_WIDTH_M / 2,
        "right_singles_sideline": COURT_WIDTH_M - SINGLES_SIDELINE_INSET_M,
        "right_doubles_sideline": COURT_WIDTH_M,
    }


def court_point_catalog() -> dict[str, tuple[float, float]]:
    crossings = {
        f"{across} x {along}": (x_m, y_m) for across, y_m in court_lines_across().items() for along, x_m in court_lines_along().items()
    }
    corners = dict(zip(CORNER_NAMES, map(tuple, court_corner_meters())))
    net_post_bases = {"left_net_post_base": (0.0, NET_Y_M), "right_net_post_base": (COURT_WIDTH_M, NET_Y_M)}
    return {**corners, **crossings, **net_post_bases}


def court_point_meters(point_names) -> np.ndarray:
    catalog = court_point_catalog()
    unknown = [name for name in point_names if name not in catalog]
    if unknown:
        raise ValueError(f"unknown court point(s) {unknown}; see python -m vision.court --list-points")
    return np.array([catalog[name] for name in point_names], dtype=float)


def cross_2d(first, second) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def check_points_fix_a_homography(court_points: np.ndarray) -> None:
    unique_points = np.unique(np.round(court_points, 6), axis=0)
    if len(unique_points) < len(court_points):
        raise ValueError("the same court point is given twice (a corner and its line-crossing name are the same point)")
    if len(court_points) < 4:
        raise ValueError(f"need at least 4 court points, got {len(court_points)}")
    for first in range(len(court_points)):
        for second in range(first + 1, len(court_points)):
            for third in range(second + 1, len(court_points)):
                for fourth in range(third + 1, len(court_points)):
                    quad = court_points[[first, second, third, fourth]]
                    if all(
                        abs(cross_2d(quad[b] - quad[a], quad[c] - quad[a])) > 1e-6
                        for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
                    ):
                        return
    raise ValueError("the court points don't fix a homography: no 4 of them are free of 3 on one line")


def homography_from_court_points(pixel_points, point_names) -> np.ndarray:
    pixels = np.asarray(pixel_points, dtype=float).reshape(-1, 2)
    court_points = court_point_meters(point_names)
    if len(pixels) != len(court_points):
        raise ValueError(f"got {len(pixels)} pixels for {len(court_points)} court points")
    if not np.all(np.isfinite(pixels)):
        raise ValueError("point pixels must be finite numbers")
    check_points_fix_a_homography(court_points)
    if len(pixels) == 4:
        return homography_between(pixels, court_points)
    homography, _ = cv2.findHomography(pixels, court_points, 0)
    if homography is None:
        raise ValueError("could not fit a homography to these points")
    return homography / homography[2, 2]


def parse_point_argument(text: str) -> tuple[str, tuple[float, float]]:
    name, separator, pixel_text = text.rpartition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError(f"expected NAME=X,Y, got {text!r}")
    return name.strip(), parse_corner_argument(pixel_text)


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


def zone_boundary_segments() -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    segments = {}
    for index, edge_m in enumerate(DEPTH_EDGES_M, start=1):
        segments[f"near_depth_{index}"] = ((0.0, NET_Y_M - edge_m), (COURT_WIDTH_M, NET_Y_M - edge_m))
        segments[f"far_depth_{index}"] = ((0.0, NET_Y_M + edge_m), (COURT_WIDTH_M, NET_Y_M + edge_m))
    segments["centre"] = ((CENTRE_X_M, 0.0), (CENTRE_X_M, COURT_LENGTH_M))
    return segments


def zone_label_points(left_handed: bool = False) -> dict[tuple[str, str], tuple[float, float]]:
    column_centres = (CENTRE_X_M / 2, (CENTRE_X_M + COURT_WIDTH_M) / 2)
    labels = {}
    for depth in DEPTH_NAMES:
        near_edge_m, far_edge_m = depth_band_from_net_m(depth)
        distance_m = (near_edge_m + far_edge_m) / 2
        for x_m in column_centres:
            near = (x_m, NET_Y_M - distance_m)
            far = (x_m, NET_Y_M + distance_m)
            labels[("near", zone_in_near_half(near, left_handed))] = near
            labels[("far", zone_in_near_half(rotate_half_turn(far), left_handed))] = far
    return labels


def boxed_text(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float, thickness: int) -> None:
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    padding = max(2, thickness * 2)
    x, y = origin
    cv2.rectangle(frame, (x - padding, y - height - padding), (x + width + padding, y + baseline + padding), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, ZONE_TEXT_COLOR_BGR, thickness, cv2.LINE_AA)


def text_scale_for_width(frame_width: int, scale_at_1280: float) -> tuple[float, int]:
    scale = scale_at_1280 * frame_width / 1280
    return scale, max(1, int(round(scale * 1.6)))


def draw_zones(frame: np.ndarray, homography: np.ndarray, left_handed: bool = False) -> np.ndarray:
    drawn_frame = draw_court_lines(frame, homography, LINE_COLOR_BGR, 2)
    for start, end in zone_boundary_segments().values():
        start_pixel, end_pixel = np.round(court_to_pixels(homography, [start, end])).astype(int)
        cv2.line(drawn_frame, tuple(map(int, start_pixel)), tuple(map(int, end_pixel)), ZONE_LINE_COLOR_BGR, 1, cv2.LINE_AA)
    frame_width = drawn_frame.shape[1]
    for (half, name), point in zone_label_points(left_handed).items():
        x, y = np.round(court_to_pixels(homography, [point])[0]).astype(int)
        scale, thickness = text_scale_for_width(frame_width, 0.4 if half == "far" else 0.5)
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        boxed_text(drawn_frame, name, (int(x - text_size[0] / 2), int(y + text_size[1] / 2)), scale, thickness)
    caption = "near half: my hit zones (right-handed, forehand on my right); far half: landing zones of my shots, named from the receiver's side as in the shots table"
    scale, thickness = text_scale_for_width(frame_width, 0.5)
    caption_width = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    if caption_width > frame_width - 20:
        scale *= (frame_width - 20) / caption_width
    boxed_text(drawn_frame, caption, (10, drawn_frame.shape[0] - 14), scale, thickness)
    return drawn_frame


def zones_image_path(video_path: Path, checks_dir: Path = ZONES_CHECK_DIR) -> Path:
    return checks_dir / f"{video_id(video_path)}_zones.jpg"


def write_zones_image(video_path: Path, frame_index: int, calibration_dir: Path = CALIBRATION_DIR, checks_dir: Path = ZONES_CHECK_DIR, left_handed: bool = False) -> Path:
    calibration = load_calibration(calibration_path(Path(video_path), calibration_dir))
    capture = open_video(Path(video_path))
    try:
        frame = read_frame(capture, frame_index)
    finally:
        capture.release()
    if frame is None:
        raise ValueError(f"could not read frame {frame_index} of {video_path}")
    output_path = zones_image_path(Path(video_path), checks_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), draw_zones(frame, np.array(calibration["pixel_to_court_homography"]), left_handed))
    return output_path


def draw_point_markers(frame: np.ndarray, calibration: dict, color: tuple[int, int, int] = CORNER_COLOR_BGR) -> np.ndarray:
    marked_frame = frame.copy()
    for point in calibration.get("reference_points", []):
        center = (int(round(point["pixel"][0])), int(round(point["pixel"][1])))
        cv2.circle(marked_frame, center, 6, color, 2)
        cv2.putText(marked_frame, f"{point['name']} ({point['error_px']:.1f} px)", (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return marked_frame


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


def line_samples_inside_frame(
    frame_size: tuple[int, int], homography: np.ndarray, start_meters, end_meters, sample_count: int = 40, search_radius: int = 8
) -> int:
    width, height = frame_size
    start_pixel, end_pixel = court_to_pixels(homography, [start_meters, end_meters])
    samples = start_pixel + np.linspace(0.05, 0.95, sample_count)[:, None] * (end_pixel - start_pixel)
    inside = (
        (samples[:, 0] >= search_radius) & (samples[:, 0] < width - search_radius)
        & (samples[:, 1] >= search_radius) & (samples[:, 1] < height - search_radius)
    )
    return int(inside.sum())


def painted_line_width_px(homography: np.ndarray, point_meters, normal_meters, width_m: float = PAINTED_LINE_WIDTH_M) -> float:
    pixels = court_to_pixels(homography, [point_meters, np.asarray(point_meters, dtype=float) + width_m * np.asarray(normal_meters, dtype=float)])
    return float(np.linalg.norm(pixels[1] - pixels[0]))


def line_sample_counts(
    frame: np.ndarray,
    homography: np.ndarray,
    start_meters,
    end_meters,
    sample_count: int = 40,
    min_search_radius: int = 8,
    width_factor: float = SEARCH_RADIUS_LINE_WIDTHS,
) -> tuple[int, int]:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    height, width = gray_frame.shape
    start_meters, end_meters = np.asarray(start_meters, dtype=float), np.asarray(end_meters, dtype=float)
    along_meters = end_meters - start_meters
    normal_meters = np.array([-along_meters[1], along_meters[0]]) / np.linalg.norm(along_meters)
    start_pixel, end_pixel = court_to_pixels(homography, [start_meters, end_meters])
    direction = end_pixel - start_pixel
    normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    found = inside = 0
    for fraction in np.linspace(0.05, 0.95, sample_count):
        radius = max(min_search_radius, int(np.ceil(width_factor * painted_line_width_px(homography, start_meters + fraction * along_meters, normal_meters))))
        point = start_pixel + fraction * direction
        if not (radius <= point[0] < width - radius and radius <= point[1] < height - radius):
            continue
        inside += 1
        found += brightness_profile_peak_offset(gray_frame, point, normal, radius) is not None
    return found, inside


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


def is_directly_under(folder: Path, path: Path) -> bool:
    return path.parent.resolve() == folder.resolve()


def video_id(video_path: Path, data_dir: Path = DATA_DIR) -> str:
    path = Path(video_path)
    if path.parent == Path(".") or is_directly_under(data_dir, path):
        return path.stem
    return f"{path.parent.name}_{path.stem}"


def calibration_path(video_path: Path, calibration_dir: Path = CALIBRATION_DIR) -> Path:
    return calibration_dir / video_id(video_path) / "court.json"


def point_calibration(video_path: Path, frame_index: int, frame_size: tuple[int, int], point_pixels: dict) -> dict:
    point_names = list(point_pixels)
    pixels = np.array([point_pixels[name] for name in point_names], dtype=float).reshape(-1, 2)
    court_points = court_point_meters(point_names)
    homography = homography_from_court_points(pixels, point_names)
    point_errors = reprojection_errors(homography, pixels, court_points)
    corners_given = all(name in point_pixels for name in CORNER_NAMES)
    corner_pixels = (
        np.array([point_pixels[name] for name in CORNER_NAMES], dtype=float) if corners_given else court_to_pixels(homography, court_corner_meters())
    )
    return {
        "video": str(video_path),
        "frame_index": int(frame_index),
        "frame_width": int(frame_size[0]),
        "frame_height": int(frame_size[1]),
        "corner_order": list(CORNER_NAMES),
        "corner_pixels": corner_pixels.tolist(),
        "corner_pixels_source": "clicked" if corners_given else "projected from the reference points",
        "court_corner_meters": court_corner_meters().tolist(),
        "pixel_to_court_homography": homography.tolist(),
        "max_corner_reprojection_error_px": float(reprojection_errors(homography, corner_pixels, court_corner_meters()).max()),
        "reference_points": [
            {"name": name, "pixel": pixel.tolist(), "court": court.tolist(), "error_px": float(error)}
            for name, pixel, court, error in zip(point_names, pixels, court_points, point_errors)
        ],
        "max_reference_error_px": float(point_errors.max()),
        "rms_reference_error_px": float(np.sqrt(np.mean(point_errors**2))),
    }


def write_calibration(output_path: Path, calibration: dict) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calibration, indent=2))
    return calibration


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
    return write_calibration(output_path, calibration)


def save_point_calibration(output_path: Path, video_path: Path, frame_index: int, frame_size: tuple[int, int], point_pixels: dict) -> dict:
    return write_calibration(output_path, point_calibration(video_path, frame_index, frame_size, point_pixels))


def reference_points(calibration: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    if "reference_points" in calibration:
        points = calibration["reference_points"]
        return [point["name"] for point in points], np.array([point["pixel"] for point in points], dtype=float), np.array([point["court"] for point in points], dtype=float)
    return list(CORNER_NAMES), np.asarray(calibration["corner_pixels"], dtype=float), court_corner_meters()


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


def click_named_points(
    frame: np.ndarray, point_names, window_name: str, color: tuple[int, int, int] = CORNER_COLOR_BGR
) -> np.ndarray | None:
    point_count = len(point_names)
    clicked_points: list[tuple[int, int]] = []

    def remember_click(event, x, y, flags, parameters):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < point_count:
            clicked_points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, remember_click)
    try:
        while True:
            shown_frame = frame.copy()
            for point in clicked_points:
                cv2.circle(shown_frame, point, 6, color, 2)
            prompt = (
                f"click {point_names[len(clicked_points)]} (u = undo, esc = cancel)"
                if len(clicked_points) < point_count
                else "enter = save, u = undo, esc = cancel"
            )
            cv2.putText(shown_frame, prompt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow(window_name, shown_frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or window_was_closed(window_name):
                return None
            if key == ord("u") and clicked_points:
                clicked_points.pop()
            if key in (10, 13) and len(clicked_points) == point_count:
                return np.array(clicked_points, dtype=float)
    finally:
        if not window_was_closed(window_name):
            cv2.destroyWindow(window_name)


def click_corners(frame: np.ndarray, window_name: str = "court calibration") -> np.ndarray | None:
    return click_named_points(frame, [f"{name} corner" for name in CORNER_NAMES], window_name)


def click_net_post_tops(frame: np.ndarray, window_name: str = "net post tops") -> np.ndarray | None:
    return click_named_points(frame, ["left net post top", "right net post top"], window_name)


def overlay_path_for(calibration_file: Path) -> Path:
    return Path(calibration_file).with_name("court_overlay.jpg")


def court_diagram(point_names, current_name: str, placed_names, height_px: int = 300) -> np.ndarray:
    scale = height_px / (COURT_LENGTH_M + 1.0)
    width_px = int((COURT_WIDTH_M + 1.0) * scale)
    diagram = np.full((height_px, width_px, 3), 40, dtype=np.uint8)

    def to_pixel(point):
        return int(round((point[0] + 0.5) * scale)), int(round((COURT_LENGTH_M + 0.5 - point[1]) * scale))

    for start, end in court_line_segments().values():
        cv2.line(diagram, to_pixel(start), to_pixel(end), (200, 200, 200), 1)
    catalog = court_point_catalog()
    for name in point_names:
        color = (0, 255, 255) if name == current_name else (0, 200, 0) if name in placed_names else (120, 120, 120)
        cv2.circle(diagram, to_pixel(catalog[name]), 5 if name == current_name else 3, color, -1)
    return diagram


def clickable_point_names() -> list[str]:
    catalog = court_point_catalog()
    corner_positions = {catalog[name] for name in CORNER_NAMES}
    return list(CORNER_NAMES) + [name for name, position in catalog.items() if name not in CORNER_NAMES and position not in corner_positions]


def same_point_pairs(point_names) -> list[tuple[str, str]]:
    catalog = court_point_catalog()
    names = list(point_names)
    return [(first, second) for index, first in enumerate(names) for second in names[index + 1 :] if catalog.get(first) is not None and catalog.get(first) == catalog.get(second)]


def points_problem(point_pixels: dict) -> str | None:
    pairs = same_point_pairs(point_pixels)
    if pairs:
        return "same point clicked twice: " + ", ".join(f"{first} / {second}" for first, second in pairs)
    try:
        check_points_fix_a_homography(court_point_meters(list(point_pixels)))
    except ValueError as error:
        return str(error)
    return None


def points_are_enough(point_pixels: dict) -> bool:
    return points_problem(point_pixels) is None


def click_court_points(frame: np.ndarray, window_name: str = "court points") -> dict | None:
    point_names = clickable_point_names()
    placed: dict[str, tuple[int, int]] = {}
    current = [0]

    def remember_click(event, x, y, flags, parameters):
        if event == cv2.EVENT_LBUTTONDOWN:
            placed[point_names[current[0]]] = (x, y)
            current[0] = (current[0] + 1) % len(point_names)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, remember_click)
    try:
        while True:
            shown_frame = frame.copy()
            for name, point in placed.items():
                cv2.circle(shown_frame, point, 6, CORNER_COLOR_BGR, 2)
                cv2.putText(shown_frame, name, (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CORNER_COLOR_BGR, 1)
            diagram = court_diagram(point_names, point_names[current[0]], placed)
            shown_frame[10 : 10 + diagram.shape[0], 10 : 10 + diagram.shape[1]] = diagram
            problem = points_problem(placed)
            ready = "enter = save" if problem is None else f"{len(placed)} placed: {problem}"
            prompt = f"click {point_names[current[0]]}   n/p = next/previous point, d = delete it, {ready}, esc = cancel"
            cv2.putText(shown_frame, prompt, (20, shown_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CORNER_COLOR_BGR, 2)
            cv2.imshow(window_name, shown_frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or window_was_closed(window_name):
                return None
            if key == ord("n"):
                current[0] = (current[0] + 1) % len(point_names)
            if key == ord("p"):
                current[0] = (current[0] - 1) % len(point_names)
            if key == ord("d"):
                placed.pop(point_names[current[0]], None)
            if key in (10, 13) and points_are_enough(placed):
                return {name: (float(x), float(y)) for name, (x, y) in placed.items()}
    finally:
        if not window_was_closed(window_name):
            cv2.destroyWindow(window_name)


def calibrate_video(
    video_path: Path,
    frame_index: int = 0,
    corner_pixels=None,
    calibration_dir: Path = CALIBRATION_DIR,
    corners_in_shuttleset_label_frame: bool = False,
    point_pixels: dict | None = None,
    pick_points: bool = False,
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
    output_path = calibration_path(Path(video_path), calibration_dir)
    frame_size = (frame.shape[1], frame.shape[0])
    if point_pixels is None and pick_points:
        point_pixels = click_court_points(frame)
        if point_pixels is None:
            return None
    if point_pixels is not None:
        calibration = save_point_calibration(output_path, Path(video_path), frame_index, frame_size, point_pixels)
        overlay = draw_point_markers(draw_court_lines(frame, np.array(calibration["pixel_to_court_homography"])), calibration)
    else:
        if corner_pixels is None:
            corner_pixels = click_corners(frame)
            if corner_pixels is None:
                return None
        calibration = save_calibration(output_path, Path(video_path), frame_index, frame_size, corner_pixels)
        overlay = draw_corner_markers(draw_court_lines(frame, np.array(calibration["pixel_to_court_homography"])), corner_pixels)
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
    corner_source.add_argument(
        "--points",
        type=parse_point_argument,
        nargs="+",
        metavar="NAME=X,Y",
        help="4 or more known court points by name (see --list-points), for when a corner is hidden",
    )
    corner_source.add_argument("--pick-points", action="store_true", help="open a click window where you choose which court point you click")
    corner_source.add_argument("--list-points", action="store_true", help="print the names of the known court points and exit")
    corner_source.add_argument("--zones", action="store_true", help="draw the 6 shot zones per half (front/mid/rear x forehand/backhand) on --frame through the saved calibration, to data/checks/court/<video id>_zones.jpg")
    parser.add_argument("--left-handed", action="store_true", help="with --zones: name the zones for a left-handed near player")
    parser.add_argument("--output-dir", type=Path, default=CALIBRATION_DIR, help="root folder for calibrations")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    if parsed.list_points:
        for name, (x_m, y_m) in court_point_catalog().items():
            print(f"{name}: ({x_m:.2f}, {y_m:.2f}) m")
        return 0
    try:
        if parsed.zones:
            print(f"saved {write_zones_image(parsed.video, parsed.frame, parsed.output_dir, left_handed=parsed.left_handed)}")
            return 0
        corner_pixels = parsed.corners
        if parsed.shuttleset_match:
            corner_pixels = shuttleset_corner_pixels(parsed.shuttleset_match)
        output_path = calibrate_video(
            parsed.video,
            parsed.frame,
            corner_pixels,
            parsed.output_dir,
            corners_in_shuttleset_label_frame=bool(parsed.shuttleset_match),
            point_pixels=dict(parsed.points) if parsed.points else None,
            pick_points=parsed.pick_points,
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
