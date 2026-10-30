import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vision.alignment import open_video, read_frame
from vision.court import (
    COURT_WIDTH_M,
    NET_Y_M,
    calibration_path,
    click_net_post_tops,
    court_corner_meters,
    court_line_segments,
    line_offsets_to_bright_pixels,
    load_calibration,
    parse_corner_argument,
    points_along_segment,
    reference_points,
)

NET_POST_HEIGHT_M = 1.55
NET_POST_NAMES = ("left", "right")
FOCAL_SEARCH_RANGE_IN_WIDTHS = (0.25, 8.0)
FOCAL_COARSE_STEPS = 400
FOCAL_REFINE_ITERATIONS = 60
LINE_COLOR_BGR = (0, 0, 255)
NET_TOP_COLOR_BGR = (0, 255, 255)
NET_POST_COLOR_BGR = (255, 160, 0)


@dataclass(frozen=True)
class Camera:
    intrinsic: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    frame_size: tuple[int, int]

    @property
    def focal_px(self) -> float:
        return float(self.intrinsic[0, 0])

    @property
    def centre(self) -> np.ndarray:
        return (-self.rotation.T @ self.translation).ravel()

    @property
    def projection(self) -> np.ndarray:
        return self.intrinsic @ np.hstack([self.rotation, self.translation.reshape(3, 1)])


def net_post_top_meters() -> np.ndarray:
    return np.array([[0.0, NET_Y_M, NET_POST_HEIGHT_M], [COURT_WIDTH_M, NET_Y_M, NET_POST_HEIGHT_M]])


def camera_reference_meters(floor_points=None) -> np.ndarray:
    floor = court_corner_meters() if floor_points is None else np.asarray(floor_points, dtype=float).reshape(-1, 2)
    return np.vstack([np.column_stack([floor, np.zeros(len(floor))]), net_post_top_meters()])


def intrinsic_matrix(focal_px: float, frame_size: tuple[int, int]) -> np.ndarray:
    width, height = frame_size
    return np.array([[focal_px, 0.0, width / 2], [0.0, focal_px, height / 2], [0.0, 0.0, 1.0]])


def project_points(camera: Camera, world_points) -> np.ndarray:
    homogeneous = np.column_stack([np.asarray(world_points, dtype=float).reshape(-1, 3), np.ones(len(np.asarray(world_points).reshape(-1, 3)))])
    projected = homogeneous @ camera.projection.T
    return projected[:, :2] / projected[:, 2:3]


def reprojection_errors(camera: Camera, world_points, pixel_points) -> np.ndarray:
    return np.linalg.norm(project_points(camera, world_points) - np.asarray(pixel_points, dtype=float).reshape(-1, 2), axis=1)


def camera_for_focal(world_points, pixel_points, focal_px: float, frame_size: tuple[int, int]) -> Camera:
    world = np.asarray(world_points, dtype=float).reshape(-1, 3)
    pixels = np.asarray(pixel_points, dtype=float).reshape(-1, 2)
    intrinsic = intrinsic_matrix(focal_px, frame_size)
    _, rotation_vector, translation = cv2.solvePnP(world, pixels, intrinsic, None, flags=cv2.SOLVEPNP_SQPNP)
    _, rotation_vector, translation = cv2.solvePnP(
        world, pixels, intrinsic, None, rotation_vector, translation, True, flags=cv2.SOLVEPNP_ITERATIVE
    )
    return Camera(intrinsic, cv2.Rodrigues(rotation_vector)[0], translation.ravel(), tuple(int(size) for size in frame_size))


def rms_reprojection_error(camera: Camera, world_points, pixel_points) -> float:
    return float(np.sqrt(np.mean(reprojection_errors(camera, world_points, pixel_points) ** 2)))


def fit_camera(world_points, pixel_points, frame_size: tuple[int, int]) -> Camera:
    if len(np.asarray(world_points).reshape(-1, 3)) < 6:
        raise ValueError("need at least 6 points (4 or more court points and 2 net-post tops) to fit the focal length")
    width = frame_size[0]
    focal_candidates = np.geomspace(width * FOCAL_SEARCH_RANGE_IN_WIDTHS[0], width * FOCAL_SEARCH_RANGE_IN_WIDTHS[1], FOCAL_COARSE_STEPS)

    def error_at(focal_px: float) -> float:
        return rms_reprojection_error(camera_for_focal(world_points, pixel_points, focal_px, frame_size), world_points, pixel_points)

    coarse_errors = [error_at(focal) for focal in focal_candidates]
    best = int(np.argmin(coarse_errors))
    low = focal_candidates[max(best - 1, 0)]
    high = focal_candidates[min(best + 1, len(focal_candidates) - 1)]
    golden = (np.sqrt(5) - 1) / 2
    for _ in range(FOCAL_REFINE_ITERATIONS):
        inner_low, inner_high = high - golden * (high - low), low + golden * (high - low)
        if error_at(inner_low) < error_at(inner_high):
            high = inner_high
        else:
            low = inner_low
    return camera_for_focal(world_points, pixel_points, (low + high) / 2, frame_size)


def pixel_ray(camera: Camera, pixel) -> tuple[np.ndarray, np.ndarray]:
    direction_in_camera = np.linalg.inv(camera.intrinsic) @ np.array([pixel[0], pixel[1], 1.0])
    direction = camera.rotation.T @ direction_in_camera
    return camera.centre, direction / np.linalg.norm(direction)


def ray_point_at_height(origin, direction, height_m: float) -> np.ndarray | None:
    if abs(direction[2]) < 1e-12:
        return None
    distance = (height_m - origin[2]) / direction[2]
    return None if distance <= 0 else np.asarray(origin) + distance * np.asarray(direction)


def floor_homography(camera: Camera) -> np.ndarray:
    court_to_pixel = camera.intrinsic @ np.column_stack([camera.rotation[:, 0], camera.rotation[:, 1], camera.translation])
    pixel_to_court = np.linalg.inv(court_to_pixel)
    return pixel_to_court / pixel_to_court[2, 2]


def floor_line_offsets(frame: np.ndarray, camera: Camera) -> np.ndarray:
    homography = floor_homography(camera)
    return np.concatenate(
        [line_offsets_to_bright_pixels(frame, homography, start, end) for name, (start, end) in court_line_segments().items() if name != "net"]
    )


def draw_camera_check(frame: np.ndarray, camera: Camera, net_post_pixels) -> np.ndarray:
    drawn = frame.copy()
    for name, (start, end) in court_line_segments().items():
        if name == "net":
            continue
        pixels = project_points(camera, np.column_stack([points_along_segment(start, end, 2), np.zeros(2)]))
        cv2.line(drawn, tuple(np.round(pixels[0]).astype(int)), tuple(np.round(pixels[1]).astype(int)), LINE_COLOR_BGR, 1, cv2.LINE_AA)
    net_top = project_points(camera, net_post_top_meters())
    cv2.line(drawn, tuple(np.round(net_top[0]).astype(int)), tuple(np.round(net_top[1]).astype(int)), NET_TOP_COLOR_BGR, 1, cv2.LINE_AA)
    for post_base, post_top in zip(project_points(camera, net_post_top_meters() * [1, 1, 0]), net_top):
        cv2.line(drawn, tuple(np.round(post_base).astype(int)), tuple(np.round(post_top).astype(int)), NET_TOP_COLOR_BGR, 1, cv2.LINE_AA)
    for clicked in np.asarray(net_post_pixels, dtype=float).reshape(-1, 2):
        cv2.circle(drawn, tuple(np.round(clicked).astype(int)), 6, NET_POST_COLOR_BGR, 2)
    return drawn


def camera_path(video_path: Path, calibration_dir: Path | None = None) -> Path:
    court_file = calibration_path(Path(video_path)) if calibration_dir is None else calibration_path(Path(video_path), calibration_dir)
    return court_file.with_name("camera.json")


def camera_check_path(camera_file: Path) -> Path:
    return Path(camera_file).with_name("camera_check.jpg")


def camera_to_dict(
    camera: Camera, floor_pixels, net_post_pixels, frame_index: int, video_path: Path, floor_names=None, floor_meters=None
) -> dict:
    floor_names = ["near_left", "near_right", "far_right", "far_left"] if floor_names is None else list(floor_names)
    world = camera_reference_meters(floor_meters)
    pixels = np.vstack([np.asarray(floor_pixels, dtype=float).reshape(-1, 2), np.asarray(net_post_pixels, dtype=float)])
    return {
        "video": str(video_path),
        "frame_index": int(frame_index),
        "frame_width": camera.frame_size[0],
        "frame_height": camera.frame_size[1],
        "reference_point_order": floor_names + ["left_net_post_top", "right_net_post_top"],
        "reference_meters": world.tolist(),
        "reference_pixels": pixels.tolist(),
        "focal_px": camera.focal_px,
        "intrinsic": camera.intrinsic.tolist(),
        "rotation": camera.rotation.tolist(),
        "translation": camera.translation.tolist(),
        "camera_centre_m": camera.centre.tolist(),
        "reference_errors_px": reprojection_errors(camera, world, pixels).tolist(),
    }


def camera_from_dict(camera_data: dict) -> Camera:
    return Camera(
        np.array(camera_data["intrinsic"], dtype=float),
        np.array(camera_data["rotation"], dtype=float),
        np.array(camera_data["translation"], dtype=float),
        (int(camera_data["frame_width"]), int(camera_data["frame_height"])),
    )


def load_camera(camera_file: Path) -> Camera:
    camera_file = Path(camera_file)
    if not camera_file.is_file():
        raise FileNotFoundError(f"camera not found: {camera_file} (run python -m vision.camera first)")
    return camera_from_dict(json.loads(camera_file.read_text()))


def calibrate_camera(video_path: Path, net_post_pixels=None, frame_index: int | None = None) -> Path | None:
    court = load_calibration(calibration_path(Path(video_path)))
    frame_index = court["frame_index"] if frame_index is None else frame_index
    capture = open_video(Path(video_path))
    try:
        frame = read_frame(capture, frame_index)
    finally:
        capture.release()
    if frame is None:
        raise ValueError(f"could not read frame {frame_index} of {video_path}")
    if net_post_pixels is None:
        net_post_pixels = click_net_post_tops(frame)
        if net_post_pixels is None:
            return None
    frame_size = (frame.shape[1], frame.shape[0])
    floor_names, floor_pixels, floor_meters = reference_points(court)
    pixels = np.vstack([floor_pixels, np.asarray(net_post_pixels, dtype=float).reshape(2, 2)])
    camera = fit_camera(camera_reference_meters(floor_meters), pixels, frame_size)
    output_path = camera_path(Path(video_path))
    camera_data = camera_to_dict(camera, floor_pixels, net_post_pixels, frame_index, Path(video_path), floor_names, floor_meters)
    offsets = floor_line_offsets(frame, camera)
    camera_data["floor_line_offsets_px"] = {
        "samples": int(len(offsets)),
        "median_abs": float(np.median(np.abs(offsets))),
        "mean_abs": float(np.mean(np.abs(offsets))),
        "p95_abs": float(np.percentile(np.abs(offsets), 95)),
    }
    output_path.write_text(json.dumps(camera_data, indent=2))
    cv2.imwrite(str(camera_check_path(output_path)), draw_camera_check(frame, camera, net_post_pixels))
    return output_path


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vision.camera",
        description="Fit a full camera (focal length and pose) from the court corners and the 2 net-post tops.",
    )
    parser.add_argument("video", type=Path, help="video with a court calibration (python -m vision.court)")
    parser.add_argument(
        "--net-posts", type=parse_corner_argument, nargs=2, metavar="X,Y", help="pixels of the left and right net-post tops (skips the click window)"
    )
    parser.add_argument("--frame", type=int, help="frame to use (default: the court calibration's frame)")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        output_path = calibrate_camera(parsed.video, parsed.net_posts, parsed.frame)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if output_path is None:
        print("cancelled: no camera saved", file=sys.stderr)
        return 1
    camera_data = json.loads(output_path.read_text())
    errors = ", ".join(f"{error:.2f}" for error in camera_data["reference_errors_px"])
    offsets = camera_data["floor_line_offsets_px"]
    print(f"focal length {camera_data['focal_px']:.1f} px, camera at {np.round(camera_data['camera_centre_m'], 2).tolist()} m")
    print(f"reference point errors (px): {errors}")
    print(f"court lines: {offsets['samples']} samples, median {offsets['median_abs']:.1f} px, 95th percentile {offsets['p95_abs']:.1f} px")
    print(f"saved {output_path}")
    print(f"saved {camera_check_path(output_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
