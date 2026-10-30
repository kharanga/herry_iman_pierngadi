import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vision.alignment import SHUTTLESET_SET_DIR, label_scale, load_set_shots, open_video, read_frame
from vision.camera import Camera, camera_path, load_camera, pixel_ray, project_points
from vision.court import calibration_path, court_to_pixels, load_calibration, pixels_to_court, video_id
from vision.hits import frames_for_fps, match_hits, video_fps_and_width
from vision.hits import output_paths as hit_output_paths
from vision.players import PLAYERS, boxes_path, require_video_file
from vision.players import positions_path as player_positions_path
from vision.shuttle import positions_path as shuttle_positions_path

POSE_MODEL_PATH = Path("data/models/yolo11s-pose.pt")
CONTACT_DIR = Path("data/contact")
FINGERPRINT_BYTES = 1024 * 1024
SOURCE_RECORD_SUFFIX = ".source.json"
SHUTTLE_MAX_FRAME_GAP = 2
FEET_MAX_FRAME_GAP = 3
REACH_RADIUS_M = 1.0
CONTACT_HEIGHT_RANGE_M = (0.0, 3.5)
INTERSECTION_CHOICES = ("toward_net", "nearer_camera", "farther_from_camera")
INTERSECTION_CHOICE = "farther_from_camera"
SHOULDER_MIN_CONFIDENCE = 0.3
POSE_MIN_BOX_OVERLAP = 0.3
LEFT_SHOULDER_INDEX = 5
RIGHT_SHOULDER_INDEX = 6
NET_Y_M = 6.7
POSE_COLUMNS = ["frame", "left", "top", "right", "bottom", "left_shoulder_x", "left_shoulder_y", "left_shoulder_conf", "right_shoulder_x", "right_shoulder_y", "right_shoulder_conf"]
CONTACT_COLUMNS = ["frame", "hitter", "hit_x", "hit_y", "contact_z", "feet_x", "feet_y", "method"]
CHECKS_DIR = Path("data/checks/contact")
SPOT_CHECK_HITS = 10
SPOT_CHECK_SEED = 1
MATCH_TOLERANCE_FRAMES = 5
TILE_WIDTH, TILE_HEIGHT = 420, 300
SHUTTLE_COLOR_BGR, CONTACT_COLOR_BGR, FEET_COLOR_BGR, LABEL_COLOR_BGR, TEXT_COLOR_BGR = (0, 255, 255), (0, 200, 0), (255, 160, 0), (0, 0, 255), (255, 255, 255)


@dataclass(frozen=True)
class ContactSettings:
    reach_radius_m: float = REACH_RADIUS_M
    intersection_choice: str = INTERSECTION_CHOICE
    min_height_m: float = CONTACT_HEIGHT_RANGE_M[0]
    max_height_m: float = CONTACT_HEIGHT_RANGE_M[1]
    shoulder_min_confidence: float = SHOULDER_MIN_CONFIDENCE
    shuttle_max_frame_gap: int = SHUTTLE_MAX_FRAME_GAP
    feet_max_frame_gap: int = FEET_MAX_FRAME_GAP


def contact_settings_for_video(fps: float, base: ContactSettings = ContactSettings()) -> ContactSettings:
    return replace(
        base,
        shuttle_max_frame_gap=frames_for_fps(base.shuttle_max_frame_gap, fps),
        feet_max_frame_gap=frames_for_fps(base.feet_max_frame_gap, fps),
    )


def closest_ray_distance(origin, direction, point) -> float:
    return max(float(np.dot(np.asarray(point, dtype=float) - origin, direction)), 0.0)


def ray_point(origin, direction, distance: float) -> np.ndarray:
    return np.asarray(origin, dtype=float) + distance * np.asarray(direction, dtype=float)


def ray_sphere_distances(origin, direction, centre, radius: float) -> list[float]:
    along = closest_ray_distance(origin, direction, centre)
    miss_squared = float(np.sum((ray_point(origin, direction, along) - centre) ** 2))
    if miss_squared > radius**2:
        return []
    half_chord = np.sqrt(radius**2 - miss_squared)
    return [float(distance) for distance in (along - half_chord, along + half_chord) if distance > 0]


def closest_point_to_vertical_line(origin, direction, floor_point) -> np.ndarray:
    horizontal = np.array([direction[0], direction[1]])
    offset = np.asarray(floor_point, dtype=float) - np.asarray(origin, dtype=float)[:2]
    horizontal_length_squared = float(horizontal @ horizontal)
    distance = float(offset @ horizontal) / horizontal_length_squared if horizontal_length_squared > 1e-12 else 0.0
    return ray_point(origin, direction, max(distance, 0.0))


def chosen_intersection(points: list[np.ndarray], camera_centre, choice: str) -> np.ndarray:
    if choice == "toward_net":
        return min(points, key=lambda point: abs(point[1] - NET_Y_M))
    if choice == "nearer_camera":
        return min(points, key=lambda point: np.linalg.norm(point - camera_centre))
    if choice == "farther_from_camera":
        return max(points, key=lambda point: np.linalg.norm(point - camera_centre))
    raise ValueError(f"intersection choice must be one of {INTERSECTION_CHOICES}, got {choice!r}")


def contact_point(origin, direction, shoulder, settings: ContactSettings = ContactSettings()) -> tuple[np.ndarray, str]:
    shoulder = np.asarray(shoulder, dtype=float)
    intersections = [ray_point(origin, direction, distance) for distance in ray_sphere_distances(origin, direction, shoulder, settings.reach_radius_m)]
    plausible = [point for point in intersections if settings.min_height_m <= point[2] <= settings.max_height_m]
    if plausible:
        return chosen_intersection(plausible, np.asarray(origin, dtype=float), settings.intersection_choice), "sphere"
    return ray_point(origin, direction, closest_ray_distance(origin, direction, shoulder)), "closest to sphere"


def shoulder_position(camera: Camera, shoulder_pixel, feet_point) -> np.ndarray:
    origin, direction = pixel_ray(camera, shoulder_pixel)
    return closest_point_to_vertical_line(origin, direction, feet_point)


def box_overlap(first_box, second_box) -> float:
    left, top = max(first_box[0], second_box[0]), max(first_box[1], second_box[1])
    right, bottom = min(first_box[2], second_box[2]), min(first_box[3], second_box[3])
    intersection = max(right - left, 0.0) * max(bottom - top, 0.0)
    union = (first_box[2] - first_box[0]) * (first_box[3] - first_box[1]) + (second_box[2] - second_box[0]) * (second_box[3] - second_box[1]) - intersection
    return float(intersection / union) if union > 0 else 0.0


def hitter_pose(poses_at_frame: pd.DataFrame, hitter_box, min_overlap: float = POSE_MIN_BOX_OVERLAP):
    if poses_at_frame.empty or hitter_box is None or not np.all(np.isfinite(hitter_box)):
        return None
    overlaps = [box_overlap((pose.left, pose.top, pose.right, pose.bottom), hitter_box) for pose in poses_at_frame.itertuples()]
    best = int(np.argmax(overlaps))
    return poses_at_frame.iloc[best] if overlaps[best] >= min_overlap else None


def hitting_shoulder_pixel(pose, left_handed: bool, min_confidence: float) -> tuple[float, float] | None:
    side = "left" if left_handed else "right"
    if pose is None or pose[f"{side}_shoulder_conf"] < min_confidence:
        return None
    return float(pose[f"{side}_shoulder_x"]), float(pose[f"{side}_shoulder_y"])


def nearest_row_within(table: pd.DataFrame, frame: int, max_gap: int):
    if table.empty:
        return None
    gaps = (table["frame"] - frame).abs()
    nearest = gaps.idxmin()
    return table.loc[nearest] if gaps[nearest] <= max_gap else None


def shuttle_pixel_near(shuttle: pd.DataFrame, frame: int, max_gap: int = SHUTTLE_MAX_FRAME_GAP) -> tuple[float, float] | None:
    row = nearest_row_within(shuttle[shuttle["visibility"] == 1], frame, max_gap)
    return None if row is None else (float(row["x"]), float(row["y"]))


def player_row_near(table: pd.DataFrame, player: str, frame: int, max_gap: int = FEET_MAX_FRAME_GAP):
    return nearest_row_within(table[table["player"] == player], frame, max_gap)


def estimate_hit_point(
    camera: Camera,
    frame: int,
    hitter: str,
    shuttle: pd.DataFrame,
    positions: pd.DataFrame,
    boxes: pd.DataFrame,
    poses: pd.DataFrame,
    left_handed: frozenset[str] = frozenset(),
    settings: ContactSettings = ContactSettings(),
) -> dict:
    feet_row = player_row_near(positions, hitter, frame, settings.feet_max_frame_gap)
    feet = None if feet_row is None else (float(feet_row["x_m"]), float(feet_row["y_m"]))
    row = {"frame": frame, "hitter": hitter, "feet_x": None if feet is None else feet[0], "feet_y": None if feet is None else feet[1]}
    fallback = {**row, "hit_x": row["feet_x"], "hit_y": row["feet_y"], "contact_z": None}
    if feet is None:
        return {**fallback, "method": "no feet"}
    shuttle_pixel = shuttle_pixel_near(shuttle, frame, settings.shuttle_max_frame_gap)
    if shuttle_pixel is None:
        return {**fallback, "method": "feet: no shuttle pixel"}
    box_row = player_row_near(boxes, hitter, frame, settings.feet_max_frame_gap)
    hitter_box = None if box_row is None else (box_row["left"], box_row["top"], box_row["right"], box_row["bottom"])
    pose = hitter_pose(poses[poses["frame"] == frame], hitter_box)
    shoulder_pixel = hitting_shoulder_pixel(pose, hitter in left_handed, settings.shoulder_min_confidence)
    if shoulder_pixel is None:
        return {**fallback, "method": "feet: no pose"}
    shoulder = shoulder_position(camera, shoulder_pixel, feet)
    origin, direction = pixel_ray(camera, shuttle_pixel)
    contact, how = contact_point(origin, direction, shoulder, settings)
    return {**row, "hit_x": float(contact[0]), "hit_y": float(contact[1]), "contact_z": float(contact[2]), "method": f"contact: {how}"}


def estimate_hit_points(
    camera: Camera,
    hits: pd.DataFrame,
    shuttle: pd.DataFrame,
    positions: pd.DataFrame,
    boxes: pd.DataFrame,
    poses: pd.DataFrame,
    left_handed: frozenset[str] = frozenset(),
    settings: ContactSettings = ContactSettings(),
) -> pd.DataFrame:
    rows = [
        estimate_hit_point(camera, int(hit.frame), hit.hitter, shuttle, positions, boxes, poses, left_handed, settings)
        for hit in hits.sort_values("frame").itertuples()
    ]
    return pd.DataFrame(rows, columns=CONTACT_COLUMNS)


def method_counts(contacts: pd.DataFrame) -> dict[str, int]:
    return {method: int(count) for method, count in contacts["method"].value_counts().sort_index().items()}


def pose_rows(frame_index: int, result) -> list[dict]:
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    keypoints = result.keypoints.xy.cpu().numpy()
    confidences = result.keypoints.conf.cpu().numpy()
    return [
        {
            "frame": frame_index,
            "left": float(box[0]), "top": float(box[1]), "right": float(box[2]), "bottom": float(box[3]),
            "left_shoulder_x": float(points[LEFT_SHOULDER_INDEX, 0]), "left_shoulder_y": float(points[LEFT_SHOULDER_INDEX, 1]),
            "left_shoulder_conf": float(confidence[LEFT_SHOULDER_INDEX]),
            "right_shoulder_x": float(points[RIGHT_SHOULDER_INDEX, 0]), "right_shoulder_y": float(points[RIGHT_SHOULDER_INDEX, 1]),
            "right_shoulder_conf": float(confidence[RIGHT_SHOULDER_INDEX]),
        }
        for box, points, confidence in zip(boxes, keypoints, confidences)
    ]


def load_pose_model(model_path: Path = POSE_MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"pose weights not found: {model_path} "
            f"(download with: cd {model_path.parent} && python -c \"from ultralytics import YOLO; YOLO('{model_path.name}')\")"
        )
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    numpy_print_options = np.get_printoptions()
    try:
        from ultralytics import YOLO

        return YOLO(str(model_path))
    finally:
        np.set_printoptions(**numpy_print_options)


def detect_poses(video_path: Path, frames, model_path: Path = POSE_MODEL_PATH) -> pd.DataFrame:
    model = load_pose_model(model_path)
    capture = open_video(Path(video_path))
    rows = []
    try:
        for frame_index in sorted(set(int(frame) for frame in frames)):
            frame = read_frame(capture, frame_index)
            if frame is not None:
                rows.extend(pose_rows(frame_index, model.predict(frame, imgsz=frame.shape[1], verbose=False)[0]))
    finally:
        capture.release()
    return pd.DataFrame(rows, columns=POSE_COLUMNS)


def contact_paths(video_path: Path, start_frame: int, end_frame: int, contact_dir: Path = CONTACT_DIR) -> tuple[Path, Path]:
    folder = contact_dir / video_id(video_path)
    return folder / f"poses_{start_frame}_{end_frame}.csv", folder / f"contacts_{start_frame}_{end_frame}.csv"


def require_file(path: Path, command: str) -> Path:
    if not Path(path).is_file():
        raise FileNotFoundError(f"file not found: {path} (run: {command})")
    return Path(path)


def range_inputs(video_path: Path, start_frame: int, end_frame: int) -> dict[str, pd.DataFrame]:
    arguments = f"{video_path} --start {start_frame} --end {end_frame}"
    hits_file, _ = hit_output_paths(video_path, start_frame, end_frame)
    files = {
        "hits": (hits_file, f"python -m vision.hits {arguments}"),
        "shuttle": (shuttle_positions_path(video_path, start_frame, end_frame), f"python -m vision.shuttle {arguments}"),
        "positions": (player_positions_path(video_path, start_frame, end_frame), f"python -m vision.players {arguments}"),
        "boxes": (boxes_path(video_path, start_frame, end_frame), f"python -m vision.players {arguments}"),
    }
    return {name: pd.read_csv(require_file(path, command)) for name, (path, command) in files.items()}


def video_fingerprint(video_path: Path) -> dict:
    path = Path(video_path)
    size = path.stat().st_size
    with path.open("rb") as video_file:
        head = video_file.read(FINGERPRINT_BYTES)
        video_file.seek(max(size - FINGERPRINT_BYTES, 0))
        tail = video_file.read(FINGERPRINT_BYTES)
    return {"size": size, "head_sha256": hashlib.sha256(head).hexdigest(), "tail_sha256": hashlib.sha256(tail).hexdigest()}


def source_record_path(output_path: Path) -> Path:
    return Path(output_path).with_name(Path(output_path).name + SOURCE_RECORD_SUFFIX)


def write_source_record(output_path: Path, video_path: Path, fingerprint: dict) -> Path:
    record_path = source_record_path(output_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps({"video": str(Path(video_path).resolve()), **fingerprint}, indent=2), encoding="utf-8")
    return record_path


def read_source_record(output_path: Path) -> dict | None:
    record_path = source_record_path(output_path)
    if not record_path.is_file():
        return None
    try:
        return json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_same_source(record: dict | None, fingerprint: dict) -> bool:
    return record is not None and all(record.get(key) == value for key, value in fingerprint.items())


def is_newer(path: Path, than: Path) -> bool:
    return Path(path).is_file() and Path(path).stat().st_mtime >= Path(than).stat().st_mtime


def poses_for_range(video_path: Path, start_frame: int, end_frame: int, hits: pd.DataFrame, model_path: Path = POSE_MODEL_PATH) -> pd.DataFrame:
    poses_file, _ = contact_paths(video_path, start_frame, end_frame)
    hits_file, _ = hit_output_paths(video_path, start_frame, end_frame)
    fingerprint = video_fingerprint(video_path)
    if is_newer(poses_file, hits_file) and is_same_source(read_source_record(poses_file), fingerprint):
        return pd.read_csv(poses_file)
    poses = detect_poses(video_path, hits["frame"], model_path)
    poses_file.parent.mkdir(parents=True, exist_ok=True)
    poses.to_csv(poses_file, index=False)
    write_source_record(poses_file, video_path, fingerprint)
    return poses


def contacts_for_range(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    camera: Camera | None = None,
    left_handed: frozenset[str] = frozenset(),
    settings: ContactSettings | None = None,
    model_path: Path = POSE_MODEL_PATH,
) -> pd.DataFrame:
    require_video_file(video_path)
    camera = load_camera(camera_path(Path(video_path))) if camera is None else camera
    settings = contact_settings_for_video(video_fps_and_width(Path(video_path))[0]) if settings is None else settings
    inputs = range_inputs(Path(video_path), start_frame, end_frame)
    poses = poses_for_range(Path(video_path), start_frame, end_frame, inputs["hits"], model_path)
    return estimate_hit_points(camera, inputs["hits"], inputs["shuttle"], inputs["positions"], inputs["boxes"], poses, left_handed, settings)


def labelled_hit_points(shots: pd.DataFrame, start_frame: int, end_frame: int, homography: np.ndarray, scale: float = 1.0) -> pd.DataFrame:
    in_range = shots.dropna(subset=["frame_num", "hit_x", "hit_y"])
    in_range = in_range[(in_range["frame_num"] >= start_frame) & (in_range["frame_num"] <= end_frame)]
    pixels = in_range[["hit_x", "hit_y"]].to_numpy(float) * scale
    meters = pixels_to_court(homography, pixels) if len(pixels) else np.zeros((0, 2))
    return pd.DataFrame(
        {
            "frame": in_range["frame_num"].astype(int).to_numpy(),
            "label_pixel_x": pixels[:, 0] if len(pixels) else [],
            "label_pixel_y": pixels[:, 1] if len(pixels) else [],
            "label_x": meters[:, 0],
            "label_y": meters[:, 1],
        }
    )


def contact_comparison(contacts: pd.DataFrame, labels: pd.DataFrame, tolerance: int = MATCH_TOLERANCE_FRAMES) -> pd.DataFrame:
    pairs = match_hits(contacts["frame"].astype(int).tolist(), labels["frame"].astype(int).tolist(), tolerance)
    contact_rows = contacts.set_index("frame")
    label_rows = labels.set_index("frame")
    rows = []
    for detected, labelled in pairs:
        contact, label = contact_rows.loc[detected], label_rows.loc[labelled]
        rows.append(
            {
                "frame": detected,
                "label_frame": labelled,
                "hitter": contact["hitter"],
                "method": contact["method"],
                "contact_error_m": float(np.hypot(contact["hit_x"] - label["label_x"], contact["hit_y"] - label["label_y"])),
                "feet_error_m": float(np.hypot(contact["feet_x"] - label["label_x"], contact["feet_y"] - label["label_y"])),
            }
        )
    return pd.DataFrame(rows, columns=["frame", "label_frame", "hitter", "method", "contact_error_m", "feet_error_m"])


def comparison_lines(comparison: pd.DataFrame, name: str) -> list[str]:
    if comparison.empty:
        return [f"{name}: no matched hits"]
    closer = int((comparison["contact_error_m"] < comparison["feet_error_m"]).sum())
    return [
        f"{name}: {len(comparison)} matched hits, distance to the labelled hit point: "
        f"contact median {comparison['contact_error_m'].median():.2f} m (mean {comparison['contact_error_m'].mean():.2f}), "
        f"feet median {comparison['feet_error_m'].median():.2f} m (mean {comparison['feet_error_m'].mean():.2f}); "
        f"contact closer on {closer} of {len(comparison)}"
    ]


def crop_around(frame: np.ndarray, centre, caption: str) -> np.ndarray:
    height, width = frame.shape[:2]
    left = int(np.clip(centre[0] - TILE_WIDTH // 2, 0, max(width - TILE_WIDTH, 0)))
    top = int(np.clip(centre[1] - TILE_HEIGHT // 2, 0, max(height - TILE_HEIGHT, 0)))
    tile = frame[top : top + TILE_HEIGHT, left : left + TILE_WIDTH].copy()
    cv2.putText(tile, caption, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR_BGR, 2)
    return tile


def marked(frame: np.ndarray, pixel, color, label: str) -> tuple[int, int]:
    point = (int(round(pixel[0])), int(round(pixel[1])))
    cv2.circle(frame, point, 7, color, 2)
    cv2.putText(frame, label, (point[0] + 8, point[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    return point


def contact_tile(frame: np.ndarray, camera: Camera, contact, shuttle_pixel, label_pixel, caption: str) -> np.ndarray:
    drawn = frame.copy()
    floor_pixel = project_points(camera, [[contact["hit_x"], contact["hit_y"], 0.0]])[0]
    feet_pixel = project_points(camera, [[contact["feet_x"], contact["feet_y"], 0.0]])[0]
    if pd.notna(contact["contact_z"]):
        air_pixel = project_points(camera, [[contact["hit_x"], contact["hit_y"], contact["contact_z"]]])[0]
        cv2.line(drawn, tuple(np.round(floor_pixel).astype(int)), tuple(np.round(air_pixel).astype(int)), CONTACT_COLOR_BGR, 1, cv2.LINE_AA)
    marked(drawn, feet_pixel, FEET_COLOR_BGR, "feet")
    marked(drawn, label_pixel, LABEL_COLOR_BGR, "label")
    centre = marked(drawn, floor_pixel, CONTACT_COLOR_BGR, "contact")
    if shuttle_pixel is not None:
        marked(drawn, shuttle_pixel, SHUTTLE_COLOR_BGR, "shuttle")
    return crop_around(drawn, centre, caption)


def write_contact_spot_check(
    video_path: Path,
    camera: Camera,
    sampled: pd.DataFrame,
    output_path: Path,
) -> Path:
    capture = open_video(Path(video_path))
    tiles = []
    try:
        for hit in sampled.itertuples():
            frame = read_frame(capture, int(hit.frame))
            if frame is None:
                continue
            caption = f"f{hit.frame} {hit.hitter} {hit.method}: contact {hit.contact_error_m:.2f} m, feet {hit.feet_error_m:.2f} m"
            contact = {"hit_x": hit.hit_x, "hit_y": hit.hit_y, "contact_z": hit.contact_z, "feet_x": hit.feet_x, "feet_y": hit.feet_y}
            shuttle_pixel = None if pd.isna(hit.shuttle_x) else (hit.shuttle_x, hit.shuttle_y)
            tiles.append(contact_tile(frame, camera, contact, shuttle_pixel, (hit.label_pixel_x, hit.label_pixel_y), caption))
    finally:
        capture.release()
    while len(tiles) % 2:
        tiles.append(np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack([np.hstack(tiles[index : index + 2]) for index in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return output_path


def compare_ranges(
    video_path: Path,
    ranges: list[tuple[int, int, int]],
    shuttleset_match_id: str,
    camera: Camera,
    left_handed: frozenset[str] = frozenset(),
    settings: ContactSettings | None = None,
    model_path: Path = POSE_MODEL_PATH,
) -> pd.DataFrame:
    homography = load_calibration(calibration_path(Path(video_path)))["pixel_to_court_homography"]
    scale = label_scale(camera.frame_size[0])
    compared = []
    for start_frame, end_frame, set_no in ranges:
        contacts = contacts_for_range(video_path, start_frame, end_frame, camera, left_handed, settings, model_path)
        labels = labelled_hit_points(load_set_shots(SHUTTLESET_SET_DIR, shuttleset_match_id, set_no), start_frame, end_frame, homography, scale)
        shuttle = range_inputs(Path(video_path), start_frame, end_frame)["shuttle"]
        comparison = contact_comparison(contacts, labels).assign(set_no=set_no)
        comparison = comparison.merge(contacts[["frame", "hit_x", "hit_y", "contact_z", "feet_x", "feet_y"]], on="frame")
        comparison = comparison.merge(labels.rename(columns={"frame": "label_frame"})[["label_frame", "label_pixel_x", "label_pixel_y"]], on="label_frame")
        gap = (settings or contact_settings_for_video(video_fps_and_width(Path(video_path))[0])).shuttle_max_frame_gap
        shuttle_pixels = [shuttle_pixel_near(shuttle, int(frame), gap) for frame in comparison["frame"]]
        comparison["shuttle_x"] = [None if pixel is None else pixel[0] for pixel in shuttle_pixels]
        comparison["shuttle_y"] = [None if pixel is None else pixel[1] for pixel in shuttle_pixels]
        compared.append(comparison)
    return pd.concat(compared, ignore_index=True) if compared else pd.DataFrame()


def spot_check_sample(comparison: pd.DataFrame, hit_count: int = SPOT_CHECK_HITS, seed: int = SPOT_CHECK_SEED) -> pd.DataFrame:
    picks = np.random.default_rng(seed).choice(len(comparison), size=min(hit_count, len(comparison)), replace=False)
    return comparison.iloc[sorted(picks)]


HIT_SPOT_CHECK_DIR = Path("data/checks/hits")
HIT_SPOT_CHECK_OFFSET_FRAMES = 6
NEAR_BOX_COLOR_BGR, FAR_BOX_COLOR_BGR = (255, 160, 0), (0, 0, 255)


def hit_spot_check_path(video_path: Path, start_frame: int, end_frame: int, checks_dir: Path = HIT_SPOT_CHECK_DIR) -> Path:
    return checks_dir / f"{video_id(video_path)}_hit_spot_check_{start_frame}_{end_frame}.jpg"


def spot_check_hits(hits: pd.DataFrame, hit_count: int, seed: int = SPOT_CHECK_SEED) -> pd.DataFrame:
    ordered = hits.sort_values("frame").reset_index(drop=True)
    picks = np.random.default_rng(seed).choice(len(ordered), size=min(hit_count, len(ordered)), replace=False)
    return ordered.iloc[sorted(picks)]


def draw_box(frame: np.ndarray, box_row, color) -> None:
    if box_row is not None:
        cv2.rectangle(frame, (int(box_row["left"]), int(box_row["top"])), (int(box_row["right"]), int(box_row["bottom"])), color, 2)


def hit_tile(frame: np.ndarray, frame_index: int, centre, inputs: dict, contact, homography: np.ndarray, caption: str, trail_frames: int) -> np.ndarray:
    drawn = frame.copy()
    for player, color in (("me", NEAR_BOX_COLOR_BGR), ("opp", FAR_BOX_COLOR_BGR)):
        draw_box(drawn, player_row_near(inputs["boxes"], player, frame_index, 0), color)
    shuttle = inputs["shuttle"]
    trail = shuttle[(shuttle["frame"] > frame_index - trail_frames) & (shuttle["frame"] <= frame_index) & (shuttle["visibility"] == 1)]
    points = trail[["x", "y"]].to_numpy(int)
    for start_point, end_point in zip(points, points[1:]):
        cv2.line(drawn, tuple(start_point), tuple(end_point), SHUTTLE_COLOR_BGR, 2, cv2.LINE_AA)
    if len(points):
        cv2.circle(drawn, tuple(points[-1]), 7, SHUTTLE_COLOR_BGR, 2)
    if contact is not None and pd.notna(contact["hit_x"]):
        floor_pixel = court_to_pixels(homography, [[contact["hit_x"], contact["hit_y"]]])[0]
        marked(drawn, floor_pixel, CONTACT_COLOR_BGR, "contact")
    return crop_around(drawn, centre, caption)


def spot_check_centre(shuttle_pixel, hitter_box) -> tuple[float, float]:
    if shuttle_pixel is not None:
        return shuttle_pixel
    if hitter_box is not None:
        return (hitter_box["left"] + hitter_box["right"]) / 2, (hitter_box["top"] + hitter_box["bottom"]) / 2
    return 0.0, 0.0


def write_hit_spot_check(video_path: Path, start_frame: int, end_frame: int, hit_count: int, output_path: Path | None = None) -> Path:
    inputs = range_inputs(Path(video_path), start_frame, end_frame)
    homography = load_calibration(calibration_path(Path(video_path)))["pixel_to_court_homography"]
    _, contacts_file = contact_paths(Path(video_path), start_frame, end_frame)
    contacts = pd.read_csv(contacts_file).set_index("frame") if contacts_file.is_file() else None
    fps = video_fps_and_width(Path(video_path))[0]
    offset = frames_for_fps(HIT_SPOT_CHECK_OFFSET_FRAMES, fps)
    capture = open_video(Path(video_path))
    rows = []
    try:
        for hit in spot_check_hits(inputs["hits"], hit_count).itertuples():
            frame_index = int(hit.frame)
            shuttle_pixel = shuttle_pixel_near(inputs["shuttle"], frame_index, frames_for_fps(SHUTTLE_MAX_FRAME_GAP, fps))
            box = player_row_near(inputs["boxes"], hit.hitter, frame_index, 0)
            centre = spot_check_centre(shuttle_pixel, box)
            contact = contacts.loc[frame_index] if contacts is not None and frame_index in contacts.index else None
            tiles = []
            for shift, label in ((-offset, "before"), (0, "hit"), (offset, "after")):
                frame = read_frame(capture, frame_index + shift)
                if frame is None:
                    tiles.append(np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8))
                    continue
                caption = f"r{hit.rally} f{frame_index} {hit.hitter} score {hit.score:.2f}" if shift == 0 else f"{label} f{frame_index + shift}"
                tiles.append(hit_tile(frame, frame_index + shift, centre, inputs, contact if shift == 0 else None, homography, caption, 3 * offset))
            rows.append(np.hstack(tiles))
    finally:
        capture.release()
    output_path = hit_spot_check_path(Path(video_path), start_frame, end_frame) if output_path is None else Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return output_path


def parse_range(text: list[str]) -> tuple[int, int, int]:
    start_frame, end_frame, set_no = (int(value) for value in text)
    if set_no not in (1, 2, 3):
        raise ValueError(f"set number must be 1, 2 or 3, got {set_no}")
    return start_frame, end_frame, set_no


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vision.contact",
        description="Estimate the racket contact point of every detected hit of tracked ranges.",
    )
    parser.add_argument("video", type=Path, help="video with a camera (python -m vision.camera) and tracked ranges")
    parser.add_argument("--range", nargs=3, action="append", required=True, metavar=("START", "END", "SET"), help="a tracked range and its set number; repeatable")
    parser.add_argument("--left-handed", default="", help="comma-separated left-handed players: me, opp")
    parser.add_argument("--model", type=Path, default=POSE_MODEL_PATH, help="YOLO pose weights")
    parser.add_argument("--compare", metavar="SHUTTLESET_MATCH", help="compare with ShuttleSet's labelled hit points and draw the spot-check image")
    parser.add_argument("--hit-spot-check", type=int, metavar="N", help="also draw N random hits of each range (before, at and after the hit) for checking by eye")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    try:
        ranges = [parse_range(text) for text in parsed.range]
        left_handed = frozenset(player for player in parsed.left_handed.split(",") if player)
        require_video_file(parsed.video)
        camera = load_camera(camera_path(parsed.video))
        for start_frame, end_frame, _ in ranges:
            contacts = contacts_for_range(parsed.video, start_frame, end_frame, camera, left_handed, model_path=parsed.model)
            _, contacts_file = contact_paths(parsed.video, start_frame, end_frame)
            contacts_file.parent.mkdir(parents=True, exist_ok=True)
            contacts.to_csv(contacts_file, index=False)
            print(f"{start_frame}-{end_frame}: hits {len(contacts)}, " + ", ".join(f"{method}: {count}" for method, count in method_counts(contacts).items()))
            print(f"saved {contacts_file}")
            if parsed.hit_spot_check:
                print(f"saved {write_hit_spot_check(parsed.video, start_frame, end_frame, parsed.hit_spot_check)}")
        if parsed.compare:
            comparison = compare_ranges(parsed.video, ranges, parsed.compare, camera, left_handed, model_path=parsed.model)
            for set_name, set_rows in (("sets 1-2", comparison[comparison["set_no"] < 3]), ("set 3", comparison[comparison["set_no"] == 3]), ("all", comparison)):
                for line in comparison_lines(set_rows, set_name):
                    print(line)
            output_path = CHECKS_DIR / f"{video_id(parsed.video)}_contact_spot_check.jpg"
            print(f"saved {write_contact_spot_check(parsed.video, camera, spot_check_sample(comparison), output_path)}")
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
