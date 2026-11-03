import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vision.camera import (
    NET_POST_HEIGHT_M,
    Camera,
    camera_from_dict,
    camera_reference_meters,
    camera_to_dict,
    fit_camera,
    floor_homography,
    floor_line_offsets,
    intrinsic_matrix,
    net_post_top_meters,
    pixel_ray,
    project_points,
    ray_point_at_height,
    reprojection_errors,
)
from vision.court import court_corner_meters, load_calibration, pixels_to_court
from tests.results_pages import docs_on_disk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_VIDEO_PATH = PROJECT_ROOT / "data" / "match.mp4"
REAL_CAMERA_PATH = PROJECT_ROOT / "data" / "calibration" / "match" / "camera.json"
REAL_COURT_PATH = PROJECT_ROOT / "data" / "calibration" / "match" / "court.json"
RESULTS_PATH = PROJECT_ROOT / "docs" / "results" / "contact.md"
TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b"
FRAME_SIZE = (1280, 720)
DOCUMENTED_NET_POSTS = [[364.5, 308.5], [910.0, 306.5]]


def looking_at(centre, target, focal_px: float = 2000.0) -> Camera:
    forward = np.asarray(target, dtype=float) - np.asarray(centre, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    return Camera(intrinsic_matrix(focal_px, FRAME_SIZE), rotation, -rotation @ np.asarray(centre, dtype=float), FRAME_SIZE)


BROADCAST_LIKE = looking_at((3.05, -20.0, 8.0), (3.05, 6.7, 0.0))


def test_reference_points_are_the_four_corners_on_the_floor_and_the_two_post_tops():
    reference = camera_reference_meters()
    assert reference.shape == (6, 3)
    assert np.allclose(reference[:4, :2], court_corner_meters()) and np.allclose(reference[:4, 2], 0.0)
    assert np.allclose(reference[4:], net_post_top_meters()) and np.allclose(reference[4:, 2], NET_POST_HEIGHT_M) and NET_POST_HEIGHT_M == 1.55


def test_projection_puts_the_court_in_front_of_the_synthetic_camera_with_x_to_the_right_and_up_on_top():
    pixels = project_points(BROADCAST_LIKE, camera_reference_meters())
    near_left, near_right, far_right, far_left = pixels[:4]
    assert near_left[0] < near_right[0] and far_left[0] < far_right[0] and far_left[1] < near_left[1]
    assert pixels[4][1] < project_points(BROADCAST_LIKE, [[0.0, 6.7, 0.0]])[0][1]


@pytest.mark.parametrize("focal_px", [900.0, 2000.0, 3500.0])
def test_fit_recovers_focal_length_and_pose_from_six_exact_points(focal_px):
    truth = looking_at((2.0, -18.0, 7.0), (3.05, 7.0, 0.0), focal_px)
    pixels = project_points(truth, camera_reference_meters())
    fitted = fit_camera(camera_reference_meters(), pixels, FRAME_SIZE)
    assert fitted.focal_px == pytest.approx(focal_px, rel=1e-3)
    assert fitted.centre == pytest.approx(truth.centre, abs=0.02)
    assert reprojection_errors(fitted, camera_reference_meters(), pixels).max() < 0.05


def test_fit_with_noisy_clicks_stays_close():
    pixels = project_points(BROADCAST_LIKE, camera_reference_meters()) + np.random.default_rng(3).normal(0.0, 1.0, (6, 2))
    fitted = fit_camera(camera_reference_meters(), pixels, FRAME_SIZE)
    assert fitted.focal_px == pytest.approx(2000.0, rel=0.1) and np.linalg.norm(fitted.centre - BROADCAST_LIKE.centre) < 2.5


def test_fewer_than_six_points_cannot_fit_a_focal_length():
    with pytest.raises(ValueError, match="at least 6 points"):
        fit_camera(camera_reference_meters()[:4], project_points(BROADCAST_LIKE, camera_reference_meters()[:4]), FRAME_SIZE)


def test_pixel_ray_passes_through_the_world_point_and_meets_the_floor_there():
    for point in ([1.0, 2.0, 0.0], [5.0, 11.0, 0.0], [3.0, 8.0, 2.5]):
        origin, direction = pixel_ray(BROADCAST_LIKE, project_points(BROADCAST_LIKE, [point])[0])
        assert np.linalg.norm(direction) == pytest.approx(1.0) and origin == pytest.approx(BROADCAST_LIKE.centre)
        along = np.dot(np.asarray(point) - origin, direction)
        assert origin + along * direction == pytest.approx(point, abs=1e-6)
        assert ray_point_at_height(origin, direction, point[2]) == pytest.approx(point, abs=1e-6)
    assert ray_point_at_height(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 0.0) is None
    assert ray_point_at_height(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), 0.0) is None


def test_floor_homography_maps_pixels_back_to_court_meters():
    floor_points = np.array([[0.0, 0.0], [6.1, 13.4], [2.0, 9.0], [4.5, 1.5]])
    pixels = project_points(BROADCAST_LIKE, np.column_stack([floor_points, np.zeros(4)]))
    assert pixels_to_court(floor_homography(BROADCAST_LIKE), pixels) == pytest.approx(floor_points, abs=1e-6)


def test_camera_dict_round_trip_keeps_the_projection():
    pixels = project_points(BROADCAST_LIKE, camera_reference_meters())
    data = json.loads(json.dumps(camera_to_dict(BROADCAST_LIKE, pixels[:4], pixels[4:], 12, Path("video.mp4"))))
    restored = camera_from_dict(data)
    assert np.allclose(restored.projection, BROADCAST_LIKE.projection) and restored.frame_size == FRAME_SIZE
    assert data["reference_point_order"][4:] == ["left_net_post_top", "right_net_post_top"] and max(data["reference_errors_px"]) < 1e-6


def test_command_without_a_court_calibration_exits_one(tmp_path):
    video = tmp_path / "no_calibration.mp4"
    video.write_bytes(b"")
    result = subprocess.run([sys.executable, "-m", "vision.camera", str(video), "--net-posts", "1,2", "3,4"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 1 and result.stderr.startswith("error: ")


real_camera_required = pytest.mark.skipif(not (REAL_CAMERA_PATH.exists() and REAL_COURT_PATH.exists() and REAL_VIDEO_PATH.exists()), reason="needs data/match.mp4 with its court and camera calibration (python -m vision.camera)")


@real_camera_required
def test_refitting_from_the_documented_clicks_reproduces_the_shipped_camera_and_results_doc():
    from vision.alignment import open_video, read_frame

    shipped = json.loads(REAL_CAMERA_PATH.read_text())
    court = load_calibration(REAL_COURT_PATH)
    pixels = np.vstack([court["corner_pixels"], DOCUMENTED_NET_POSTS])
    assert np.allclose(shipped["reference_pixels"], pixels)
    capture = open_video(REAL_VIDEO_PATH)
    frame = read_frame(capture, court["frame_index"])
    capture.release()
    camera = fit_camera(camera_reference_meters(), pixels, (frame.shape[1], frame.shape[0]))
    assert camera.focal_px == pytest.approx(shipped["focal_px"], rel=1e-6) and np.allclose(camera.rotation, shipped["rotation"], atol=1e-6)
    errors = reprojection_errors(camera, camera_reference_meters(), pixels)
    offsets = np.abs(floor_line_offsets(frame, camera))
    doc = RESULTS_PATH.read_text()
    assert f"| Focal length | {camera.focal_px:.0f} px (horizontal field of view {np.degrees(2 * np.arctan(frame.shape[1] / 2 / camera.focal_px)):.0f}°) |" in doc
    x, y, z = camera.centre
    assert f"x {x:.2f}, y {y:.1f}".replace("-", "−") in doc and f"z {z:.2f} m" in doc
    assert f"corners {errors[0]:.1f}, {errors[1]:.1f}, {errors[2]:.1f}, {errors[3]:.1f} px; post tops {errors[4]:.1f}, {errors[5]:.1f} px (RMS {np.sqrt(np.mean(errors ** 2)):.1f})" in doc
    assert f"| Court lines ({len(offsets)} samples" in doc and f"median {np.median(offsets):.1f} px, 95th percentile {np.percentile(offsets, 95):.1f} px" in doc
    assert np.median(offsets) <= 2.0 and errors.max() < 4.0


def test_camera_source_has_no_comments_or_docstrings_and_docs_have_no_task_ids():
    source = (PROJECT_ROOT / "vision" / "camera.py").read_text()
    assert '"""' not in source and "'''" not in source and not re.search(r"(?m)^\s*#", source)
    for doc in docs_on_disk(("docs/vision/camera.md", "docs/results/contact.md")):
        assert not re.search(TASK_ID_PATTERN, (PROJECT_ROOT / doc).read_text())
    assert "Examples need:" not in (PROJECT_ROOT / "docs" / "vision" / "camera.md").read_text()
