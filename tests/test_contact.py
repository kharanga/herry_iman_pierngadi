import filecmp
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from vision.camera import Camera, camera_path, intrinsic_matrix, load_camera, pixel_ray, project_points
from vision.contact import (
    CONTACT_COLUMNS,
    INTERSECTION_CHOICE,
    INTERSECTION_CHOICES,
    POSE_COLUMNS,
    REACH_RADIUS_M,
    SHUTTLE_MAX_FRAME_GAP,
    ContactSettings,
    box_overlap,
    chosen_intersection,
    closest_point_to_vertical_line,
    compare_ranges,
    comparison_lines,
    contact_comparison,
    contact_paths,
    contact_point,
    contacts_for_range,
    estimate_hit_point,
    estimate_hit_points,
    hitter_pose,
    hitting_shoulder_pixel,
    is_newer,
    main,
    method_counts,
    parse_range,
    ray_sphere_distances,
    shoulder_position,
    shuttle_pixel_near,
    spot_check_sample,
    write_contact_spot_check,
)
from vision.hits import output_paths as hit_output_paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = Path("data/match.mp4")
RESULTS_PATH = PROJECT_ROOT / "docs" / "results" / "contact.md"
SHUTTLESET_SET_DIR = PROJECT_ROOT / "data" / "CoachAI-Projects" / "ShuttleSet" / "set"
AXELSEN_GINTING = "Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals"
TRACKED_RANGES = [(14201, 14947, 1), (16551, 17343, 1), (81360, 82116, 2), (107560, 108103, 3), (114673, 116641, 3)]
TUNING_RADII = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8)
TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b"
FRAME_SIZE = (1280, 720)


def looking_at(centre, target, focal_px: float = 2000.0) -> Camera:
    forward = np.asarray(target, dtype=float) - np.asarray(centre, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    return Camera(intrinsic_matrix(focal_px, FRAME_SIZE), rotation, -rotation @ np.asarray(centre, dtype=float), FRAME_SIZE)


CAMERA = looking_at((3.05, -20.0, 8.0), (3.05, 6.7, 0.0))


def test_ray_sphere_distances_cover_crossing_tangent_miss_behind_and_inside():
    origin, direction = np.zeros(3), np.array([1.0, 0.0, 0.0])
    assert ray_sphere_distances(origin, direction, np.array([5.0, 0.0, 0.0]), 1.0) == pytest.approx([4.0, 6.0])
    assert ray_sphere_distances(origin, direction, np.array([5.0, 1.0, 0.0]), 1.0) == pytest.approx([5.0, 5.0])
    assert ray_sphere_distances(origin, direction, np.array([5.0, 2.0, 0.0]), 1.0) == []
    assert ray_sphere_distances(origin, direction, np.array([-5.0, 0.0, 0.0]), 1.0) == []
    assert ray_sphere_distances(origin, direction, np.array([0.5, 0.0, 0.0]), 1.0) == pytest.approx([1.5])


def test_closest_point_to_a_vertical_line():
    direction = np.array([1.0, 0.0, -1.0]) / np.sqrt(2)
    assert closest_point_to_vertical_line(np.array([0.0, 0.0, 10.0]), direction, (5.0, 0.0)) == pytest.approx([5.0, 0.0, 5.0])
    assert closest_point_to_vertical_line(np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, -1.0]), (5.0, 0.0)) == pytest.approx([0.0, 0.0, 10.0])


def test_shoulder_position_recovers_a_point_above_the_feet():
    shoulder = np.array([2.4, 10.5, 1.45])
    pixel = project_points(CAMERA, [shoulder])[0]
    assert shoulder_position(CAMERA, pixel, (2.4, 10.5)) == pytest.approx(shoulder, abs=1e-6)


def test_contact_is_on_the_shuttle_ray_at_reach_distance_and_the_one_farther_from_the_camera():
    shoulder = np.array([2.4, 10.5, 1.45])
    for offset in ([0.0, 0.6, 0.8], [0.0, -0.6, 0.8], [0.7, 0.0, 0.7]):
        shuttle = shoulder + np.asarray(offset) / np.linalg.norm(offset) * REACH_RADIUS_M
        origin, direction = pixel_ray(CAMERA, project_points(CAMERA, [shuttle])[0])
        contact, how = contact_point(origin, direction, shoulder)
        assert how == "sphere"
        assert np.linalg.norm(contact - shoulder) == pytest.approx(REACH_RADIUS_M)
        assert np.linalg.norm(np.cross(contact - origin, direction)) == pytest.approx(0.0, abs=1e-9)
        both = [origin + distance * direction for distance in ray_sphere_distances(origin, direction, shoulder, REACH_RADIUS_M)]
        assert np.linalg.norm(contact - origin) == pytest.approx(max(np.linalg.norm(point - origin) for point in both))
    assert INTERSECTION_CHOICE == "farther_from_camera" and REACH_RADIUS_M == 1.0


def test_intersection_rules_and_an_unknown_rule():
    points = [np.array([1.0, 6.0, 1.0]), np.array([1.0, 9.0, 1.0])]
    camera_centre = np.array([1.0, -20.0, 8.0])
    assert chosen_intersection(points, camera_centre, "nearer_camera") is points[0]
    assert chosen_intersection(points, camera_centre, "farther_from_camera") is points[1]
    assert chosen_intersection(points, camera_centre, "toward_net") is points[0]
    assert INTERSECTION_CHOICES == ("toward_net", "nearer_camera", "farther_from_camera")
    with pytest.raises(ValueError, match="intersection choice"):
        chosen_intersection(points, camera_centre, "random")


def test_heights_outside_the_range_and_a_missed_sphere_use_the_closest_point_to_the_shoulder():
    shoulder = np.array([2.4, 10.5, 1.45])
    shuttle = shoulder + np.array([0.0, 0.6, 0.8])
    origin, direction = pixel_ray(CAMERA, project_points(CAMERA, [shuttle])[0])
    contact, how = contact_point(origin, direction, shoulder, ContactSettings(max_height_m=0.5))
    closest = origin + np.dot(shoulder - origin, direction) * direction
    assert how == "closest to sphere" and contact == pytest.approx(closest)
    far_shuttle = shoulder + np.array([2.5, 0.0, 0.5])
    origin, direction = pixel_ray(CAMERA, project_points(CAMERA, [far_shuttle])[0])
    contact, how = contact_point(origin, direction, shoulder)
    assert how == "closest to sphere" and np.linalg.norm(contact - shoulder) > REACH_RADIUS_M
    assert np.dot(contact - shoulder, direction) == pytest.approx(0.0, abs=1e-9)


def test_box_overlap_and_the_pose_that_belongs_to_the_hitter():
    assert box_overlap((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_overlap((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_overlap((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    poses = pose_table([(100, (0, 0, 10, 10)), (100, (40, 40, 60, 60))])
    assert hitter_pose(poses, (41, 41, 61, 61))["left"] == 40
    assert hitter_pose(poses, (8, 8, 30, 30)) is None
    assert hitter_pose(poses.iloc[0:0], (0, 0, 10, 10)) is None and hitter_pose(poses, None) is None
    assert hitter_pose(poses, (np.nan, 0, 10, 10)) is None


def pose_table(boxes, right=(500.0, 300.0, 0.9), left=(480.0, 300.0, 0.9)) -> pd.DataFrame:
    return pd.DataFrame(
        [{"frame": frame, "left": box[0], "top": box[1], "right": box[2], "bottom": box[3],
          "left_shoulder_x": left[0], "left_shoulder_y": left[1], "left_shoulder_conf": left[2],
          "right_shoulder_x": right[0], "right_shoulder_y": right[1], "right_shoulder_conf": right[2]} for frame, box in boxes],
        columns=POSE_COLUMNS)


def test_hitting_shoulder_follows_handedness_and_confidence():
    pose = pose_table([(100, (0, 0, 10, 10))], right=(500.0, 300.0, 0.9), left=(480.0, 310.0, 0.2)).iloc[0]
    assert hitting_shoulder_pixel(pose, False, 0.3) == (500.0, 300.0)
    assert hitting_shoulder_pixel(pose, True, 0.3) is None and hitting_shoulder_pixel(pose, True, 0.1) == (480.0, 310.0)
    assert hitting_shoulder_pixel(None, False, 0.3) is None


def test_shuttle_pixel_is_the_nearest_visible_one_within_two_frames():
    shuttle = pd.DataFrame({"frame": [100, 101, 103, 110], "x": [1.0, 2.0, 3.0, 4.0], "y": [5.0, 6.0, 7.0, 8.0], "visibility": [0, 0, 1, 1]})
    assert SHUTTLE_MAX_FRAME_GAP == 2
    assert shuttle_pixel_near(shuttle, 101) == (3.0, 7.0) and shuttle_pixel_near(shuttle, 100) is None
    assert shuttle_pixel_near(shuttle, 112) == (4.0, 8.0) and shuttle_pixel_near(shuttle, 113) is None


def synthetic_hit_inputs(shoulder, shuttle_point, feet=(2.4, 10.5)):
    shoulder_pixel = project_points(CAMERA, [shoulder])[0]
    shuttle_pixel = project_points(CAMERA, [shuttle_point])[0]
    box = (shoulder_pixel[0] - 40, shoulder_pixel[1] - 30, shoulder_pixel[0] + 40, shoulder_pixel[1] + 120)
    shuttle = pd.DataFrame({"frame": [100], "x": [shuttle_pixel[0]], "y": [shuttle_pixel[1]], "visibility": [1]})
    positions = pd.DataFrame({"frame": [100], "player": ["opp"], "x_m": [feet[0]], "y_m": [feet[1]]})
    boxes = pd.DataFrame({"frame": [100], "player": ["opp"], "left": [box[0]], "top": [box[1]], "right": [box[2]], "bottom": [box[3]]})
    poses = pose_table([(100, box)], right=(shoulder_pixel[0], shoulder_pixel[1], 0.9))
    return shuttle, positions, boxes, poses


def test_estimate_hit_point_gives_the_contact_floor_point_or_a_named_fallback():
    shoulder = np.array([2.4, 10.5, 1.45])
    contact = shoulder + np.array([0.0, 0.6, 0.8])
    shuttle, positions, boxes, poses = synthetic_hit_inputs(shoulder, contact)
    estimate = estimate_hit_point(CAMERA, 100, "opp", shuttle, positions, boxes, poses)
    assert estimate["method"] == "contact: sphere" and (estimate["feet_x"], estimate["feet_y"]) == (2.4, 10.5)
    assert np.linalg.norm(np.array([estimate["hit_x"], estimate["hit_y"], estimate["contact_z"]]) - shoulder) == pytest.approx(1.0)
    pixel = project_points(CAMERA, [[estimate["hit_x"], estimate["hit_y"], estimate["contact_z"]]])[0]
    assert pixel == pytest.approx(shuttle[["x", "y"]].to_numpy()[0], abs=1e-6)
    no_shuttle = estimate_hit_point(CAMERA, 100, "opp", shuttle.assign(visibility=0), positions, boxes, poses)
    no_pose = estimate_hit_point(CAMERA, 100, "opp", shuttle, positions, boxes, poses.iloc[0:0])
    low_confidence = estimate_hit_point(CAMERA, 100, "opp", shuttle, positions, boxes, poses.assign(right_shoulder_conf=0.1))
    left_handed = estimate_hit_point(CAMERA, 100, "opp", shuttle, positions, boxes, poses.assign(left_shoulder_conf=0.1), frozenset({"opp"}))
    no_feet = estimate_hit_point(CAMERA, 100, "me", shuttle, positions, boxes, poses)
    for fallback, method in ((no_shuttle, "feet: no shuttle pixel"), (no_pose, "feet: no pose"), (low_confidence, "feet: no pose"), (left_handed, "feet: no pose")):
        assert fallback["method"] == method and (fallback["hit_x"], fallback["hit_y"], fallback["contact_z"]) == (2.4, 10.5, None)
    assert no_feet["method"] == "no feet" and no_feet["hit_x"] is None
    hits = pd.DataFrame({"frame": [100, 100], "hitter": ["opp", "me"]})
    table = estimate_hit_points(CAMERA, hits, shuttle, positions, boxes, poses)
    assert list(table.columns) == CONTACT_COLUMNS and method_counts(table) == {"contact: sphere": 1, "no feet": 1}


def test_comparison_matches_within_five_frames_and_summarises():
    contacts = pd.DataFrame({"frame": [100, 130, 200], "hitter": ["me", "opp", "me"], "hit_x": [1.0, 2.0, 3.0], "hit_y": [1.0, 10.0, 2.0],
                             "contact_z": [2.0, 2.0, 2.0], "feet_x": [1.0, 2.0, 3.0], "feet_y": [0.0, 11.0, 2.0], "method": ["contact: sphere"] * 3})
    labels = pd.DataFrame({"frame": [102, 136, 199], "label_pixel_x": [0.0] * 3, "label_pixel_y": [0.0] * 3, "label_x": [1.0, 2.0, 3.0], "label_y": [1.5, 10.0, 4.0]})
    comparison = contact_comparison(contacts, labels)
    assert comparison[["frame", "label_frame"]].values.tolist() == [[100, 102], [200, 199]]
    assert comparison["contact_error_m"].tolist() == pytest.approx([0.5, 2.0]) and comparison["feet_error_m"].tolist() == pytest.approx([1.5, 2.0])
    line = comparison_lines(comparison, "set 3")[0]
    assert line.startswith("set 3: 2 matched hits") and "contact median 1.25 m" in line and "feet median 1.75 m" in line and "contact closer on 1 of 2" in line
    assert comparison_lines(comparison.iloc[0:0], "set 3") == ["set 3: no matched hits"]
    sample = spot_check_sample(pd.DataFrame({"frame": range(30)}), 10, 1)
    assert sample["frame"].tolist() == sorted(sample["frame"].tolist()) and sample["frame"].is_unique and len(sample) == 10
    assert sample.equals(spot_check_sample(pd.DataFrame({"frame": range(30)}), 10, 1))


def test_range_parsing_and_command_errors(tmp_path):
    assert parse_range(["1", "2", "3"]) == (1, 2, 3)
    with pytest.raises(ValueError):
        parse_range(["1", "2", "4"])
    assert main([str(tmp_path / "missing.mp4"), "--range", "1", "2", "1"]) == 1
    result = subprocess.run([sys.executable, "-m", "vision.contact", str(tmp_path / "missing.mp4")], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "--range" in result.stderr


def pose_caches_are_fresh() -> bool:
    return all(is_newer(contact_paths(VIDEO_PATH, start, end)[0], hit_output_paths(VIDEO_PATH, start, end)[0]) for start, end, _ in TRACKED_RANGES)


real_contacts_required = pytest.mark.skipif(
    not ((PROJECT_ROOT / VIDEO_PATH).exists() and camera_path(VIDEO_PATH).exists() and SHUTTLESET_SET_DIR.is_dir() and pose_caches_are_fresh()),
    reason="needs data/match.mp4, its camera, ShuttleSet and the pose caches of the five ranges (python -m vision.contact)")


@pytest.fixture(scope="module")
def shipped_comparison():
    camera = load_camera(camera_path(VIDEO_PATH))
    return camera, compare_ranges(VIDEO_PATH, TRACKED_RANGES, AXELSEN_GINTING, camera)


def independent_errors(contacts: pd.DataFrame, set_no: int, start: int, end: int) -> pd.DataFrame:
    import json

    homography = np.array(json.loads(Path("data/calibration/match/court.json").read_text())["pixel_to_court_homography"], dtype=float)
    labels = pd.read_csv(SHUTTLESET_SET_DIR / AXELSEN_GINTING / f"set{set_no}.csv").dropna(subset=["frame_num", "hit_x", "hit_y"])
    labels = labels[(labels["frame_num"] >= start) & (labels["frame_num"] <= end)]
    meters = cv2.perspectiveTransform(labels[["hit_x", "hit_y"]].to_numpy(float).reshape(-1, 1, 2), homography).reshape(-1, 2)
    label_points = dict(zip(labels["frame_num"].astype(int), map(tuple, meters)))
    candidates = sorted((abs(int(frame) - label), int(frame), label) for frame in contacts["frame"] for label in label_points if abs(int(frame) - label) <= 5)
    used_frames, used_labels, rows = set(), set(), []
    by_frame = contacts.set_index("frame")
    for _, frame, label in candidates:
        if frame in used_frames or label in used_labels:
            continue
        used_frames.add(frame)
        used_labels.add(label)
        contact, (label_x, label_y) = by_frame.loc[frame], label_points[label]
        rows.append({"set_no": set_no, "contact": np.hypot(contact["hit_x"] - label_x, contact["hit_y"] - label_y), "feet": np.hypot(contact["feet_x"] - label_x, contact["feet_y"] - label_y)})
    return pd.DataFrame(rows)


@real_contacts_required
def test_shipped_contacts_are_reproduced_and_lie_on_the_shuttle_ray(shipped_comparison):
    camera, _ = shipped_comparison
    methods = {}
    for start, end, _ in TRACKED_RANGES:
        shipped = pd.read_csv(contact_paths(VIDEO_PATH, start, end)[1])
        pd.testing.assert_frame_equal(shipped, contacts_for_range(VIDEO_PATH, start, end, camera), check_dtype=False)
        shuttle = pd.read_csv(Path("data/shuttle/match") / f"shuttle_{start}_{end}.csv")
        for contact in shipped.itertuples():
            methods[contact.method] = methods.get(contact.method, 0) + 1
            pixel = project_points(camera, [[contact.hit_x, contact.hit_y, contact.contact_z]])[0]
            assert pixel == pytest.approx(shuttle_pixel_near(shuttle, int(contact.frame)), abs=1e-6)
            assert 0.0 <= contact.contact_z <= 3.5 or contact.method == "contact: closest to sphere"
    assert methods == {"contact: sphere": 74, "contact: closest to sphere": 52}


@real_contacts_required
def test_held_out_gate_recomputed_independently_matches_the_results_doc(shipped_comparison):
    errors = pd.concat([independent_errors(pd.read_csv(contact_paths(VIDEO_PATH, start, end)[1]), set_no, start, end) for start, end, set_no in TRACKED_RANGES], ignore_index=True)
    _, comparison = shipped_comparison
    assert len(errors) == len(comparison) == 110
    assert sorted(errors["contact"].round(9)) == sorted(comparison["contact_error_m"].round(9))
    doc = RESULTS_PATH.read_text()
    for name, part in (("Sets 1-2 (tuning)", errors[errors["set_no"] < 3]), ("**Set 3 (held out)**", errors[errors["set_no"] == 3]), ("All", errors)):
        closer = int((part["contact"] < part["feet"]).sum())
        contact_median = f"**{part['contact'].median():.2f} m**" if name.startswith("**") else f"{part['contact'].median():.2f} m"
        feet_median = f"**{part['feet'].median():.2f} m**" if name.startswith("**") else f"{part['feet'].median():.2f} m"
        assert f"| {name} | {len(part)} | {contact_median} | {part['contact'].mean():.2f} m | {feet_median} | {part['feet'].mean():.2f} m | {closer} of {len(part)} |" in doc
    held_out = errors[errors["set_no"] == 3]
    assert held_out["contact"].median() < held_out["feet"].median() - 0.3 and (held_out["contact"] < held_out["feet"]).sum() >= 30
    quartiles = lambda values: " / ".join(f"{value:.2f}" for value in values.quantile([0.25, 0.5, 0.75]))
    assert f"contact {quartiles(held_out['contact'])} m, feet {quartiles(held_out['feet'])} m" in re.sub(r"\s+", " ", doc)


@real_contacts_required
def test_tuning_grid_on_sets_one_and_two_picks_the_shipped_settings_without_set_three(shipped_comparison):
    camera, _ = shipped_comparison
    doc = RESULTS_PATH.read_text()
    tuning_scores = {}
    for radius in TUNING_RADII:
        cells = []
        for choice in INTERSECTION_CHOICES:
            comparison = compare_ranges(VIDEO_PATH, [rng for rng in TRACKED_RANGES if rng[2] < 3], AXELSEN_GINTING, camera, settings=ContactSettings(reach_radius_m=radius, intersection_choice=choice))
            assert set(comparison["set_no"]) == {1, 2} and len(comparison) == 70
            median, mean = comparison["contact_error_m"].median(), comparison["contact_error_m"].mean()
            tuning_scores[(radius, choice)] = (round(median, 2), round(mean, 2))
            cells.append(f"{median:.2f} / {mean:.2f}")
        row = re.search(rf"^\| {radius} \| (.+) \|$", doc, re.MULTILINE)
        assert row and [cell.replace("**", "").strip() for cell in row.group(1).split(" | ")] == cells
    best = min(tuning_scores, key=lambda setting: tuning_scores[setting])
    assert best == (REACH_RADIUS_M, INTERSECTION_CHOICE)


@real_contacts_required
def test_contact_spot_check_image_regenerates_byte_for_byte(shipped_comparison, tmp_path):
    camera, comparison = shipped_comparison
    sample = spot_check_sample(comparison)
    output = write_contact_spot_check(VIDEO_PATH, camera, sample, tmp_path / "spot_check.jpg")
    assert len(sample) == 10 and filecmp.cmp(output, PROJECT_ROOT / "data" / "checks" / "contact" / "match_contact_spot_check.jpg", shallow=False)
    doc = RESULTS_PATH.read_text()
    for frame in (14347, 14675, 16708, 81544, 16894, 114811, 115743, 17297):
        assert frame in sample["frame"].tolist() and str(frame) in doc


def test_contact_source_has_no_comments_or_docstrings_and_docs_have_no_task_ids():
    source = (PROJECT_ROOT / "vision" / "contact.py").read_text()
    assert '"""' not in source and "'''" not in source and not re.search(r"(?m)^\s*#", source)
    for doc in ("docs/vision/contact.md", "docs/results/contact.md"):
        assert not re.search(TASK_ID_PATTERN, (PROJECT_ROOT / doc).read_text())
    assert "Examples need:" not in (PROJECT_ROOT / "docs" / "vision" / "contact.md").read_text()
