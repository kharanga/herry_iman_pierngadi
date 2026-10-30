# vision/contact.py

Estimates where the racket met the shuttle at each detected hit, and its floor point,
which becomes the shot's `hit_x/hit_y` (`vision/features.py`). ShuttleSet's hit point
is the floor point under the racket contact, not the hitter's feet; on a reaching shot
they are about 1 m apart (`docs/results/features.md`).

One camera can't measure depth, so the contact is where two constraints meet:
1. **The shuttle is on the line of sight through its pixel** at the hit frame
   (`vision/camera.py`, `pixel_ray`). The shuttle pixel comes from the shuttle track
   (`vision/shuttle.py`): the visible position at the hit frame, or the nearest one
   within 2 frames; otherwise there is no ray and the hit falls back to the feet.
2. **The racket head is within reach of the hitting shoulder.** YOLO11 pose runs on the
   hit frames only; the pose whose box overlaps the hitter's tracked box most (IoU at
   least 0.3) gives the hitting-side shoulder pixel (right shoulder, left for players
   given as left-handed), if its keypoint confidence is at least 0.3; otherwise the hit
   falls back to the feet. The shoulder's 3-D position is the point on the shoulder
   pixel's line of sight closest to the vertical line above the hitter's tracked feet
   (the feet are on the floor, so the floor calibration places them).

The contact is where the shuttle's line of sight meets a sphere of radius 1.0 m
(arm plus racket, tuned below) around the shoulder. Of the up to two intersections,
those 0 to 3.5 m above the floor are kept and the one farther from the camera is
taken. If the ray misses the sphere (the shuttle seen further than the reach from the
shoulder, often a pose or timing error), the point on the ray closest to the shoulder
is used. The contact's `(x, y)` is the hit point; its height is kept as `contact_z`.

Tuning (sets 1-2 of the hit-detection ranges, set 3 held out; `docs/results/contact.md`):
radius 0.6 to 1.8 m and three rules for picking an intersection (toward the net,
nearer the camera, farther from the camera). Radius 0.8 and 1.0 m with "farther from
the camera" tied on the median; 1.0 m has the lower mean.

The geometry (`ray_sphere_distances`, `closest_point_to_vertical_line`,
`contact_point`, `shoulder_position`, `box_overlap`, `hitter_pose`,
`hitting_shoulder_pixel`, `shuttle_pixel_near`, `estimate_hit_point`, the comparison
functions) is pure; `detect_poses`, `poses_for_range`, `contacts_for_range`,
`compare_ranges`, `write_contact_spot_check` and `main` read files or run the model.

## Command

```
.venv/bin/python -m vision.contact <video> --range START END SET [--range ...] [--left-handed me,opp] [--model path] [--compare SHUTTLESET_MATCH]
```

Needs the camera (`python -m vision.camera`), and for each range the player, shuttle
and hit files (`python -m vision.players`, `vision.shuttle`, `vision.hits` with the same
`--start`/`--end`), and the pose weights `data/models/yolo11s-pose.pt` (download:
`cd data/models && ../../.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11s-pose.pt')"`).

Writes per range `data/contact/<video stem>/poses_<start>_<end>.csv` (the pose model's
shoulders on each hit frame, reused while it is newer than the hits file; about 1 s
per hit on the CPU) and `contacts_<start>_<end>.csv` (`CONTACT_COLUMNS`: `frame`,
`hitter`, `hit_x`, `hit_y` (the contact floor point, or the feet on a fallback),
`contact_z`, `feet_x`, `feet_y`, `method`). `method` is `contact: sphere`,
`contact: closest to sphere`, `feet: no shuttle pixel`, `feet: no pose` or `no feet`;
the counts are printed per range.

With `--compare`: matches the hits to ShuttleSet's labelled hit points (±5 frames),
prints the median and mean distance of the contact estimate and of the feet to the
labelled hit point for sets 1-2, set 3 and all, and writes
`data/checks/contact/<video stem>_contact_spot_check.jpg`: 10 random matched hits
(numpy seed 1) with the shuttle pixel (yellow), the contact floor point (green, with a
line up to the contact in the air), the feet (blue) and the labelled hit point (red).

Exit `1` with `error: ...` for a missing video, camera, track file or pose weights, or a
set number outside 1-3; `2` for invalid arguments.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `REACH_RADIUS_M` | `1.0` | arm plus racket, shoulder to contact |
| `INTERSECTION_CHOICE`, `INTERSECTION_CHOICES` | `"farther_from_camera"`, `("toward_net", "nearer_camera", "farther_from_camera")` | which intersection to take |
| `CONTACT_HEIGHT_RANGE_M` | `(0.0, 3.5)` | plausible contact heights |
| `SHUTTLE_MAX_FRAME_GAP`, `FEET_MAX_FRAME_GAP` | `2`, `3` | how far from the hit frame a shuttle pixel or player position may be taken |
| `SHOULDER_MIN_CONFIDENCE`, `POSE_MIN_BOX_OVERLAP` | `0.3`, `0.3` | pose keypoint and box-match thresholds |

The two frame gaps are for 30 fps: `contact_settings_for_video(fps)` scales them with
`vision.hits.frames_for_fps` (at 60 fps: 4 and 6 frames), and `contacts_for_range` uses
the video's frame rate unless settings are given.

```python
>>> from vision.contact import ContactSettings, contact_settings_for_video
>>> contact_settings_for_video(30.0) == ContactSettings()
True
>>> scaled = contact_settings_for_video(60.0)
>>> scaled.shuttle_max_frame_gap, scaled.feet_max_frame_gap
(4, 6)

```
| `POSE_MODEL_PATH` | `Path("data/models/yolo11s-pose.pt")` | default pose weights |

## Geometry

### `ray_point(origin, direction, distance)`, `closest_ray_distance(origin, direction, point) -> float`, `ray_sphere_distances(origin, direction, centre, radius) -> list[float]`
Points along a ray; the distance along it to the point closest to `point` (never
negative); the distances where it enters and leaves a sphere (empty if it misses,
only positive ones).

```python
>>> import numpy as np
>>> from vision.contact import closest_ray_distance, ray_point, ray_sphere_distances
>>> along_x = (np.zeros(3), np.array([1.0, 0.0, 0.0]))
>>> ray_sphere_distances(*along_x, np.array([5.0, 0.0, 0.0]), 1.0), ray_sphere_distances(*along_x, np.array([5.0, 2.0, 0.0]), 1.0)
([4.0, 6.0], [])
>>> closest_ray_distance(*along_x, np.array([5.0, 2.0, 0.0])), ray_point(np.zeros(3), np.array([0.0, 0.0, 1.0]), 2.5).tolist()
(5.0, [0.0, 0.0, 2.5])

```

### `closest_point_to_vertical_line(origin, direction, floor_point) -> ndarray`
The point on the ray closest to the vertical line through `floor_point` (meters).

```python
>>> from vision.contact import closest_point_to_vertical_line
>>> down_the_court = np.array([0.0, 0.9, -0.3]) / np.linalg.norm([0.0, 0.9, -0.3])
>>> closest_point_to_vertical_line(np.array([3.0, -20.0, 8.0]), down_the_court, (3.0, 2.0)).round(3).tolist()
[3.0, 2.0, 0.667]

```

### `chosen_intersection(points, camera_centre, choice)`, `contact_point(origin, direction, shoulder, settings=ContactSettings()) -> (point, how)`
The contact on a shuttle ray for a given shoulder position. The example uses the
fitted broadcast camera from `docs/vision/camera.md` and a shuttle 0.84 m from the
shoulder:

```python
>>> from vision.camera import camera_for_focal, camera_reference_meters, fit_camera, pixel_ray, project_points
>>> from vision.contact import ContactSettings, contact_point
>>> broadcast = camera_for_focal(camera_reference_meters(), [[289.4, 659.6], [987.2, 657.4], [864.2, 301.6], [412.6, 303.0],
...                              [364.5, 308.5], [910.0, 306.5]], 2660.0, (1280, 720))
>>> camera = fit_camera(camera_reference_meters(), project_points(broadcast, camera_reference_meters()).round(1), (1280, 720))
>>> shoulder = np.array([3.0, 2.0, 1.45])
>>> origin, direction = pixel_ray(camera, project_points(camera, [[3.4, 2.5, 2.0]])[0].round(1))
>>> [round(distance, 2) for distance in ray_sphere_distances(origin, direction, shoulder, 1.0)]
[24.25, 25.53]
>>> point, how = contact_point(origin, direction, shoulder)
>>> point.round(2).tolist(), how
([3.4, 2.78, 1.93], 'sphere')
>>> contact_point(origin, direction, shoulder, ContactSettings(intersection_choice="nearer_camera"))[0].round(2).tolist()
[3.39, 1.55, 2.25]
>>> point, how = contact_point(origin, direction, np.array([0.5, 2.0, 1.45]))
>>> point.round(2).tolist(), how
([3.4, 2.13, 2.1], 'closest to sphere')

```

The true shuttle (3.4, 2.5, 2.0) lies between the two intersections: the depth along
the line of sight is what one camera can't see, which is why the radius and the rule
were tuned against labels.

### `shoulder_position(camera, shoulder_pixel, feet_point) -> ndarray`

```python
>>> from vision.contact import shoulder_position
>>> shoulder_position(camera, project_points(camera, [[3.0, 2.0, 1.45]])[0], (3.0, 2.0)).round(2).tolist()
[3.0, 2.0, 1.45]

```

## Pose and track lookups

### `box_overlap(first_box, second_box) -> float`, `hitter_pose(poses_at_frame, hitter_box, min_overlap=0.3)`, `hitting_shoulder_pixel(pose, left_handed, min_confidence)`
Intersection over union of two `left, top, right, bottom` boxes; the pose row matching
the hitter's box (or `None`); its hitting-side shoulder pixel (or `None` below the
confidence).

```python
>>> import pandas as pd
>>> from vision.contact import box_overlap, hitter_pose, hitting_shoulder_pixel
>>> box_overlap((0, 0, 10, 10), (5, 0, 15, 10)), box_overlap((0, 0, 10, 10), (20, 20, 30, 30))
(0.3333333333333333, 0.0)
>>> poses = pd.DataFrame([
...     {"frame": 100, "left": 0, "top": 0, "right": 10, "bottom": 20, "left_shoulder_x": 3, "left_shoulder_y": 5, "left_shoulder_conf": 0.9,
...      "right_shoulder_x": 7, "right_shoulder_y": 5, "right_shoulder_conf": 0.2},
...     {"frame": 100, "left": 50, "top": 0, "right": 60, "bottom": 20, "left_shoulder_x": 53, "left_shoulder_y": 4, "left_shoulder_conf": 0.9,
...      "right_shoulder_x": 57, "right_shoulder_y": 4, "right_shoulder_conf": 0.95}])
>>> pose = hitter_pose(poses, (49, 1, 61, 21))
>>> int(pose["left"]), hitting_shoulder_pixel(pose, False, 0.3), hitter_pose(poses, (200, 0, 210, 10)) is None
(50, (57.0, 4.0), True)
>>> hitting_shoulder_pixel(poses.iloc[0], False, 0.3) is None, hitting_shoulder_pixel(poses.iloc[0], True, 0.3)
(True, (3.0, 5.0))

```

### `nearest_row_within(table, frame, max_gap)`, `shuttle_pixel_near(shuttle, frame, max_gap=2)`, `player_row_near(table, player, frame, max_gap=3)`

```python
>>> from vision.contact import shuttle_pixel_near
>>> shuttle = pd.DataFrame({"frame": [98, 99, 100, 101], "x": [600, 610, 0, 630], "y": [300, 305, 0, 315], "visibility": [1, 1, 0, 1], "inpainted": 0})
>>> shuttle_pixel_near(shuttle, 100), shuttle_pixel_near(shuttle, 104)
((610.0, 305.0), None)

```

### `estimate_hit_point(camera, frame, hitter, shuttle, positions, boxes, poses, left_handed=frozenset(), settings=ContactSettings()) -> dict`, `estimate_hit_points(...) -> DataFrame`, `method_counts(contacts) -> dict`
One hit's `CONTACT_COLUMNS` row (with the fallback reason in `method`), all hits of a
range, and how many hits took each method.

## Files and model

### `load_pose_model(model_path)`, `pose_rows(frame_index, result)`, `detect_poses(video_path, frames, model_path)`, `contact_paths(...)`, `is_newer(path, than)`, `poses_for_range(...)`, `range_inputs(...)`, `contacts_for_range(...)`
`load_pose_model` raises `FileNotFoundError` with the download command and restores
numpy's print options after importing Ultralytics (as in `vision/players.py`).

```python
>>> from pathlib import Path
>>> from vision.contact import contact_paths, load_pose_model
>>> [path.as_posix() for path in contact_paths(Path("data/match.mp4"), 14201, 14947)]
['data/contact/match/poses_14201_14947.csv', 'data/contact/match/contacts_14201_14947.csv']
>>> load_pose_model(Path("data/models/missing-pose.pt"))
Traceback (most recent call last):
FileNotFoundError: pose weights not found: data/models/missing-pose.pt (download with: cd data/models && python -c "from ultralytics import YOLO; YOLO('missing-pose.pt')")

```

## Comparison with ShuttleSet

### `labelled_hit_points(shots, start_frame, end_frame, homography, scale=1.0)`, `contact_comparison(contacts, labels, tolerance=5)`, `comparison_lines(comparison, name)`, `compare_ranges(...)`, `spot_check_sample(...)`, `contact_tile(...)`, `write_contact_spot_check(...)`
ShuttleSet's hit pixels in a range mapped to court meters; per matched hit the
distance of the contact estimate and of the feet to the labelled point; the printed
summary; all ranges together; the sampled hits and their image.

```python
>>> from vision.contact import comparison_lines, contact_comparison
>>> contacts = pd.DataFrame({"frame": [100, 130], "hitter": ["me", "opp"], "hit_x": [3.0, 2.0], "hit_y": [3.0, 10.0], "contact_z": [1.8, 2.1],
...                          "feet_x": [3.0, 2.0], "feet_y": [2.0, 11.0], "method": "contact: sphere"})
>>> labels = pd.DataFrame({"frame": [101, 129], "label_pixel_x": [0.0, 0.0], "label_pixel_y": [0.0, 0.0], "label_x": [3.0, 2.0], "label_y": [3.2, 10.0]})
>>> contact_comparison(contacts, labels)[["frame", "label_frame", "contact_error_m", "feet_error_m"]].round(2)
   frame  label_frame  contact_error_m  feet_error_m
0    100          101              0.2           1.2
1    130          129              0.0           1.0
>>> print(comparison_lines(contact_comparison(contacts, labels), "example")[0])
example: 2 matched hits, distance to the labelled hit point: contact median 0.10 m (mean 0.10), feet median 1.10 m (mean 1.10); contact closer on 2 of 2

```

### `parse_range(text)`, `main(arguments)`
