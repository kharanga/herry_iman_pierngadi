# vision/camera.py

A full pinhole camera for one video: focal length, rotation and position, so a pixel
can be turned into a line of sight in court meters (and a 3-D point into a pixel).
The floor calibration (`vision/court.py`) only maps the floor plane; the shuttle and
the racket are in the air, so contact estimation (`vision/contact.py`) needs the camera.

World coordinates are court meters as everywhere else: origin at the near-left corner,
`x` across (0 to 6.1), `y` along (0 to 13.4), and `z` up from the floor.

The camera is fitted to the court calibration's reference points at height 0 (the 4
outer corners, or whichever 4 or more known court points were clicked when a corner is
hidden, `docs/vision/court.md`) and the 2 net-post tops at `x = 0` and `x = 6.1`,
`y = 6.7`, height 1.55 m. The principal point is the image centre, pixels are square and there
is no lens distortion, so 7 numbers are unknown (focal length, 3 for rotation, 3 for
position). For each focal length `cv2.solvePnP` gives the best pose; the focal length
is the one with the lowest root-mean-square reprojection error over the 6 points
(a log-spaced scan from 0.25 to 8 times the frame width, then a golden-section
refinement). The 4 corners alone can't fix the focal length (a floor homography fits
any focal length); the post tops, off the floor, do.

On `data/match.mp4` (`docs/results/contact.md`): focal length 2661 px, camera 21.9 m
behind the near baseline and 8.5 m up, reference errors 0.9 to 3.5 px, court lines
1.0 px median from the painted lines.

## Command

```
.venv/bin/python -m vision.camera <video> [--net-posts X,Y X,Y] [--frame N]
```

| Argument | Meaning |
|---|---|
| `video` | a video with a court calibration (`python -m vision.court`), whose corners are reused |
| `--net-posts` | pixels of the left and right net-post tops (the top of each post, where the net cord is tied); without it a click window opens on the frame (`click_net_post_tops` in `vision/court.py`: click left, then right, `u` undoes, Enter saves, Esc cancels) |
| `--frame` | frame to show and check on (default: the court calibration's frame) |

Writes `data/calibration/<video stem>/camera.json` (focal length, intrinsic matrix,
rotation, translation, camera position, the 6 reference points and their errors, and
the court-line check) and `camera_check.jpg` next to it: all floor lines reprojected
in red, the net top line and both posts in yellow, the clicked post tops circled in
blue. Prints the focal length, the camera position, the 6 errors and the line check.
Exit `1` with `error: ...` for a missing video or court calibration, or a frame that
can't be read; `1` with `cancelled: ...` if the click window is closed; `2` for
invalid arguments.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `NET_POST_HEIGHT_M` | `1.55` | height of the net posts (the net is 1.55 m at the posts) |
| `FOCAL_SEARCH_RANGE_IN_WIDTHS` | `(0.25, 8.0)` | focal lengths searched, in frame widths |
| `FOCAL_COARSE_STEPS`, `FOCAL_REFINE_ITERATIONS` | `400`, `60` | scan and refinement steps |

## Class

### `Camera(intrinsic, rotation, translation, frame_size)`
Frozen dataclass: `intrinsic` 3 x 3, `rotation` 3 x 3 (world to camera), `translation`
(3,), `frame_size` `(width, height)`. Properties: `focal_px`, `centre` (camera
position in court meters), `projection` (3 x 4).

## Functions

The examples use a camera with a known pose (the broadcast one, rounded), the pixels
it sees of the 6 reference points, and fit it back from those pixels:

```python
>>> import numpy as np
>>> from vision.camera import camera_for_focal, camera_reference_meters, fit_camera, project_points
>>> broadcast = camera_for_focal(camera_reference_meters(), [[289.4, 659.6], [987.2, 657.4], [864.2, 301.6], [412.6, 303.0],
...                              [364.5, 308.5], [910.0, 306.5]], 2660.0, (1280, 720))
>>> pixels = project_points(broadcast, camera_reference_meters()).round(1)
>>> pixels.tolist()
[[290.3, 659.8], [986.0, 657.3], [861.8, 301.7], [414.6, 303.3], [361.8, 308.3], [913.5, 306.3]]
>>> camera = fit_camera(camera_reference_meters(), pixels, (1280, 720))
>>> round(camera.focal_px, 1), camera.centre.round(2).tolist()
(2660.0, [3.08, -21.89, 8.45])

```

### `net_post_top_meters() -> ndarray`, `camera_reference_meters(floor_points=None) -> ndarray`
The 2 post tops, and the reference points in fitting order: the floor points at height
0 (default the 4 corners near_left, near_right, far_right, far_left), then left and
right post top.

```python
>>> from vision.camera import net_post_top_meters
>>> net_post_top_meters().tolist()
[[0.0, 6.7, 1.55], [6.1, 6.7, 1.55]]
>>> camera_reference_meters().shape
(6, 3)
>>> camera_reference_meters([[0.0, 0.76], [3.05, 4.72], [6.1, 4.72], [3.05, 0.0], [0.0, 0.0]]).shape
(7, 3)

```

### `intrinsic_matrix(focal_px, frame_size) -> ndarray`

```python
>>> from vision.camera import intrinsic_matrix
>>> intrinsic_matrix(2000.0, (1280, 720)).tolist()
[[2000.0, 0.0, 640.0], [0.0, 2000.0, 360.0], [0.0, 0.0, 1.0]]

```

### `camera_for_focal(world_points, pixel_points, focal_px, frame_size) -> Camera`
The best pose for a given focal length (`cv2.solvePnP`, SQPnP start then iterative
refinement).

### `fit_camera(world_points, pixel_points, frame_size) -> Camera`
Focal length and pose with the lowest reprojection error. Errors: `ValueError` with
fewer than 6 points (here 4 corners and no post tops).

```python
>>> fit_camera(camera_reference_meters()[:4], pixels[:4], (1280, 720))
Traceback (most recent call last):
ValueError: need at least 6 points (4 or more court points and 2 net-post tops) to fit the focal length

```

### `project_points(camera, world_points) -> ndarray`, `reprojection_errors(...)`, `rms_reprojection_error(...)`
3-D points (meters) to pixels; distances to given pixels; their root mean square.

```python
>>> from vision.camera import rms_reprojection_error
>>> project_points(camera, [[3.05, 2.0, 2.5]]).round(1).tolist()
[[637.1, 326.7]]
>>> rms_reprojection_error(camera, camera_reference_meters(), pixels) < 0.1
True

```

### `pixel_ray(camera, pixel) -> (origin, direction)`, `ray_point_at_height(origin, direction, height_m) -> ndarray | None`
The line of sight through a pixel: from the camera position, unit direction in court
meters. Where it reaches a given height (`None` if never, in front of the camera).

```python
>>> from vision.camera import pixel_ray, ray_point_at_height
>>> origin, direction = pixel_ray(camera, pixels[0])
>>> origin.round(2).tolist(), direction.round(3).tolist()
([3.08, -21.89, 8.45], [-0.13, 0.925, -0.357])
>>> (ray_point_at_height(origin, direction, 0.0).round(2) + 0.0).tolist()
[0.0, 0.0, 0.0]

```

### `floor_homography(camera) -> ndarray`, `floor_line_offsets(frame, camera) -> ndarray`
The camera's pixel-to-floor homography, the same kind as the court calibration's;
and the court-line check of `vision/court.py` run with it (pixel offsets from each
reprojected floor line to the painted line).

```python
>>> from vision.camera import floor_homography
>>> from vision.court import pixels_to_court
>>> (pixels_to_court(floor_homography(camera), pixels[:4]).round(2) + 0.0).tolist()
[[0.0, 0.0], [6.1, 0.0], [6.1, 13.4], [0.0, 13.4]]

```

### `draw_camera_check(frame, camera, net_post_pixels) -> ndarray`
Copy of the frame with the floor lines (red), the net top line and the posts
(yellow) reprojected, and the clicked post tops circled (blue).

### `camera_path(video_path)`, `camera_check_path(camera_file)`, `camera_to_dict(camera, floor_pixels, net_post_pixels, frame_index, video_path, floor_names=None, floor_meters=None)`, `camera_from_dict(data)`, `load_camera(camera_file)`, `calibrate_camera(video_path, net_post_pixels=None, frame_index=None)`
`calibrate_camera` takes the floor points from the court calibration
(`vision.court.reference_points`: the clicked named points, or the 4 corners for a
corner calibration).
Files and the whole calibration. `load_camera` raises `FileNotFoundError` naming the
command when the file is missing. `calibrate_camera` returns the saved path, or `None`
if the click window was cancelled.

```python
>>> from pathlib import Path
>>> from vision.camera import camera_check_path, camera_from_dict, camera_path, camera_to_dict
>>> camera_path(Path("data/match.mp4")).as_posix(), camera_check_path(Path("data/calibration/match/camera.json")).as_posix()
('data/calibration/match/camera.json', 'data/calibration/match/camera_check.jpg')
>>> stored = camera_to_dict(camera, pixels[:4], pixels[4:], 12300, Path("data/match.mp4"))
>>> round(camera_from_dict(stored).focal_px, 1), len(stored["reference_errors_px"])
(2660.0, 6)

```
