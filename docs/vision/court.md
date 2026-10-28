# vision/court.py

Court calibration: maps video pixels to court meters for one video. You give the
4 outer court corners on one frame (by clicking, or as numbers), and the module
computes the pixel -> court-meter homography, saves it with the corners as JSON
under `data/calibration/<video_stem>/court.json`, and writes an overlay image with
the reprojected court lines next to it (`court_overlay.jpg`).

Court coordinates follow the SPEC contract: meters, origin at the near-left
corner (the camera stands behind the near baseline), `x` across the court
(0 to 6.1), `y` along the court (0 to 13.4). The corners are the outer corners of
the doubles court (outer edge of the baseline and doubles sideline).

The math (`homography_from_corners`, `pixels_to_court`, `court_to_pixels`,
`court_line_segments`, ...) is pure numpy: arrays in, arrays out, no GUI and no
files. Only `click_corners` opens a window, and only `save_calibration`,
`load_calibration`, `shuttleset_corner_pixels` and `calibrate_video` touch files.

## Command

```
.venv/bin/python -m vision.court <video> [--frame N] [--corners X,Y X,Y X,Y X,Y | --shuttleset-match MATCH] [--output-dir DIR]
```

| Argument | Meaning |
|---|---|
| `video` | path to the match video |
| `--frame` | frame index to calibrate on (default 0); pick a frame from the normal camera view |
| `--corners` | 4 corner pixels `x,y` in order near_left near_right far_right far_left; skips the click window |
| `--shuttleset-match` | take the corners from ShuttleSet `homography.csv` for this match folder; skips the click window. ShuttleSet corner pixels refer to a 1280 x 720 frame, so they are scaled by `video width / 1280` (`label_scale` from `vision.alignment`) to the video's actual size; this assumes the video keeps the 16:9 aspect ratio |
| `--output-dir` | root folder for calibrations (default `data/calibration`) |

Without `--corners` or `--shuttleset-match` an OpenCV window opens on the frame:
click near_left, near_right, far_right, far_left (the prompt names the next one),
`u` undoes the last click, Enter saves once 4 are placed, Esc or the window's close button cancels.

Examples:

```
.venv/bin/python -m vision.court data/my_match.mp4 --frame 300
.venv/bin/python -m vision.court data/match.mp4 --frame 12300 --shuttleset-match Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals
saved data/calibration/match/court.json
saved data/calibration/match/court_overlay.jpg
```

Exit codes: `0` when the calibration was saved; `1` with `error: ...` on stderr
when the video is missing or can't be opened, the frame can't be read, the
ShuttleSet match is not in `homography.csv`, or the corners are degenerate or in
the wrong order; `1` with `cancelled: no calibration saved` when the click window
is closed with Esc; `2` for invalid arguments (a corner not written as `x,y`,
or both `--corners` and `--shuttleset-match`).

## Calibration file

`court.json` holds: `video`, `frame_index`, `frame_width`, `frame_height`,
`corner_order` (`["near_left", "near_right", "far_right", "far_left"]`),
`corner_pixels` (4 x 2), `court_corner_meters` (4 x 2),
`pixel_to_court_homography` (3 x 3, maps pixel `(x, y, 1)` to court meters) and
`max_corner_reprojection_error_px`. `corner_pixels` are always in the video's own
pixels (ShuttleSet corners are saved after scaling).

Limitation: the folder is keyed by the video file name without its extension only,
so two different videos with the same name (for example `a/match.mp4` and
`b/match.mp4`) share `data/calibration/match/court.json`, and the second
calibration overwrites the first. Give recordings unique file names.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `COURT_WIDTH_M` | `6.1` | doubles width (outer edges of the doubles sidelines) |
| `COURT_LENGTH_M` | `13.4` | length (outer edges of the baselines) |
| `NET_Y_M` | `6.7` | net position along the court |
| `SINGLES_SIDELINE_INSET_M` | `0.46` | singles sideline distance from the doubles sideline |
| `SHORT_SERVICE_LINE_FROM_NET_M` | `1.98` | short service line distance from the net |
| `DOUBLES_LONG_SERVICE_LINE_FROM_BASELINE_M` | `0.76` | doubles long service line distance from the baseline |
| `CORNER_NAMES` | `("near_left", "near_right", "far_right", "far_left")` | order of every corner array |
| `CALIBRATION_DIR` | `Path("data/calibration")` | default output root |
| `SHUTTLESET_HOMOGRAPHY_CSV` | `Path("data/CoachAI-Projects/ShuttleSet/set/homography.csv")` | ShuttleSet per-video corners |
| `LINE_COLOR_BGR`, `CORNER_COLOR_BGR` | `(0, 0, 255)`, `(0, 255, 255)` | red lines, yellow corner markers |

```python
>>> from vision.court import COURT_LENGTH_M, COURT_WIDTH_M, CORNER_NAMES, NET_Y_M
>>> COURT_WIDTH_M, COURT_LENGTH_M, NET_Y_M
(6.1, 13.4, 6.7)
>>> CORNER_NAMES
('near_left', 'near_right', 'far_right', 'far_left')

```

## Functions

The examples use a synthetic trapezoid seen from behind the near baseline: near
corners at the bottom of a 500 x 400 image, far corners higher and closer together.

```python
>>> import numpy as np
>>> corner_pixels = [[100, 300], [400, 300], [350, 100], [150, 100]]

```

### `court_corner_meters() -> np.ndarray`
The 4 court corners in meters, in `CORNER_NAMES` order (4 x 2 float array). No inputs, no errors.

```python
>>> from vision.court import court_corner_meters
>>> court_corner_meters()
array([[ 0. ,  0. ],
       [ 6.1,  0. ],
       [ 6.1, 13.4],
       [ 0. , 13.4]])

```

### `as_corner_array(corner_pixels) -> np.ndarray`
Converts `corner_pixels` (anything array-like with 4 `(x, y)` pairs) to a 4 x 2 float array.
Errors: `ValueError` if the shape is not 4 x 2 or a value is NaN or infinite.

```python
>>> from vision.court import as_corner_array
>>> as_corner_array([[1, 2], [3, 4], [5, 6], [7, 8]]).dtype
dtype('float64')
>>> as_corner_array([[1, 2], [3, 4], [5, 6]])
Traceback (most recent call last):
ValueError: expected 4 corners as (x, y) pixels, got shape (3, 2)

```

### `homography_between(source_points, target_points) -> np.ndarray`
Exact 3 x 3 homography mapping 4 `source_points` onto 4 `target_points` (both 4 x 2
float arrays), with the bottom-right entry fixed to 1. Solves the 8 x 8 linear
system directly (no OpenCV). Errors: `numpy.linalg.LinAlgError` if the system is singular.

```python
>>> from vision.court import homography_between
>>> unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
>>> np.round(homography_between(unit_square, unit_square * 2 + 5), 6)
array([[2., 0., 5.],
       [0., 2., 5.],
       [0., 0., 1.]])

```

### `turn_directions(corners) -> np.ndarray`
Cross product of each pair of consecutive edges of the quadrilateral `corners`
(4 x 2 array); all the same sign means convex. Returns 4 floats.

```python
>>> from vision.court import turn_directions
>>> turn_directions(np.array(corner_pixels, dtype=float))
array([-60000., -40000., -40000., -60000.])

```

### `check_corners_form_convex_quadrilateral(corners) -> None`
Raises `ValueError` if three corners lie on one line (or two coincide), or if the
quadrilateral is not convex (corners clicked out of order, so the edges cross).

```python
>>> from vision.court import check_corners_form_convex_quadrilateral
>>> check_corners_form_convex_quadrilateral(np.array([[100, 300], [350, 100], [400, 300], [150, 100]], dtype=float))
Traceback (most recent call last):
ValueError: corners do not form a convex quadrilateral; give them in order near_left, near_right, far_right, far_left
>>> check_corners_form_convex_quadrilateral(np.array([[0, 0], [10, 10], [20, 20], [0, 30]], dtype=float))
Traceback (most recent call last):
ValueError: corners are degenerate: three of them lie on one line or two are the same point

```

### `check_corner_order_seen_from_behind_near_baseline(corners) -> None`
Raises `ValueError` unless the near corners are lower in the image than the far
corners and each left corner is left of its right corner. This catches a convex
but rotated or mirrored order (for example starting at a far corner).

```python
>>> from vision.court import check_corner_order_seen_from_behind_near_baseline
>>> check_corner_order_seen_from_behind_near_baseline(np.array([[150, 100], [350, 100], [400, 300], [100, 300]], dtype=float))
Traceback (most recent call last):
ValueError: corner order looks wrong for a camera behind the near baseline; give them in order near_left, near_right, far_right, far_left

```

### `homography_from_corners(corner_pixels) -> np.ndarray`
The pixel -> court-meter homography (3 x 3) from the 4 corner pixels in
`CORNER_NAMES` order. Errors: `ValueError` from `as_corner_array`,
`check_corners_form_convex_quadrilateral` or `check_corner_order_seen_from_behind_near_baseline`.

```python
>>> from vision.court import homography_from_corners, pixels_to_court
>>> pixel_to_court = homography_from_corners(corner_pixels)
>>> np.round(pixels_to_court(pixel_to_court, corner_pixels), 6)
array([[ 0. ,  0. ],
       [ 6.1,  0. ],
       [ 6.1, 13.4],
       [ 0. , 13.4]])

```

### `apply_homography(homography, points) -> np.ndarray`
Applies a 3 x 3 `homography` to `points` (N x 2, or a single `(x, y)`), dividing
by the third coordinate. Returns an N x 2 float array.

```python
>>> from vision.court import apply_homography
>>> apply_homography(np.diag([2.0, 3.0, 1.0]), (1, 1))
array([[2., 3.]])

```

### `pixels_to_court(homography, pixel_points) -> np.ndarray`
Pixel points (N x 2) to court meters (N x 2) with a pixel -> court homography.
The middle of the near baseline maps to `x = 3.05, y = 0`; perspective makes the
image midpoint between near and far baselines land beyond the net (5.36 m, not 6.7 m),
because the far half is squeezed into fewer pixels.

```python
>>> np.round(pixels_to_court(pixel_to_court, [[250, 300], [250, 200]]), 3)
array([[3.05, 0.  ],
       [3.05, 5.36]])

```

### `court_to_pixels(homography, court_points) -> np.ndarray`
The inverse: court meters (N x 2) to pixels (N x 2), using the same pixel -> court homography.

```python
>>> from vision.court import court_to_pixels
>>> np.round(court_to_pixels(pixel_to_court, [[3.05, 6.7], [0, 0]]), 1)
array([[250., 180.],
       [100., 300.]])

```

### `reprojection_errors(homography, pixel_points, court_points) -> np.ndarray`
For each pair, the distance in pixels between `pixel_points[i]` and
`court_points[i]` projected into the image. Returns N floats.

```python
>>> from vision.court import reprojection_errors
>>> reprojection_errors(pixel_to_court, [[100, 302], [400, 300]], [[0, 0], [6.1, 0]]).round(3)
array([2., 0.])

```

### `court_line_segments() -> dict[str, tuple[tuple[float, float], tuple[float, float]]]`
Every painted line of an official badminton court, plus the net, as named
segments `(start, end)` in court meters: both baselines, doubles and singles
sidelines, doubles long service lines, short service lines, the two centre lines
(service line to baseline) and the net. 13 segments.

```python
>>> from vision.court import court_line_segments
>>> segments = court_line_segments()
>>> len(segments)
13
>>> np.round(segments["near_short_service_line"], 2).tolist()
[[0.0, 4.72], [6.1, 4.72]]
>>> segments["left_singles_sideline"], segments["far_centre_line"]
(((0.46, 0.0), (0.46, 13.4)), ((3.05, 8.68), (3.05, 13.4)))

```

### `points_along_segment(start, end, point_count) -> np.ndarray`
`point_count` evenly spaced points from `start` to `end` inclusive (point_count x 2).

```python
>>> from vision.court import points_along_segment
>>> points_along_segment((0, 0), (6.1, 0), 3)
array([[0.  , 0.  ],
       [3.05, 0.  ],
       [6.1 , 0.  ]])

```

### `draw_court_lines(frame, homography, color=LINE_COLOR_BGR, thickness=1) -> np.ndarray`
A copy of `frame` (H x W x 3 `uint8`) with every segment of `court_line_segments`
projected into the image and drawn. The input frame is not changed.

```python
>>> from vision.court import draw_court_lines
>>> blank_frame = np.zeros((400, 500, 3), dtype=np.uint8)
>>> drawn_frame = draw_court_lines(blank_frame, pixel_to_court)
>>> drawn_frame[300, 250].tolist(), int(blank_frame.max())
([0, 0, 245], 0)

```

### `draw_corner_markers(frame, corner_pixels, color=CORNER_COLOR_BGR) -> np.ndarray`
A copy of `frame` with a circle and the corner name at each corner. Errors: as `as_corner_array`.

```python
>>> from vision.court import draw_corner_markers
>>> int(draw_corner_markers(blank_frame, corner_pixels)[300, 106].max()) > 0
True

```

### `brightness_profile_peak_offset(gray_frame, point, normal, search_radius) -> float | None`
Samples `gray_frame` (H x W `uint8`) along `normal` (unit 2-vector) from
`point - search_radius * normal` to `point + search_radius * normal` and returns
the offset (pixels along `normal`) of the brightest sample, averaged if tied.
Returns `None` if the profile leaves the image or the peak is less than 40 grey
levels above the profile median (no line there).

```python
>>> from vision.court import brightness_profile_peak_offset
>>> gray_frame = np.zeros((50, 50), dtype=np.uint8)
>>> gray_frame[23, :] = 255
>>> brightness_profile_peak_offset(gray_frame, np.array([25.0, 25.0]), np.array([0.0, 1.0]), 5)
-2.0
>>> brightness_profile_peak_offset(np.zeros((50, 50), dtype=np.uint8), np.array([25.0, 25.0]), np.array([0.0, 1.0]), 5) is None
True

```

### `line_offsets_to_bright_pixels(frame, homography, start_meters, end_meters, sample_count=40, search_radius=8) -> np.ndarray`
Measures how far a projected court line sits from the painted (bright) line.
Projects the segment into `frame` (BGR or grey), takes `sample_count` points
between 5% and 95% along it, and returns `brightness_profile_peak_offset` for
each point where a line was found. The normal is the image direction `(dx, dy)`
of the segment rotated to `(-dy, dx)`; for a left-to-right line, positive means
the painted line is lower in the image. Samples hidden by players or off-image are dropped.

```python
>>> from vision.court import line_offsets_to_bright_pixels
>>> painted_frame = np.zeros((400, 500), dtype=np.uint8)
>>> painted_frame[298:301, :] = 255
>>> line_offsets_to_bright_pixels(painted_frame, pixel_to_court, (0, 0), (6.1, 0), sample_count=5)
array([-1., -1., -1., -1., -1.])

```

### `calibration_path(video_path, calibration_dir=CALIBRATION_DIR) -> Path`
Where the calibration for `video_path` is stored: `<calibration_dir>/<video stem>/court.json`.

```python
>>> from pathlib import Path
>>> from vision.court import calibration_path, overlay_path_for
>>> calibration_path(Path("data/match.mp4")).as_posix()
'data/calibration/match/court.json'

```

### `overlay_path_for(calibration_file) -> Path`
The overlay image path next to a calibration file (`court_overlay.jpg`).

```python
>>> overlay_path_for(Path("data/calibration/match/court.json")).as_posix()
'data/calibration/match/court_overlay.jpg'

```

### `save_calibration(output_path, video_path, frame_index, frame_size, corner_pixels) -> dict`
Computes the homography from `corner_pixels` and writes the calibration JSON to
`output_path` (folders created as needed). `frame_size` is `(width, height)`.
Returns the saved dictionary (lists, not arrays). Errors: as `homography_from_corners`.

### `load_calibration(calibration_file) -> dict`
Reads a calibration JSON; `pixel_to_court_homography` and `corner_pixels` come
back as numpy arrays. Errors: `FileNotFoundError` naming the path if it is missing.

```python
>>> import tempfile
>>> from vision.court import load_calibration, save_calibration
>>> temporary_folder = tempfile.TemporaryDirectory()
>>> calibration_file = Path(temporary_folder.name) / "my_match" / "court.json"
>>> saved = save_calibration(calibration_file, Path("my_match.mp4"), 300, (500, 400), corner_pixels)
>>> saved["frame_width"], saved["frame_height"], saved["max_corner_reprojection_error_px"] < 1e-6
(500, 400, True)
>>> loaded = load_calibration(calibration_file)
>>> np.allclose(loaded["pixel_to_court_homography"], pixel_to_court)
True
>>> load_calibration(Path(temporary_folder.name) / "missing.json")  # doctest: +ELLIPSIS
Traceback (most recent call last):
FileNotFoundError: calibration not found: .../missing.json (run python -m vision.court first)

```

### `shuttleset_corner_pixels(match_folder, homography_csv=SHUTTLESET_HOMOGRAPHY_CSV) -> np.ndarray`
Reads a match's court corners from ShuttleSet `homography.csv` and returns them
in `CORNER_NAMES` order (ShuttleSet's `downleft`, `downright`, `upright`, `upleft`).
The values are unscaled pixels of ShuttleSet's 1280 x 720 label frame; scale them
for other video sizes (`calibrate_video(..., corners_in_shuttleset_label_frame=True)` does).
Errors: `FileNotFoundError` if the CSV is missing, `ValueError` if the match isn't in it.

```python
>>> from vision.court import shuttleset_corner_pixels
>>> homography_csv = Path(temporary_folder.name) / "homography.csv"
>>> _ = homography_csv.write_text(
...     "id,video,homography_matrix,upleft_x,upright_x,downleft_x,downright_x,upleft_y,upright_y,downleft_y,downright_y\n"
...     '1,Some_Match,"[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",150,350,100,400,100,100,300,300\n'
... )
>>> shuttleset_corner_pixels("Some_Match", homography_csv)
array([[100., 300.],
       [400., 300.],
       [350., 100.],
       [150., 100.]])
>>> shuttleset_corner_pixels("Other_Match", homography_csv)  # doctest: +ELLIPSIS
Traceback (most recent call last):
ValueError: match not in .../homography.csv: Other_Match

```

### `parse_corner_argument(corner_text) -> tuple[float, float]`
Parses one `--corners` value `"x,y"`. Errors: `argparse.ArgumentTypeError` otherwise.

```python
>>> from vision.court import parse_corner_argument
>>> parse_corner_argument("289.4,659.6")
(289.4, 659.6)

```

### `window_was_closed(window_name) -> bool`
`True` if the OpenCV window `window_name` is no longer visible (closed with its
close button, or never opened), else `False`. Used by `click_corners`. No errors.

```python
>>> from vision.court import window_was_closed
>>> window_was_closed("a window that was never opened")
True

```

### `click_corners(frame, window_name="court calibration") -> np.ndarray | None`
Opens an OpenCV window on `frame` and collects 4 clicks (4 x 2 float array, in
`CORNER_NAMES` order), or returns `None` on Esc. Keys: `u` undo, Enter save; the window's close button acts like Esc.
Needs a display, so it has no doctest; use `--corners` or `calibrate_video(..., corner_pixels=...)` headless.

### `calibrate_video(video_path, frame_index=0, corner_pixels=None, calibration_dir=CALIBRATION_DIR, corners_in_shuttleset_label_frame=False) -> Path | None`
Reads frame `frame_index`, gets the corners (`corner_pixels` if given, otherwise
`click_corners`; with `corners_in_shuttleset_label_frame=True` the given corners
are taken as 1280 x 720 ShuttleSet pixels and multiplied by `label_scale(video width)`), saves the calibration to `calibration_path(video_path, calibration_dir)`
and the overlay (court lines plus corner markers) next to it. Returns the
calibration path, or `None` if the click window was cancelled.
Errors: `FileNotFoundError`/`OSError` from `open_video`, `ValueError` if the frame
can't be read or the corners are invalid.

```python
>>> import cv2
>>> from vision.court import calibrate_video
>>> video_path = Path(temporary_folder.name) / "tiny.avi"
>>> writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (500, 400))
>>> for frame_index in range(3):
...     _ = writer.write(np.zeros((400, 500, 3), dtype=np.uint8))
>>> writer.release()
>>> saved_path = calibrate_video(video_path, 1, corner_pixels, Path(temporary_folder.name) / "calibration")
>>> saved_path.relative_to(temporary_folder.name).as_posix(), overlay_path_for(saved_path).is_file()
('calibration/tiny/court.json', True)
>>> shuttleset_style_corners = [[200, 600], [800, 600], [700, 200], [300, 200]]
>>> scaled_path = calibrate_video(video_path, 1, shuttleset_style_corners, Path(temporary_folder.name) / "scaled", corners_in_shuttleset_label_frame=True)
>>> load_calibration(scaled_path)["corner_pixels"].tolist()
[[78.125, 234.375], [312.5, 234.375], [273.4375, 78.125], [117.1875, 78.125]]
>>> calibrate_video(video_path, 10, corner_pixels, Path(temporary_folder.name) / "calibration")  # doctest: +ELLIPSIS
Traceback (most recent call last):
ValueError: could not read frame 10 of .../tiny.avi

```

### `main(arguments) -> int`
Command entry point: parses `arguments` (`list[str]`, without the program name),
runs `calibrate_video`, prints the saved paths, and returns the exit code above.
Errors: invalid arguments raise `SystemExit(2)` from argparse.

```python
>>> import contextlib
>>> import io
>>> from vision.court import main
>>> error_output = io.StringIO()
>>> with contextlib.redirect_stderr(error_output):
...     main([str(video_path), "--frame", "1", "--corners", "100,300", "350,100", "400,300", "150,100", "--output-dir", temporary_folder.name])
1
>>> error_output.getvalue()
'error: corners do not form a convex quadrilateral; give them in order near_left, near_right, far_right, far_left\n'
>>> main([str(video_path), "--frame", "1", "--corners", "100,300", "400,300", "350,100", "150,100", "--output-dir", temporary_folder.name])  # doctest: +ELLIPSIS
saved .../tiny/court.json
saved .../tiny/court_overlay.jpg
0
>>> with contextlib.redirect_stderr(io.StringIO()):
...     main([str(video_path), "--corners", "1;2", "3,4", "5,6", "7,8"])
Traceback (most recent call last):
SystemExit: 2
>>> temporary_folder.cleanup()

```
