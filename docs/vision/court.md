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
files. Only `click_corners` and `click_net_post_tops` (through `click_named_points`) open a window, and only `save_calibration`,
`load_calibration`, `shuttleset_corner_pixels` and `calibrate_video` touch files.

## Command

```
.venv/bin/python -m vision.court <video> [--frame N] [--corners X,Y X,Y X,Y X,Y | --shuttleset-match MATCH | --points NAME=X,Y ... | --pick-points | --list-points] [--output-dir DIR]
```

| Argument | Meaning |
|---|---|
| `video` | path to the match video |
| `--frame` | frame index to calibrate on (default 0); pick a frame from the normal camera view |
| `--corners` | 4 corner pixels `x,y` in order near_left near_right far_right far_left; skips the click window |
| `--shuttleset-match` | take the corners from ShuttleSet `homography.csv` for this match folder; skips the click window. ShuttleSet corner pixels refer to a 1280 x 720 frame, so they are scaled by `video width / 1280` (`label_scale` from `vision.alignment`) to the video's actual size; this assumes the video keeps the 16:9 aspect ratio |
| `--points` | 4 or more known court points by name with their pixels, `NAME=X,Y` (quote names with spaces), for when a corner is hidden or blocked; skips the click window. No 4 of them may have 3 on one line (they must fix a homography) |
| `--pick-points` | opens a click window where you choose which known court point you click (`click_court_points`) |
| `--list-points` | prints the names and court positions of the known points and exits |
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

Calibrating with named points (a near corner below the frame):

```
.venv/bin/python -m vision.court data/recordings/stand_in/set1.mp4 --frame 600 --points \
  "near_left=217,762" "near_long_service_line x left_doubles_sideline=383,736" \
  "near_long_service_line x centre_line=574,889" "near_baseline x centre_line=356,960" \
  "near_short_service_line x centre_line=1140,718" "near_short_service_line x right_singles_sideline=1584,810" \
  "near_short_service_line x right_doubles_sideline=1714,837"
```

**Known court points.** The 30 crossings of the 6 lines across the court
(`near_baseline`, `near_long_service_line` (doubles), `near_short_service_line`,
`far_short_service_line`, `far_long_service_line`, `far_baseline`) with the 5 lines
along it (`left_doubles_sideline`, `left_singles_sideline`, `centre_line`,
`right_singles_sideline`, `right_doubles_sideline`), named `<across> x <along>`; the
4 corners again under their short names (`near_left` = `near_baseline x
left_doubles_sideline`, ...); and the 2 net-post bases (`left_net_post_base`,
`right_net_post_base`, on the doubles sidelines at the net). The centre line runs only
from each short service line to its baseline, but all its crossings with lines across
the court exist. Click a corner at the outer edge of its lines, as the corner
calibration does; for a crossing, at the outer edge of the sideline and the middle of
the line across (the lines are about 4 cm wide, so the choice matters by 1-3 px).

**Click window** (`--pick-points`): a small court diagram in the top-left corner shows
every known point once. That is 32 points: the 4 corners under their short names, and no
crossing that sits on a corner (`clickable_point_names`). The current point is yellow,
placed ones green. A click places the current point and moves to the next; `n`/`p` step
to the next or previous point without clicking (to skip hidden ones), `d` removes the
current point's click, Enter saves once the placed points fix a homography, Esc or the
close button cancels. Until Enter can save, the prompt gives the reason
(`points_problem`): fewer than 4 points, 3 on one line, or "same point clicked twice"
with both names (possible with `--points`, or in a file from before this list).
`--points` still takes either name for a corner.

## Calibration file

`court.json` holds: `video`, `frame_index`, `frame_width`, `frame_height`,
`corner_order` (`["near_left", "near_right", "far_right", "far_left"]`),
`corner_pixels` (4 x 2), `court_corner_meters` (4 x 2),
`pixel_to_court_homography` (3 x 3, maps pixel `(x, y, 1)` to court meters) and
`max_corner_reprojection_error_px`. `corner_pixels` are always in the video's own
pixels (ShuttleSet corners are saved after scaling).

A calibration from named points (`--points`, `--pick-points`) has the same keys, plus
`corner_pixels_source` (`"clicked"` if all 4 corners were among the points, else
`"projected from the reference points"`: the corners are where the homography puts
them, possibly outside the frame), `reference_points` (one entry per clicked point:
`name`, `pixel`, `court` meters, `error_px` = distance between the clicked pixel and
the point reprojected through the homography), `max_reference_error_px` and
`rms_reference_error_px`. With 4 points the errors are 0 (the homography is exact);
with more they show how consistent the clicks are. The overlay marks each clicked
point with its name and error.

**Video id.** Every per-video output folder of the vision steps is named by
`video_id(video_path)`: `data/calibration/<video id>/`, `data/players/<video id>/`,
`data/shuttle/<video id>/`, `data/hits/<video id>/`, `data/contact/<video id>/`, and the
check images start with it. A video directly under `data/` (or given as a bare file
name) keeps its stem (`data/match.mp4` -> `match`); any other gets its parent folder's
name in front (`data/recordings/stand_in/set1.mp4` -> `stand_in_set1`), so the usual
`set1.mp4` of different recordings no longer share a calibration. Two videos with the
same stem in two folders of the same name would still collide.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `COURT_WIDTH_M` | `6.1` | doubles width (outer edges of the doubles sidelines) |
| `COURT_LENGTH_M` | `13.4` | length (outer edges of the baselines) |
| `NET_Y_M` | `6.7` | net position along the court |
| `SINGLES_SIDELINE_INSET_M` | `0.46` | singles sideline distance from the doubles sideline |
| `SHORT_SERVICE_LINE_FROM_NET_M` | `1.98` | short service line distance from the net |
| `DOUBLES_LONG_SERVICE_LINE_FROM_BASELINE_M` | `0.76` | doubles long service line distance from the baseline |
| `PAINTED_LINE_WIDTH_M` | `0.04` | width of a painted court line |
| `SEARCH_RADIUS_LINE_WIDTHS` | `1.5` | the view check's search window, in painted-line widths at that point (at least 8 px) |
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

### `video_id(video_path, data_dir=DATA_DIR) -> str`, `is_directly_under(folder, path) -> bool`
The name of a video's output folders (see "Video id" above). `DATA_DIR` is
`Path("data")`; paths are compared after resolving, so absolute and relative paths to
the same video get the same id.

```python
>>> from pathlib import Path
>>> from vision.court import video_id
>>> video_id(Path("data/match.mp4")), video_id(Path("data/recordings/stand_in/set1.mp4")), video_id(Path("videos/game_3.mp4")), video_id(Path("match.mp4"))
('match', 'stand_in_set1', 'videos_game_3', 'match')

```

### `calibration_path(video_path, calibration_dir=CALIBRATION_DIR) -> Path`
Where the calibration for `video_path` is stored: `<calibration_dir>/<video id>/court.json`
(`video_id` above).

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
close button, or never opened), else `False`. Used by `click_named_points`. No errors.

```python
>>> from vision.court import window_was_closed
>>> window_was_closed("a window that was never opened")
True

```

### `court_lines_across() -> dict`, `court_lines_along() -> dict`, `court_point_catalog() -> dict`
The `y` of each line across the court, the `x` of each line along it, and every known
point by name (36 names: 30 crossings, 4 corner aliases, 2 net-post bases).

```python
>>> from vision.court import court_lines_across, court_lines_along, court_point_catalog
>>> {name: round(y_m, 2) for name, y_m in court_lines_across().items()}
{'near_baseline': 0.0, 'near_long_service_line': 0.76, 'near_short_service_line': 4.72, 'far_short_service_line': 8.68, 'far_long_service_line': 12.64, 'far_baseline': 13.4}
>>> court_lines_along()
{'left_doubles_sideline': 0.0, 'left_singles_sideline': 0.46, 'centre_line': 3.05, 'right_singles_sideline': 5.64, 'right_doubles_sideline': 6.1}
>>> catalog = court_point_catalog()
>>> len(catalog), catalog["far_long_service_line x right_singles_sideline"], catalog["left_net_post_base"]
(36, (5.64, 12.64), (0.0, 6.7))

```

### `court_point_meters(point_names) -> ndarray`, `cross_2d(first, second) -> float`, `check_points_fix_a_homography(court_points)`
Court meters of named points (`ValueError` for an unknown name); the 2-D cross product;
and the check that a set of court points determines a homography: at least 4, no point
twice, and at least one 4 of them with no 3 on one line. Errors: `ValueError` naming
which condition failed.

```python
>>> from vision.court import check_points_fix_a_homography, court_point_meters
>>> court_point_meters(["near_left", "far_baseline x centre_line"]).tolist()
[[0.0, 0.0], [3.05, 13.4]]
>>> check_points_fix_a_homography(court_point_meters(["near_left", "near_right", "far_right"]))
Traceback (most recent call last):
ValueError: need at least 4 court points, got 3
>>> check_points_fix_a_homography(court_point_meters(["near_left", "near_baseline x centre_line", "near_right", "far_baseline x centre_line"]))
Traceback (most recent call last):
ValueError: the court points don't fix a homography: no 4 of them are free of 3 on one line
>>> check_points_fix_a_homography(court_point_meters(["near_left", "near_baseline x left_doubles_sideline", "far_right", "far_left"]))
Traceback (most recent call last):
ValueError: the same court point is given twice (a corner and its line-crossing name are the same point)
>>> court_point_meters(["mid court"])
Traceback (most recent call last):
ValueError: unknown court point(s) ['mid court']; see python -m vision.court --list-points

```

### `homography_from_court_points(pixel_points, point_names) -> ndarray`
Pixel -> court homography from 4 or more named points: exact for 4 (as for the corners),
least squares (`cv2.findHomography`, no outlier rejection) for more. Errors:
`ValueError` from the checks above, for a pixel/name count mismatch, or non-finite pixels.

Without the near-left corner, from the other 3 corners and 2 crossings on the left
sideline (pixels from the broadcast calibration), the hidden corner comes back where it
was:

```python
>>> from vision.court import court_to_pixels, homography_from_court_points
>>> names = ["near_right", "far_right", "far_left", "near_long_service_line x left_doubles_sideline", "near_short_service_line x left_doubles_sideline"]
>>> pixels = [[987.2, 657.4], [864.2, 301.6], [412.6, 303.0], [299.9, 629.3], [345.7, 496.8]]
>>> court_to_pixels(homography_from_court_points(pixels, names), [[0.0, 0.0]]).round(1).tolist()
[[289.5, 659.6]]

```

### `parse_point_argument(text) -> (str, (float, float))`

```python
>>> from vision.court import parse_point_argument
>>> parse_point_argument("near_short_service_line x centre_line=640,496")
('near_short_service_line x centre_line', (640.0, 496.0))

```

### `point_calibration(video_path, frame_index, frame_size, point_pixels) -> dict`, `save_point_calibration(output_path, ...) -> dict`, `write_calibration(output_path, calibration) -> dict`, `reference_points(calibration) -> (names, pixels, court meters)`
Build (and save) a calibration from `{name: (x, y)}`; `reference_points` returns the
clicked points of a point calibration, or the 4 corners of a corner calibration.

```python
>>> from vision.court import point_calibration, reference_points
>>> calibration = point_calibration("club.mp4", 600, (1280, 720), dict(zip(names, pixels)))
>>> calibration["corner_pixels_source"], [round(value, 1) for value in calibration["corner_pixels"][0]]
('projected from the reference points', [289.5, 659.6])
>>> reference_points(calibration)[0][:2], reference_points({"corner_pixels": [[1, 2]] * 4})[0]
(['near_right', 'far_right'], ['near_left', 'near_right', 'far_right', 'far_left'])

```

### `painted_line_width_px(homography, point_meters, normal_meters, width_m=PAINTED_LINE_WIDTH_M) -> float`
How many pixels a painted line's width covers at a court point, across the line
(`normal_meters` is the unit direction across it). Near a camera the lines are many
pixels wide; on the broadcast they are 1-2 px.

```python
>>> from vision.court import homography_from_corners, painted_line_width_px
>>> homography = homography_from_corners([[289.4, 659.6], [987.2, 657.4], [864.2, 301.6], [412.6, 303.0]])
>>> round(painted_line_width_px(homography, (3.05, 0.0), (0.0, 1.0)), 2), round(painted_line_width_px(homography, (3.05, 13.4), (0.0, 1.0)), 2)
(1.64, 0.69)

```

### `line_sample_counts(frame, homography, start_meters, end_meters, sample_count=40, min_search_radius=8, width_factor=SEARCH_RADIUS_LINE_WIDTHS) -> (int, int)`
The view check's count for one floor line: `(found, inside)`. Each of `sample_count`
points between 5% and 95% of the line gets a search window of
`max(min_search_radius, ceil(width_factor x painted_line_width_px))` pixels; samples
whose window leaves the frame don't count, the rest count as found when a bright
painted line peak is in the window (`brightness_profile_peak_offset`). The window
grows only where lines are wider than about 5 px (a camera close to the court), so on
the broadcast every window stays 8 px.

```python
>>> import cv2
>>> from vision.court import court_line_segments, court_to_pixels, line_sample_counts
>>> drawn_court = np.full((720, 1280, 3), 90, dtype=np.uint8)
>>> for start, end in court_line_segments().values():
...     start_pixel, end_pixel = np.round(court_to_pixels(homography, [start, end])).astype(int)
...     _ = cv2.line(drawn_court, tuple(map(int, start_pixel)), tuple(map(int, end_pixel)), (255, 255, 255), 3)
>>> line_sample_counts(drawn_court, homography, (0.0, 0.0), (6.1, 0.0))
(40, 40)
>>> line_sample_counts(np.full((720, 1280, 3), 90, dtype=np.uint8), homography, (0.0, 0.0), (6.1, 0.0))
(0, 40)
>>> line_sample_counts(drawn_court, homography, (0.0, -3.0), (6.1, -3.0))
(0, 0)

```

### `line_samples_inside_frame(frame_size, homography, start_meters, end_meters, sample_count=40, search_radius=8) -> int`
How many of a line's check samples (as in `line_offsets_to_bright_pixels`) have their
whole search window inside the frame. The view check in `vision/players.py` divides by
this, so lines out of frame don't count as missing.

```python
>>> from vision.court import homography_from_corners, line_samples_inside_frame
>>> homography = homography_from_corners([[289.4, 659.6], [987.2, 657.4], [864.2, 301.6], [412.6, 303.0]])
>>> line_samples_inside_frame((1280, 720), homography, (0.0, 0.0), (6.1, 0.0)), line_samples_inside_frame((1280, 720), homography, (0.0, -3.0), (6.1, -3.0))
(40, 0)

```

### `zone_boundary_segments() -> dict`, `zone_label_points(left_handed=False) -> dict`, `boxed_text(frame, text, origin, scale, thickness)`, `text_scale_for_width(frame_width, scale_at_1280) -> (float, int)`, `draw_zones(frame, homography, left_handed=False) -> ndarray`, `zones_image_path(video_path, checks_dir=ZONES_CHECK_DIR) -> Path`, `write_zones_image(video_path, frame_index, calibration_dir=CALIBRATION_DIR, checks_dir=ZONES_CHECK_DIR, left_handed=False) -> Path`
The six shot zones per half of `analytics.court` drawn on a frame through the
calibration (`python -m vision.court <video> --frame N --zones [--left-handed]`): the
court lines in red, the zone boundaries in yellow (the two depth lines per half at
`DEPTH_EDGES_M` from the net, 2.75 m and 4.00 m, and the centre line over the full
length; the short service line at 1.98 m is a painted line drawn in red, not a zone edge), and each zone's name (front/mid/rear x forehand/backhand) at
its centre. The near half carries the near player's hit zones
(`zone_in_near_half`); the far half carries the landing zones of the near player's
shots, which the `shots` table names from the receiver's side (the point is rotated by
a half turn first, so the receiver's forehand is on the image's left for two
right-handers). Every text is drawn once, in white on a filled black box
(`boxed_text`), at a size scaled to the frame width (`text_scale_for_width`, sizes given
for a 1280-pixel frame). The caption is shrunk further if it would run off the frame.
Output `data/checks/court/<video id>_zones.jpg`; errors: a missing calibration or video,
or an unreadable frame.

```python
>>> from vision.court import draw_zones, zone_boundary_segments, zone_label_points, zones_image_path
>>> sorted(zone_boundary_segments())
['centre', 'far_depth_1', 'far_depth_2', 'near_depth_1', 'near_depth_2']
>>> [tuple(round(v, 2) for v in point) for point in zone_boundary_segments()["near_depth_1"]], zone_boundary_segments()["centre"]
([(0.0, 3.95), (6.1, 3.95)], ((3.05, 0.0), (3.05, 13.4)))
>>> labels = zone_label_points()
>>> len(labels), tuple(round(v, 3) for v in labels[("near", "front_forehand")]), tuple(round(v, 3) for v in labels[("far", "front_forehand")])
(12, (4.575, 5.325), (1.525, 8.075))
>>> sorted(name for half, name in labels if half == "near")
['front_backhand', 'front_forehand', 'mid_backhand', 'mid_forehand', 'rear_backhand', 'rear_forehand']
>>> draw_zones(np.zeros((720, 1280, 3), dtype=np.uint8), homography).shape
(720, 1280, 3)
>>> from vision.court import text_scale_for_width
>>> text_scale_for_width(1280, 0.5), text_scale_for_width(1920, 0.5)
((0.5, 1), (0.75, 1))
>>> zones_image_path(Path("data/recordings/stand_in/set1.mp4")).as_posix()
'data/checks/court/stand_in_set1_zones.jpg'

```

### `draw_point_markers(frame, calibration, color=CORNER_COLOR_BGR) -> ndarray`, `court_diagram(point_names, current_name, placed_names, height_px=300) -> ndarray`, `clickable_point_names() -> list[str]`, `same_point_pairs(point_names) -> list[(str, str)]`, `points_problem(point_pixels) -> str | None`, `points_are_enough(point_pixels) -> bool`, `click_court_points(frame, window_name="court points") -> dict | None`
These are:
- the overlay markers of a point calibration;
- the click window's court diagram;
- the click list, with each physical point once;
- pairs of names given for the same point;
- why the placed points don't fix a homography yet (`None` when they do), and whether they do;
- the click window itself. It returns `{name: (x, y)}`, or `None` when cancelled; it needs
  a display, so it has no doctest.

```python
>>> from vision.court import clickable_point_names, court_diagram, points_are_enough, points_problem, same_point_pairs
>>> clickable = clickable_point_names()
>>> len(clickable), clickable[:4], "near_baseline x left_doubles_sideline" in clickable
(32, ['near_left', 'near_right', 'far_right', 'far_left'], False)
>>> same_point_pairs(["near_left", "near_baseline x left_doubles_sideline", "far_left"])
[('near_left', 'near_baseline x left_doubles_sideline')]
>>> points_problem({"near_left": (1.0, 2.0), "near_baseline x left_doubles_sideline": (1.0, 2.0)})
'same point clicked twice: near_left / near_baseline x left_doubles_sideline'
>>> points_problem({"near_left": (1.0, 2.0)})
'need at least 4 court points, got 1'
>>> points_problem(dict(zip(names, pixels))) is None
True
>>> points_are_enough(dict(zip(names, pixels))), points_are_enough({"near_left": (1.0, 2.0)})
(True, False)
>>> court_diagram(list(catalog), "near_left", {"near_right"}).shape
(300, 147, 3)

```

### `click_named_points(frame, point_names, window_name, color=CORNER_COLOR_BGR) -> np.ndarray | None`
Opens an OpenCV window on `frame` and collects one click per name (`len(point_names)`
x 2 float array, in that order; the prompt reads `click <name> (u = undo, esc = cancel)`),
or returns `None` on Esc. Keys: `u` undo, Enter save once all are placed; the window's
close button acts like Esc; extra clicks are ignored.

### `click_corners(frame, window_name="court calibration") -> np.ndarray | None`
`click_named_points` for the 4 corners (`<corner name> corner`, in `CORNER_NAMES` order).
Needs a display, so it has no doctest; use `--corners` or `calibrate_video(..., corner_pixels=...)` headless.

### `click_net_post_tops(frame, window_name="net post tops") -> np.ndarray | None`
`click_named_points` for the 2 net-post tops (`left net post top`, `right net post top`),
used by `python -m vision.camera` (`docs/vision/camera.md`) without `--net-posts`.

### `calibrate_video(video_path, frame_index=0, corner_pixels=None, calibration_dir=CALIBRATION_DIR, corners_in_shuttleset_label_frame=False, point_pixels=None, pick_points=False) -> Path | None`
With `point_pixels` (`{name: (x, y)}`) or `pick_points=True` it saves a point
calibration instead of a corner one (below otherwise unchanged).
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
>>> saved_path == Path(temporary_folder.name) / "calibration" / video_id(video_path) / "court.json", overlay_path_for(saved_path).is_file()
(True, True)
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
saved ..._tiny/court.json
saved ..._tiny/court_overlay.jpg
0
>>> with contextlib.redirect_stderr(io.StringIO()):
...     main([str(video_path), "--corners", "1;2", "3,4", "5,6", "7,8"])
Traceback (most recent call last):
SystemExit: 2
>>> temporary_folder.cleanup()

```
