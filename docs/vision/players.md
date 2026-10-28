# vision/players.py

Player tracking: finds both players in every frame of a video and writes their
court positions in meters. People are detected with Ultralytics YOLO (person class)
and followed with ByteTrack. Each person's foot point (bottom centre of the box) is
mapped to court meters with the video's court calibration (`vision/court.py`).
Only people standing on the court (with a 1 m margin, 2 m behind the far baseline) are kept, which drops the
umpire, line judges, coaches and the crowd. The near-side player is labelled `me`
and the far-side player `opp`, so the camera must stand behind your own baseline.

Court coordinates follow the same contract as `vision/court.py`: meters, origin at
the near-left corner, `x` across (0 to 6.1), `y` along the court (0 to 13.4), net
at `y = 6.7`.

How a frame is processed:
1. **View check.** Broadcast video cuts to close-ups and replays where the court
   calibration means nothing. For every floor line, 40 points are checked for a
   bright painted line nearby (`line_offsets_to_bright_pixels` from `vision/court.py`).
   If fewer than 60% are found, the frame isn't the main camera view: it is
   skipped, not passed to the tracker, and counted. On the main view about 97-99% of
   samples are found; on close-ups and replays 1-26%.
2. **Detection and tracking.** YOLO with ByteTrack, persons only, at the video's
   full width. Overlapping boxes with IoU above 0.5 are merged (non-maximum
   suppression; YOLO's default is 0.7), because a second, wider box around a
   player with an outstretched arm otherwise starts a new track.
3. **On-court filter.** Foot point to meters; keep people with
   `-1 <= x <= 7.1` and `-1 <= y <= 15.4`. The margin behind the far baseline is 2 m
   because the calibration assumes feet on the floor: when the far player jumps, the
   foot point rises in the image and maps up to about 1 m further back. A jumping
   near player maps towards the net instead, so the near end keeps 1 m. On the
   broadcast the nearest bystanders map to `x` beyond -1.9 or 8.3 and `y` beyond 17.
4. **Labelling.** A track's side is fixed the first time it is seen on court:
   near half (`y < 6.7`) is `me`, far half is `opp`. Each frame, each player gets
   the detection of the track it had in the previous frame; if that track is gone,
   the most confident on-court track of that side.

Because a track's side is fixed when it first appears, `me` and `opp` follow the
track IDs. A player whose position is reported in the wrong half therefore shows a
tracking error (an ID swap); it isn't hidden by re-labelling each frame.

The per-frame steps (`foot_points`, `court_positions_of_feet`, `on_court_mask`,
`side_of_court`, `court_line_sample_share`, `is_main_view`, `sides_with_new_tracks`,
`chosen_detection_index`, `label_players`) are pure: arrays in, arrays or rows out.
`track_people` is the thin wrapper around the YOLO call. Only `load_person_tracker`,
`track_players`, `split_frames_by_view` and `save_positions` touch files.

## Model weights

Weights live under `data/models/` (gitignored), never in the project root. The
default is `data/models/yolo11s.pt`. To download it:

```
cd data/models && ../../.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"
```

Ultralytics downloads into the current folder, which is why the command changes
into `data/models` first. `load_person_tracker` sets `YOLO_AUTOINSTALL=False` (unless
it is already set) so Ultralytics doesn't pip-install packages by itself; ByteTrack
needs `lap`, which is in `requirements.txt`. Ultralytics is AGPL-3.0.

## Command

```
.venv/bin/python -m vision.players <video> (--start N --end N | --shuttleset-rally MATCH SET RALLY) [options]
```

| Argument | Meaning |
|---|---|
| `video` | path to the match video; needs a calibration (`python -m vision.court`) |
| `--start`, `--end` | first and last frame to track (inclusive) |
| `--shuttleset-rally` | track one labelled rally: ShuttleSet match folder, set number, rally number. Tracks from the rally's first hit frame to its last, plus `--padding` frames each side |
| `--padding` | frames before and after a ShuttleSet rally (default 30); the start is clamped to frame 0 |
| `--calibration` | `court.json` to use (default `data/calibration/<video stem>/court.json`) |
| `--model` | YOLO weights (default `data/models/yolo11s.pt`) |
| `--overlay-frame N` | save an overlay of frame N to `data/checks/players/<video stem>_players_frame<N>.jpg`; repeatable. `me` in blue, `opp` in red with track id and meters, everyone else grey |
| `--views-only` | only run the view check and print how many frames are main view and how many are skipped; no YOLO, about 12 ms per frame |

Output: `data/players/<video stem>/positions_<start>_<end>.csv`, one row per
player per main-view frame where that player was found:

| Column | Type | Meaning |
|---|---|---|
| `frame` | int | frame index in the video |
| `player` | str | `me` (near side) or `opp` (far side) |
| `x_m`, `y_m` | float | foot position in court meters, rounded to mm |
| `track_id` | int | ByteTrack id (only meaningful within one run) |
| `confidence` | float | YOLO person confidence |

Printed summary (example from a ShuttleSet rally):

```
frames: 747, main view: 718, skipped (not main view): 29
me: detected 687/687 frames, 1 track ids, one id (#2) 687/687 = 100.0%, in own half 100.0%
opp: detected 687/687 frames, 1 track ids, one id (#1) 687/687 = 100.0%, in own half 100.0%
saved data/players/match/positions_14201_14947.csv
```

The player lines measure only main-view frames between the first and the last hit
(with `--start`/`--end`: all main-view frames in the range). Exit codes: `0` on
success; `1` with `error: ...` on stderr when the video, calibration, weights,
set CSV or rally is missing, the start frame is negative, the range is reversed,
or no frame in the range could be read (for example a range past the end of the
video: `error: no frames could be read from 900000 to 900005 (the video has 140250 frames)`);
nothing is written in those cases. `2` for invalid arguments.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `PLAYERS` | `("me", "opp")` | player labels, near side first |
| `PERSON_CLASS_ID` | `0` | COCO class id of a person |
| `COURT_MARGIN_M` | `1.0` | how far outside the sidelines and near baseline a foot point may be and still count as on court |
| `BEHIND_FAR_BASELINE_MARGIN_M` | `2.0` | the same, behind the far baseline (jumping far player) |
| `DUPLICATE_BOX_MAX_IOU` | `0.5` | overlap above which YOLO keeps only the more confident of two boxes |
| `LINE_SAMPLES_PER_LINE` | `40` | samples per floor line in the view check |
| `MAIN_VIEW_MIN_LINE_SAMPLE_SHARE` | `0.6` | minimum share of line samples found for a main-view frame |
| `DEFAULT_MODEL_PATH` | `Path("data/models/yolo11s.pt")` | default weights |
| `PLAYERS_DIR` | `Path("data/players")` | root of the position files |
| `CHECKS_DIR` | `Path("data/checks/players")` | overlay images |
| `POSITION_COLUMNS` | `["frame", "player", "x_m", "y_m", "track_id", "confidence"]` | columns of the position file |

```python
>>> from vision.players import COURT_MARGIN_M, MAIN_VIEW_MIN_LINE_SAMPLE_SHARE, PLAYERS, POSITION_COLUMNS
>>> PLAYERS, COURT_MARGIN_M, MAIN_VIEW_MIN_LINE_SAMPLE_SHARE
(('me', 'opp'), 1.0, 0.6)
>>> POSITION_COLUMNS
['frame', 'player', 'x_m', 'y_m', 'track_id', 'confidence']

```

## Classes

### `FrameDetections(boxes_xyxy, track_ids, confidences)`
The tracked people in one frame. `boxes_xyxy`: `(N, 4)` float array of
`left, top, right, bottom` pixels; `track_ids`: `(N,)` int array; `confidences`:
`(N,)` float array. Frozen dataclass. `empty_detections()` returns one with `N = 0`.

### `PlayerTrackingState(side_by_track_id={}, track_id_by_player={})`
What labelling remembers between frames: the side (`me`/`opp`) fixed for each
track id, and the track id each player had last. Start with `PlayerTrackingState()`.

### `TrackingRun(positions, main_view_frames, skipped_frames, overlay_paths)`
Result of `track_players`: the positions `DataFrame` (`POSITION_COLUMNS`), the
frame indices that were tracked, the frame indices skipped by the view check, and
the overlay images written.

## Functions

The examples use the calibration of a real broadcast video, built from its 4 court
corner pixels:

```python
>>> import numpy as np
>>> from vision.court import homography_from_corners
>>> homography = homography_from_corners([[289.4, 659.6], [987.2, 657.4], [864.2, 301.6], [412.6, 303.0]])

```

### `foot_points(boxes_xyxy) -> ndarray`
Bottom centre of each box, `(N, 2)` pixels: `((left + right) / 2, bottom)`.
Input: anything shaped `(N, 4)` or a single box of 4 numbers.

```python
>>> from vision.players import foot_points
>>> foot_points([[598, 388, 678, 532], [600, 240, 660, 350]])
array([[638., 532.],
       [630., 350.]])

```

### `court_positions_of_feet(homography, boxes_xyxy) -> ndarray`
Foot points mapped to court meters, `(N, 2)`; `(0, 2)` for no boxes.
`homography` is the pixel -> court `3 x 3` matrix from the calibration.

```python
>>> from vision.players import court_positions_of_feet
>>> boxes = np.array([[598, 388, 678, 532], [600, 240, 660, 350], [200, 330, 244, 388], [1028, 231, 1109, 357]])
>>> positions = court_positions_of_feet(homography, boxes)
>>> positions.round(2)
array([[ 3.05,  3.52],
       [ 2.94, 10.82],
       [-1.93,  9.06],
       [ 8.4 , 10.4 ]])
>>> court_positions_of_feet(homography, np.zeros((0, 4))).shape
(0, 2)

```

The four boxes are the near player, the far player, a line judge to the left of the
court and the umpire's chair on the right.

### `on_court_mask(court_positions, margin_m=1.0, far_margin_m=2.0) -> ndarray`
Boolean `(N,)`: `True` where `-margin_m <= x <= 6.1 + margin_m` and `-margin_m <= y <= 13.4 + far_margin_m`.

```python
>>> from vision.players import on_court_mask
>>> on_court_mask(positions)
array([ True,  True, False, False])
>>> on_court_mask([[-0.9, 15.3], [-1.1, 5.0], [3.0, -1.2], [3.0, 17.2]])
array([ True, False, False, False])

```

### `side_of_court(y_m) -> str`
`"me"` for the near half (`y < 6.7`), `"opp"` otherwise (the net line counts as far).

```python
>>> from vision.players import side_of_court
>>> side_of_court(3.52), side_of_court(10.82), side_of_court(6.7)
('me', 'opp', 'opp')

```

### `court_line_sample_share(frame, homography) -> float`
Share (0 to 1) of the 480 floor-line samples (12 lines x 40) where a bright painted
line is found within 8 px of where the calibration puts it. Input: `frame`, BGR or
grayscale `uint8` image. Even a perfect drawing of the court scores below 1: where two
lines cross, the brightness profile has no single peak and the sample doesn't count.

```python
>>> import cv2
>>> from vision.court import court_line_segments, court_to_pixels
>>> from vision.players import court_line_sample_share
>>> court_frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
>>> for name, (start, end) in court_line_segments().items():
...     start_pixel, end_pixel = np.round(court_to_pixels(homography, [start, end])).astype(int)
...     _ = cv2.line(court_frame, tuple(map(int, start_pixel)), tuple(map(int, end_pixel)), (255, 255, 255), 3)
>>> round(court_line_sample_share(court_frame, homography), 2)
0.94
>>> court_line_sample_share(np.full((720, 1280, 3), 90, dtype=np.uint8), homography)
0.0

```

### `is_main_view(line_sample_share, minimum_share=0.6) -> bool`

```python
>>> from vision.players import is_main_view
>>> is_main_view(0.98), is_main_view(0.19)
(True, False)

```

### `sides_with_new_tracks(side_by_track_id, track_ids, court_positions) -> dict`
Copy of `side_by_track_id` with a side added for every track id not seen before,
from its `y`. Sides already known never change. The input dict is not modified.

```python
>>> from vision.players import sides_with_new_tracks
>>> sides_with_new_tracks({1: "opp"}, [1, 2], [[3.0, 5.0], [3.0, 2.0]])
{1: 'opp', 2: 'me'}

```

### `chosen_detection_index(candidate_indices, track_ids, confidences, previous_track_id) -> int | None`
Which candidate detection a player gets: the one whose track id equals
`previous_track_id`, otherwise the most confident; `None` if there are no candidates.

```python
>>> from vision.players import chosen_detection_index
>>> track_ids, confidences = np.array([5, 7, 9]), np.array([0.6, 0.9, 0.8])
>>> chosen_detection_index([0, 1], track_ids, confidences, previous_track_id=5)
0
>>> chosen_detection_index([0, 1], track_ids, confidences, previous_track_id=None)
1
>>> chosen_detection_index([], track_ids, confidences, previous_track_id=5) is None
True

```

### `label_players(frame_index, detections, court_positions, state, margin_m=1.0) -> (list[dict], PlayerTrackingState)`
One frame of labelling. Inputs: the frame index, the frame's `FrameDetections`,
their court positions (`court_positions_of_feet`) and the state from the previous
frame. Output: up to two rows (`POSITION_COLUMNS` as keys, `me` first; a player
with no on-court detection of its side gets no row) and the new state. The input
state is not modified.

```python
>>> from vision.players import FrameDetections, PlayerTrackingState, label_players
>>> detections = FrameDetections(boxes, np.array([2, 1, 6, 3]), np.array([0.92, 0.88, 0.70, 0.87]))
>>> rows, state = label_players(16900, detections, positions, PlayerTrackingState())
>>> for row in rows:
...     print(row)
{'frame': 16900, 'player': 'me', 'x_m': 3.047, 'y_m': 3.521, 'track_id': 2, 'confidence': 0.92}
{'frame': 16900, 'player': 'opp', 'x_m': 2.944, 'y_m': 10.817, 'track_id': 1, 'confidence': 0.88}
>>> state
PlayerTrackingState(side_by_track_id={2: 'me', 1: 'opp'}, track_id_by_player={'me': 2, 'opp': 1})

```

If the far player's track later ends and a new track appears in the near half, it
is a new `me` candidate, not `opp`; `opp` simply has no row until a far-side track exists.

```python
>>> near_only = FrameDetections(boxes[:1], np.array([8]), np.array([0.9]))
>>> near_rows, _ = label_players(16901, near_only, positions[:1], state)
>>> [(row["player"], row["track_id"]) for row in near_rows]
[('me', 8)]

```

### `id_continuity(positions, player, frames) -> dict`
How well a player kept one track id over `frames` (any iterable of frame indices,
usually the main-view frames of one rally). Output keys: `player`, `frames`,
`frames_detected` (rows of that player in those frames), `track_ids` (distinct ids),
`modal_track_id` (most common id, `None` if never detected), `frames_with_modal_id`,
and `share_with_one_id` = `frames_with_modal_id / frames`. Frames without a
detection count against the share.

```python
>>> import pandas as pd
>>> from vision.players import id_continuity
>>> positions_table = pd.DataFrame({
...     "frame": [10, 11, 12, 13, 10, 11, 13],
...     "player": ["me", "me", "me", "me", "opp", "opp", "opp"],
...     "x_m": 3.0, "y_m": [2.0, 2.1, 2.2, 2.3, 10.0, 10.1, 10.2],
...     "track_id": [2, 2, 2, 2, 1, 1, 4], "confidence": 0.9})
>>> id_continuity(positions_table, "me", range(10, 14))["share_with_one_id"]
1.0
>>> continuity = id_continuity(positions_table, "opp", range(10, 14))
>>> continuity["frames_detected"], continuity["track_ids"], continuity["modal_track_id"], continuity["share_with_one_id"]
(3, 2, 1, 0.5)

```

### `share_in_own_half(positions, player) -> float`
Share of a player's rows in that player's half (`me`: `y < 6.7`, `opp`: `y >= 6.7`);
`0.0` if the player has no rows.

```python
>>> from vision.players import share_in_own_half
>>> share_in_own_half(positions_table, "me"), share_in_own_half(positions_table, "opp")
(1.0, 1.0)

```

### `rally_frame_range(shots, rally) -> (int, int)`
First and last hit frame of a rally in a ShuttleSet set table (`rally`,
`frame_num` columns). Errors: `ValueError` if the rally has no labelled frames.

```python
>>> from vision.players import rally_frame_range
>>> shots = pd.DataFrame({"rally": [1, 1, 1, 2], "frame_num": [12337.0, 12359.0, 12406.0, 12812.0]})
>>> rally_frame_range(shots, 1)
(12337, 12406)
>>> rally_frame_range(shots, 9)
Traceback (most recent call last):
ValueError: rally 9 has no labeled shots

```

### `draw_players(frame, detections, frame_rows) -> ndarray`
Copy of `frame` with every tracked box: players in their colour (`me` blue,
`opp` red) with label `me #2 (3.0, 3.5) m` and a dot at the foot point, everyone
else grey with just the track id.

```python
>>> from vision.players import draw_players
>>> blank = np.zeros((720, 1280, 3), dtype=np.uint8)
>>> drawn = draw_players(blank, detections, rows)
>>> drawn[532, 638].tolist(), int(blank.sum())
([255, 160, 0], 0)

```

### `load_person_tracker(model_path) -> YOLO`
Loads YOLO weights. Errors: `FileNotFoundError` with the download command if the
file doesn't exist (no automatic download into the current folder). Importing
Ultralytics changes numpy's global print options; they are saved before the import
and restored afterwards, so arrays print the same before and after loading the tracker.

```python
>>> from pathlib import Path
>>> from vision.players import load_person_tracker
>>> load_person_tracker(Path("data/models/missing.pt"))  # doctest: +ELLIPSIS
Traceback (most recent call last):
FileNotFoundError: model weights not found: data/models/missing.pt (download with: cd data/models && python -c "from ultralytics import YOLO; YOLO('missing.pt')")

```

### `require_video_file(video_path) -> None`
Raises `FileNotFoundError: video not found: <path>` if the video file doesn't exist.
Called first by `track_players` and `split_frames_by_view`, so a wrong video path is
reported as such and not as a missing calibration.

```python
>>> from vision.players import require_video_file
>>> require_video_file(Path("data/missing.mp4"))
Traceback (most recent call last):
FileNotFoundError: video not found: data/missing.mp4

```

### `check_frame_range(start_frame, end_frame) -> None`
Errors: `ValueError` if `start_frame < 0` or `end_frame < start_frame`. A negative
start would make OpenCV seek to frame 0 while frames were numbered from the negative
start, shifting every frame index.

```python
>>> from vision.players import check_frame_range
>>> check_frame_range(0, 0)
>>> check_frame_range(-5, 3)
Traceback (most recent call last):
ValueError: start frame must be >= 0, got -5
>>> check_frame_range(10, 5)
Traceback (most recent call last):
ValueError: end frame 5 is before start frame 10

```

### `frames_in_range(capture, start_frame, end_frame) -> iterator of (int, ndarray)`
Yields `(frame_index, frame)` from `start_frame` to `end_frame` inclusive, stopping
early at the end of the video, with indices that match the video's own frame numbers.
Errors: those of `check_frame_range`, and `ValueError` if not a single frame could be
read (raised once the iteration ends).

```python
>>> import tempfile
>>> video_path = Path(tempfile.mkdtemp()) / "five_frames.avi"
>>> writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
>>> for frame_index in range(5):
...     _ = writer.write(np.full((48, 64, 3), frame_index * 50, dtype=np.uint8))
>>> writer.release()
>>> from vision.players import frames_in_range
>>> capture = cv2.VideoCapture(str(video_path))
>>> [(frame_index, int(frame[0, 0, 0])) for frame_index, frame in frames_in_range(capture, 3, 10)]
[(3, 150), (4, 200)]
>>> list(frames_in_range(capture, 7, 9))
Traceback (most recent call last):
ValueError: no frames could be read from 7 to 9 (the video has 5 frames)
>>> capture.release()

```

### `track_people(tracker, frame) -> FrameDetections`
Runs YOLO + ByteTrack on one frame (persons only, NMS IoU 0.5, `persist=True` so
ids carry over between calls on the same tracker). Returns `empty_detections()` when
nobody is tracked.

### `track_players(video_path, start_frame, end_frame, calibration_file=None, model_path=DEFAULT_MODEL_PATH, overlay_frames=(), checks_dir=CHECKS_DIR) -> TrackingRun`
Tracks frames `start_frame` to `end_frame` (inclusive, stops early at the end of
the video): view check, tracking, on-court filter and labelling for each frame,
and saves overlays for the frames listed in `overlay_frames`. Errors: `ValueError`
for a negative start, a reversed range, or a range with no readable frame (see
`frames_in_range`); `FileNotFoundError` for a missing video (checked first after the
range), calibration or weights file; `OSError` if the video can't be opened.

### `split_frames_by_view(video_path, start_frame, end_frame, calibration_file=None) -> (list[int], list[int])`
Only the view check: main-view frame indices and skipped frame indices. Same
errors as `track_players` (no weights needed).

### `positions_path(video_path, start_frame, end_frame, players_dir=PLAYERS_DIR) -> Path`

```python
>>> from vision.players import positions_path
>>> positions_path(Path("data/match.mp4"), 14201, 14947).as_posix()
'data/players/match/positions_14201_14947.csv'

```

### `save_positions(positions, output_path) -> Path`
Writes the positions as CSV (creating folders) and returns `output_path`.

### `frames_to_track(parsed) -> (int, int, list[int])`
From parsed command arguments: first and last frame to track and the frames to
measure. With `--shuttleset-rally` that is the rally's first hit minus `--padding`
(never below 0), its last hit plus `--padding`, and the frames from first to last hit; otherwise
`--start`, `--end` and every frame between. Errors: `ValueError` if neither is given
or the rally has no labels; `FileNotFoundError` if the set CSV is missing.

```python
>>> import argparse
>>> from vision.players import frames_to_track
>>> start, end, measured = frames_to_track(argparse.Namespace(shuttleset_rally=None, start=100, end=104, padding=30))
>>> start, end, measured
(100, 104, [100, 101, 102, 103, 104])
>>> frames_to_track(argparse.Namespace(shuttleset_rally=None, start=100, end=None, padding=30))
Traceback (most recent call last):
ValueError: give --start and --end, or --shuttleset-rally

```

### `summary_lines(run, measured_frames) -> list[str]`
The printed summary: a frame/view count line, then one line per player with
`id_continuity` and `share_in_own_half` over `measured_frames`.

```python
>>> from vision.players import TrackingRun, summary_lines
>>> run = TrackingRun(positions_table, main_view_frames=[10, 11, 12, 13], skipped_frames=[9], overlay_paths=[])
>>> for line in summary_lines(run, [10, 11, 12, 13]):
...     print(line)
frames: 5, main view: 4, skipped (not main view): 1
me: detected 4/4 frames, 1 track ids, one id (#2) 4/4 = 100.0%, in own half 100.0%
opp: detected 3/4 frames, 2 track ids, one id (#1) 2/4 = 50.0%, in own half 100.0%

```

### `main(arguments) -> int`
Command entry point: parses `arguments` (without the program name), runs the
tracking or view count, prints the summary, and returns the exit code.
