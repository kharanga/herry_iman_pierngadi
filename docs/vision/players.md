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
   If fewer are found than 60% of the share found on the calibration frame (the frame
   the court was calibrated on, in `court.json`), the frame isn't the main camera view:
   it is skipped, not passed to the tracker, and counted. On the broadcast the
   calibration frame scores 0.99, so the threshold is 0.594; the main view scores about
   0.97-0.99 and close-ups and replays 0.01-0.26. A camera close to the court finds
   fewer samples on every frame (lines wider than the search window, lines behind the
   net), which is why the threshold is relative: the phone camera of the stand-in club
   video scores 0.515 on its calibration frame, so its threshold is 0.309.
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
`chosen_detection_index`, `label_players`, `detections_of_result`) are pure: arrays in,
arrays or rows out. `track_people` is the thin wrapper around the YOLO call; it takes a
list of frames and runs them as one batch. `track_players` reads the video in chunks of
`DETECTION_BATCH_FRAMES` frames (`chunked`), runs the view check on each frame
(`frames_with_view`), drops the frames that fail it and sends the rest of the chunk to
`track_people` in one call. ByteTrack still sees the frames one after the other, so track
ids are the same as with one frame per call. Reading and view-checking the next chunks
happens on a background thread (`prefetched`) while the GPU works on the current one.
Before the first chunk, `warm_up_tracker` runs the model once on a black frame: the first
call on the GPU compiles its kernels and takes seconds, and Ultralytics drops the boxes of
any batch whose post-processing runs past its time limit (about 2.4 s), so without the
warm-up the first batch of a run could come back empty. Only `load_person_tracker`,
`track_players`, `split_frames_by_view` and `save_positions` touch files; `track_people`
and `warm_up_tracker` are the only functions that call the model. `track_people` is the
seam for another detector: frames in, person boxes with track ids out. A HailoRT
implementation on the Pi would be a second version of that one function (its detector
followed by ByteTrack), and nothing else in the module would change.

YOLO runs on the Mac's GPU (`mps`, Apple Metal) when PyTorch can see it, else on the CPU
(`default_device`); `--device` overrides. The batch size is a trade-off on an 8 GB
machine: at 1280 px each frame in a batch needs about 150 MB of GPU memory for its
activations, so 8 frames use about 2.3 GB, 15 about 4.5 GB, and 30 more than the
machine has; the run then swaps and is slower than the CPU. The default 8 is the largest
size measured that doesn't slow down (see `docs/results/players.md`). On the CPU a
batch brings nothing and costs memory, so `--device cpu` is best with `--batch-frames 1`.

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
.venv/bin/python -m vision.players <video> (--start N --end N | --shuttleset-rally MATCH SET RALLY | --stretches CSV [--merge]) [options]
.venv/bin/python -m vision.players <video> --plan-stretches CSV [--stretch-seconds 180] [--stretch-count N]
```

| Argument | Meaning |
|---|---|
| `video` | path to the match video; needs a calibration (`python -m vision.court`) |
| `--start`, `--end` | first and last frame to track (inclusive) |
| `--shuttleset-rally` | track one labelled rally: ShuttleSet match folder, set number, rally number. Tracks from the rally's first hit frame to its last, plus `--padding` frames each side |
| `--stretches CSV` | track every `start_frame, end_frame` row of the file (`python -m vision.label_offset <video> --stretches` writes one) with one loaded model, each into its own `positions_<start>_<end>.csv` / `boxes_` pair and record. Rows that already have both files are skipped, so a stopped run resumes where it stopped (a stretch's files are written only once it is complete). Prints `N stretches: T tracked (F frames), S already tracked` |
| `--merge` | with `--stretches`: after tracking, join every stretch's files into `positions_0_<last>.csv` and `boxes_0_<last>.csv` (last = the video's frame count - 1), the files `vision.hits --start 0 --end <last>` and `vision.label_ui` read. Frames outside the stretches have no rows; the movement detector treats them like frames off the main view. The merged tracking record is the first stretch's, plus the list of `stretches`. Every stretch must be tracked first |
| `--plan-stretches CSV` | write a stretches file for the video and exit. Without `--stretch-count`: back-to-back stretches of `--stretch-seconds` (default 180) covering the whole video, so a long video tracks in resumable pieces. With `--stretch-count N`: N stretches of that length, one centred in each of N equal slices of the video, for tracking a spread sample of a long video |
| `--padding` | frames before and after a ShuttleSet rally (default 30); the start is clamped to frame 0 |
| `--calibration` | `court.json` to use (default `data/calibration/<video id>/court.json`) |
| `--model` | YOLO weights (default `data/models/yolo11s.pt`) |
| `--overlay-frame N` | save an overlay of frame N to `data/checks/players/<video id>_players_frame<N>.jpg`; repeatable. `me` in blue, `opp` in red with track id and meters, everyone else grey |
| `--views-only` | only run the view check and print how many frames are main view and how many are skipped; no YOLO, about 12 ms per frame |
| `--device` | `mps` or `cpu`; default `mps` when PyTorch reports it available, else `cpu` |
| `--batch-frames N` | frames read per chunk and sent to YOLO in one call (default `DETECTION_BATCH_FRAMES`, 8); `1` tracks one frame at a time |
| `--frame-stride N` | track every Nth frame only (default 1). The files keep the video's frame numbers (0, 2, 4, ... for stride 2) and cover the sampled frames only; skipped frames are decoded with `grab()` (no conversion) and never view-checked. The record file (below) tells `vision.hits` and `vision.features` the stride, so they treat the tracks as a video at `fps / N` |
| `--players-dir DIR` | root of the output files (default `data/players`); use another root to keep a second configuration's tracks next to the default ones |

Output: `data/players/<video id>/positions_<start>_<end>.csv`, one row per
player per main-view frame where that player was found, and next to it
`boxes_<start>_<end>.csv` with the same rows' pixel boxes (`BOX_COLUMNS`: `frame`,
`player`, `left`, `top`, `right`, `bottom`), which the movement hit detector uses for
the box-shape features, and `positions_<start>_<end>.csv.tracking.json`, the tracking
record: `frame_stride`, `video_fps`, `effective_fps` (`video_fps / frame_stride`),
`frame_width`, `model`, `device`, `batch_frames`. A missing record means stride 1.

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
| `MAX_DETECTION_WIDTH_PX` | `1280` | YOLO runs at the frame width, but at most this (a 1920-wide video is detected at 1280, about twice as fast) |
| `DETECTION_BATCH_FRAMES` | `8` | frames per chunk and per YOLO call (measured best on an 8 GB M2; 30 runs out of GPU memory) |
| `PREFETCHED_CHUNKS` | `2` | chunks read and view-checked ahead of the YOLO call |
| `GPU_DEVICE`, `CPU_DEVICE` | `"mps"`, `"cpu"` | the two devices `default_device` chooses between |
| `LINE_SAMPLES_PER_LINE` | `40` | samples per floor line in the view check |
| `MAIN_VIEW_MIN_LINE_SAMPLE_SHARE` | `0.6` | `is_main_view`'s default threshold (the commands use the relative one below) |
| `MAIN_VIEW_SHARE_OF_CALIBRATION_FRAME` | `0.6` | a main-view frame needs at least this fraction of the calibration frame's share |
| `DEFAULT_MODEL_PATH` | `Path("data/models/yolo11s.pt")` | default weights |
| `PLAYERS_DIR` | `Path("data/players")` | root of the position files |
| `CHECKS_DIR` | `Path("data/checks/players")` | overlay images |
| `TRACKING_RECORD_SUFFIX` | `".tracking.json"` | appended to the positions file name for the tracking record |
| `POSITION_COLUMNS` | `["frame", "player", "x_m", "y_m", "track_id", "confidence"]` | columns of the position file |
| `BOX_COLUMNS` | `["frame", "player", "left", "top", "right", "bottom"]` | columns of the box file (pixels, one decimal) |

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

### `TrackingRun(positions, main_view_frames, skipped_frames, overlay_paths, boxes)`
Result of `track_players`: the positions `DataFrame` (`POSITION_COLUMNS`), the
frame indices that were tracked, the frame indices skipped by the view check, the
overlay images written, and the labelled players' boxes (`BOX_COLUMNS`, one row per
position row).

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
Share (0 to 1) of the floor-line samples (12 lines x 40) where a bright painted line is
found within 8 px of where the calibration puts it, counting only samples whose search
window is inside the frame (`vision.court.line_samples_inside_frame`): a camera that
doesn't see the whole court (a near corner below the frame) isn't mistaken for a
cutaway. On a broadcast frame all 480 are inside. Input: `frame`, BGR or
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

### `main_view_min_share(calibration_frame_share, share_of_calibration_frame=0.6) -> float`, `calibration_frame_share(video_path, calibration) -> float`
The threshold `track_players` and `split_frames_by_view` use, from the share on the
calibration frame; and that share, read from the video at the calibration's
`frame_index` with its homography. Errors: `ValueError` if the frame can't be read
(plus those of `open_video`).

```python
>>> import tempfile
>>> from pathlib import Path
>>> from vision.players import calibration_frame_share, main_view_min_share
>>> main_view_min_share(0.99), round(main_view_min_share(0.515), 3)
(0.594, 0.309)
>>> video_folder = Path(tempfile.mkdtemp())
>>> writer = cv2.VideoWriter(str(video_folder / "court.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720))
>>> _ = writer.write(court_frame)
>>> _ = writer.write(np.full((720, 1280, 3), 90, dtype=np.uint8))
>>> writer.release()
>>> round(calibration_frame_share(video_folder / "court.avi", {"frame_index": 0, "pixel_to_court_homography": homography}), 2)
0.94
>>> calibration_frame_share(video_folder / "court.avi", {"frame_index": 5, "pixel_to_court_homography": homography})  # doctest: +ELLIPSIS
Traceback (most recent call last):
ValueError: could not read the calibration frame 5 of .../court.avi

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

### `player_box_rows(detections, frame_rows) -> list[dict]`
The pixel box of each labelled player in one frame, as `BOX_COLUMNS` rows, in the
same order as the `label_players` rows.

```python
>>> from vision.players import player_box_rows
>>> player_box_rows(detections, rows)
[{'frame': 16900, 'player': 'me', 'left': 598.0, 'top': 388.0, 'right': 678.0, 'bottom': 532.0}, {'frame': 16900, 'player': 'opp', 'left': 600.0, 'top': 240.0, 'right': 660.0, 'bottom': 350.0}]

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

### `default_device() -> str`
`"mps"` when `torch.backends.mps.is_available()`, else `"cpu"`. Never raises.

```python
>>> from vision.players import CPU_DEVICE, GPU_DEVICE, default_device
>>> default_device() in (GPU_DEVICE, CPU_DEVICE)
True

```

### `chunked(items, size) -> iterator of list`
Groups any iterable into lists of `size` items; the last list is shorter when the
count doesn't divide. Lazy: it reads only one chunk ahead, so video frames aren't all
held in memory. Errors: `ValueError` for a size below 1.

```python
>>> from vision.players import chunked
>>> list(chunked(range(7), 3))
[[0, 1, 2], [3, 4, 5], [6]]
>>> list(chunked([], 3))
[]
>>> list(chunked(range(2), 0))
Traceback (most recent call last):
ValueError: chunk size must be >= 1, got 0

```

### `prefetched(items, depth=PREFETCHED_CHUNKS) -> iterator`
Yields the items of any iterable in order, producing them on a background thread up to
`depth` items ahead of the consumer. An exception raised while producing is re-raised in
the consumer at that point; closing the generator early (or an exception in the
consumer) stops the producer and waits for it, so a video capture that feeds it is never
released while the thread still reads from it.

```python
>>> from vision.players import prefetched
>>> list(prefetched(chunked(range(7), 3)))
[[0, 1, 2], [3, 4, 5], [6]]
>>> def failing_after_one():
...     yield 1
...     raise ValueError("decoder failed")
>>> list(prefetched(failing_after_one()))
Traceback (most recent call last):
ValueError: decoder failed
>>> early = prefetched(iter(range(1000)))
>>> next(early), next(early)
(0, 1)
>>> early.close()

```

### `frames_with_view(frames, homography, minimum_share) -> iterator of (int, ndarray, bool)`
Adds the view check to a stream of `(frame_index, frame)` pairs: each item becomes
`(frame_index, frame, in_main_view)` with `is_main_view(court_line_sample_share(frame,
homography), minimum_share)`. Pure and lazy. With `court_frame` from the
`court_line_sample_share` example and a blank frame:

```python
>>> from vision.players import frames_with_view
>>> blank_frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
>>> [(index, in_view) for index, _, in_view in frames_with_view([(5, court_frame), (6, blank_frame)], homography, 0.6)]
[(5, True), (6, False)]

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

### `check_frame_stride(frame_stride) -> None`, `sampled_frames(start_frame, end_frame, frame_stride=1) -> ndarray`
A stride below 1 is a `ValueError`; `sampled_frames` is the frame numbers a stride
keeps, `start_frame, start_frame + stride, ...` up to `end_frame`.

```python
>>> from vision.players import check_frame_stride, sampled_frames
>>> sampled_frames(10, 17, 3)
array([10, 13, 16])
>>> check_frame_stride(0)
Traceback (most recent call last):
ValueError: frame stride must be >= 1, got 0

```

### `frames_in_range(capture, start_frame, end_frame, frame_stride=1) -> iterator of (int, ndarray)`
Yields `(frame_index, frame)` from `start_frame` to `end_frame` inclusive, stopping
early at the end of the video, with indices that match the video's own frame numbers;
with a stride, only every `frame_stride`-th frame from `start_frame` (the others are
skipped with `grab()`). Errors: those of `check_frame_range` and `check_frame_stride`,
and `ValueError` if not a single frame could be read (raised once the iteration ends).

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
>>> [(frame_index, int(frame[0, 0, 0])) for frame_index, frame in frames_in_range(capture, 0, 4, 2)]
[(0, 0), (2, 100), (4, 200)]
>>> list(frames_in_range(capture, 7, 9))
Traceback (most recent call last):
ValueError: no frames could be read from 7 to 9 (the video has 5 frames)
>>> capture.release()

```

### `frame_shape_of_capture(capture) -> (int, int, int)`
`(height, width, 3)` of an open `cv2.VideoCapture`, from its properties, without reading
a frame. With the five-frame video from the `frames_in_range` example:

```python
>>> from vision.players import frame_shape_of_capture
>>> capture = cv2.VideoCapture(str(video_path))
>>> frame_shape_of_capture(capture)
(48, 64, 3)
>>> capture.release()

```

### `warm_up_tracker(tracker, frame_shape, device=CPU_DEVICE) -> None`
Runs the detector once (`predict`, not `track`, so ByteTrack's state is untouched) on a
black frame of `frame_shape` at the same `imgsz` and `device` that `track_people` will
use. Called by `track_players` before the first chunk; see the note on the GPU's first
call above.

### `detections_of_result(result) -> FrameDetections`
Turns one Ultralytics `Results` object into `FrameDetections` (boxes, track ids,
confidences as NumPy arrays on the CPU); `empty_detections()` when it has no boxes or no
track ids.

```python
>>> from types import SimpleNamespace
>>> from vision.players import detections_of_result
>>> detections_of_result(SimpleNamespace(boxes=None)).track_ids
array([], dtype=int64)
>>> detections_of_result(SimpleNamespace(boxes=SimpleNamespace(id=None))).boxes_xyxy.shape
(0, 4)

```

### `track_people(tracker, frames, device=CPU_DEVICE) -> list[FrameDetections]`
Runs YOLO + ByteTrack on a list of frames in one call (persons only, NMS IoU 0.5,
`persist=True` so ids carry over between calls on the same tracker, inference at
`min(frame width, MAX_DETECTION_WIDTH_PX)` on `device`). Ultralytics runs the batch
through one detector pass and then updates the single tracker frame by frame in list
order, so the result is the same sequence of ids as one call per frame. Returns one
`FrameDetections` per input frame, in order (`empty_detections()` where nobody is
tracked), and `[]` for an empty list without calling the model. One frame:
`track_people(tracker, [frame], device)[0]`.

### `track_players(video_path, start_frame, end_frame, calibration_file=None, model_path=DEFAULT_MODEL_PATH, overlay_frames=(), checks_dir=CHECKS_DIR, device=None, batch_frames=DETECTION_BATCH_FRAMES, frame_stride=1, tracker=None) -> TrackingRun`
Tracks frames `start_frame` to `end_frame` (inclusive, stops early at the end of
the video; every `frame_stride`-th frame only): warms the model up, then reads
`batch_frames` sampled frames at a time on a background thread, runs the view check on
each, sends the main-view frames of the chunk to `track_people` in one call on `device`
(`default_device()` when `None`), then applies the on-court filter and labelling per
frame and saves overlays for the frames listed in `overlay_frames`. An already loaded
`tracker` (from `load_person_tracker`) is reused instead of loading `model_path`, so many
short ranges of one video can be tracked in one process; ByteTrack's state carries over
between calls, which only affects track ids (`me`/`opp` don't depend on them). Errors: `ValueError`
for a negative start, a reversed range, a `batch_frames` below 1, or a range with no
readable frame (see `frames_in_range`); `FileNotFoundError` for a missing video (checked
first after the range), calibration or weights file; `OSError` if the video can't be
opened.

### `read_stretches(stretches_file) -> list[(int, int)]`, `stretch_is_tracked(video_path, start_frame, end_frame, players_dir=PLAYERS_DIR) -> bool`, `track_stretches(video_path, stretches, calibration_file=None, model_path=DEFAULT_MODEL_PATH, device=None, batch_frames=DETECTION_BATCH_FRAMES, players_dir=PLAYERS_DIR) -> (tracked, skipped, frames)`
The `--stretches` mode. `track_stretches` loads the model once (only when a stretch is
left to track) and passes it to `track_players` for every stretch; ByteTrack's state
carries over from one stretch to the next, which only changes track ids.

```python
>>> import tempfile
>>> from vision.players import read_stretches, stretch_is_tracked
>>> folder = Path(tempfile.mkdtemp())
>>> _ = (folder / "s.csv").write_text("start_frame,end_frame\n0,99\n150,300\n")
>>> read_stretches(folder / "s.csv")
[(0, 99), (150, 300)]
>>> _ = (folder / "bad.csv").write_text("start,end\n0,99\n")
>>> read_stretches(folder / "bad.csv")  # doctest: +ELLIPSIS
Traceback (most recent call last):
ValueError: .../bad.csv needs start_frame and end_frame columns
>>> stretch_is_tracked(Path("data/never.mp4"), 0, 99, folder)
False

```

### `even_stretches(frame_count, stretch_frames, stretch_count=None) -> list[(int, int)]`, `write_stretches(stretches, stretches_file) -> Path`, `video_last_frame(video_path) -> int`
The `--plan-stretches` mode. Back-to-back stretches cover every frame, and the last one
is shorter. Spread stretches are centred in equal slices; they must fit, or it's a
`ValueError`. `video_last_frame` is OpenCV's frame count minus 1.

```python
>>> from vision.players import even_stretches
>>> even_stretches(1000, 300)
[(0, 299), (300, 599), (600, 899), (900, 999)]
>>> even_stretches(1000, 100, stretch_count=4)
[(75, 174), (325, 424), (575, 674), (825, 924)]
>>> even_stretches(1000, 300, stretch_count=4)
Traceback (most recent call last):
ValueError: 4 stretches of 300 frames don't fit in 1000 frames

```

### `merged_tables(tables) -> DataFrame`, `merge_stretch_tracks(video_path, stretches, last_frame, players_dir=PLAYERS_DIR) -> (Path, Path)`
The `--merge` step. `merged_tables` stacks the stretches' tables, keeps the first row of
a frame and player that two overlapping stretches both hold, and sorts by frame and
player. `merge_stretch_tracks` writes the merged positions, boxes and record for frames
0 to `last_frame`. A stretch that is not tracked yet is a `ValueError`, and so are
stretches whose tracking records disagree on `frame_stride` or `effective_fps`; nothing
is written then. A resume with another `--frame-stride` would otherwise mix rates in one
file. `--merge` without `--stretches` exits 2 with an error.

```python
>>> from vision.players import merged_tables
>>> first = pd.DataFrame({"frame": [0, 1, 1], "player": ["me", "me", "opp"], "x_m": [1.0, 1.1, 3.0]})
>>> second = pd.DataFrame({"frame": [1, 5], "player": ["me", "me"], "x_m": [9.9, 1.5]})
>>> merged_tables([second, first])[["frame", "player", "x_m"]].values.tolist()
[[0, 'me', 1.0], [1, 'me', 9.9], [1, 'opp', 3.0], [5, 'me', 1.5]]
>>> merged_tables([]).empty
True

```

### `split_frames_by_view(video_path, start_frame, end_frame, calibration_file=None) -> (list[int], list[int])`
Only the view check: main-view frame indices and skipped frame indices. Same
errors as `track_players` (no weights needed).

### `positions_path(...)`, `boxes_path(video_path, start_frame, end_frame, players_dir=PLAYERS_DIR) -> Path`

```python
>>> from vision.players import boxes_path, positions_path
>>> positions_path(Path("data/match.mp4"), 14201, 14947).as_posix()
'data/players/match/positions_14201_14947.csv'
>>> boxes_path(Path("data/match.mp4"), 14201, 14947).as_posix()
'data/players/match/boxes_14201_14947.csv'

```

### `tracking_record_path(positions_file) -> Path`, `tracking_record(video_fps, frame_width, frame_stride, model_path, device, batch_frames) -> dict`, `write_tracking_record(positions_file, record) -> Path`, `read_tracking_record(positions_file) -> dict`, `frame_stride_of(positions_file) -> int`, `effective_fps(video_fps, frame_stride) -> float`
The tracking record next to a positions file. `read_tracking_record` gives
`{"frame_stride": 1}` when there is none, so tracks written before the record existed
count as stride 1; `effective_fps` is `video_fps / frame_stride`.

```python
>>> from vision.players import effective_fps, frame_stride_of, read_tracking_record, tracking_record, tracking_record_path, write_tracking_record
>>> positions_file = Path(tempfile.mkdtemp()) / "positions_0_99.csv"
>>> tracking_record_path(positions_file).name
'positions_0_99.csv.tracking.json'
>>> frame_stride_of(positions_file), read_tracking_record(positions_file)
(1, {'frame_stride': 1})
>>> record = tracking_record(60.0, 1920, 2, Path("data/models/yolo11n.pt"), "mps", 8)
>>> record["effective_fps"], record["model"], record["frame_stride"]
(30.0, 'yolo11n.pt', 2)
>>> _ = write_tracking_record(positions_file, record)
>>> frame_stride_of(positions_file), effective_fps(60.0, frame_stride_of(positions_file))
(2, 30.0)

```

### `video_fps_and_width(video_path) -> (float, float)`
The video's frame rate and width from its properties (also exported by `vision.hits`).

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
