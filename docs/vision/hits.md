# vision/hits.py

Hit detection and rally segmentation from the shuttle track (`vision/shuttle.py`) and
the player boxes (`vision/players.py`) of a frame range. A hit is a sharp change of
the shuttle's velocity while the shuttle is within reach of a player; the hitter is
the player whose box is nearest to the shuttle; rallies are runs of hits with no long
gap and no resting shuttle between them, and each rally ends where the shuttle comes
to rest (or, if it never does, at the last frame it was tracked).

All positions are video pixels. The shuttle flies above the floor, so the court
homography can't place it in meters; the players' boxes are in the same pixel space
and their height (roughly the player's standing height) gives a scale that makes the
near and the far player comparable. So:

- **Hit score** at frame `t`: with `v_before` the shuttle's mean velocity over the 3
  frames before `t` and `v_after` over the 3 frames after,
  `score = |v_after - v_before| / (|v_before| + |v_after| + 2)`. Straight flight gives
  about 0 (velocity changes only by gravity and drag); a reversal at speed gives
  about 1. The `+ 2` px/frame stops a nearly still shuttle from scoring high.
- **Reach** of a player: the distance from the shuttle to the player's box (0 inside
  the box) divided by the box height, in "body heights". Arm plus racket is about
  0.75 body heights, so a shuttle further than that from both players is not at a hit.
- **Hitter**: the player with the smaller reach at the hit frame. At the net the two
  boxes overlap in the image and this is ambiguous; a rule that uses which player the
  shuttle approached before the hit was tried and gave the same accuracy with
  different errors, so the simpler rule stays (see `docs/results/hits.md`).
- **Candidates** are frames with `score >= 0.4`, `|v_after| >= 8` px/frame (the shuttle
  leaves the racket at speed; a shuttle tumbling on the floor or carried by a player
  doesn't) and reach `<= 0.75` for at least one player. Within 8 frames of a stronger
  candidate, weaker ones are dropped (two labelled hits in one match of ShuttleSet are
  closer than that in only a few percent of cases).
- **Rest**: 20 consecutive tracked frames whose positions stay within 8 px of the
  first. A lob's apex is slower than this (gravity alone moves the shuttle about 19
  px over 20 frames at far-court scale); a shuttle lying on the floor is not.

The detection functions (`window_velocities`, `hit_scores`, `reach_distances`,
`detect_hits`, `is_at_rest`, `shuttle_rest_frame`, `rally_numbers`,
`segment_rallies`, `match_hits`, ...) are pure functions on arrays and data frames;
only `load_frame_tracks` and `main` read files, and `load_frame_tracks` runs the
view check on the video to drop broadcast cutaways.

## Command

```
.venv/bin/python -m vision.hits <video> --start N --end N [--evaluate MATCH_FOLDER SET_NO]
```

Needs the tracks of exactly that range from the two tracking commands:
`python -m vision.players <video> --start N --end N` (writes `positions_*.csv` and
`boxes_*.csv`) and `python -m vision.shuttle <video> --start N --end N`. The video
and its calibration are needed too (view check; `--evaluate` maps the labelled
hitter's position to a side).

| Argument | Meaning |
|---|---|
| `--start`, `--end` | the tracked range (inclusive) |
| `--evaluate` | compare with ShuttleSet's hit frames for this match folder and set: precision and recall within ±5 frames, hitter accuracy, frame offset |

Output under `data/hits/<video stem>/`:
- `hits_<start>_<end>.csv` (`HIT_COLUMNS`): `rally` (1-based within the range),
  `frame`, `hitter` (`me` near side, `opp` far side), `score`, `reach` (of the hitter,
  body heights).
- `rallies_<start>_<end>.csv` (`RALLY_COLUMNS`): `rally`, `start_frame` (first hit),
  `end_frame` (rest frame, else last tracked frame before the next rally), `hits`.

Printed: `rallies: N, hits: N`, with `--evaluate` four more lines (see
`evaluation_lines`), one line per miss and false positive with its cause (see
`error_breakdown`) and the count of each kind on InpaintNet-filled frames, then the
saved paths. Exit codes: `0` on success; `1` with
`error: ...` for a missing track file (the message names the command to run), video,
calibration or set CSV, or a bad range; `2` for invalid arguments.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `VELOCITY_WINDOW_FRAMES` | `3` | frames on each side of `t` for `v_before` and `v_after` |
| `SPEED_FLOOR_PX_PER_FRAME` | `2.0` | added to the score's denominator |
| `HIT_SCORE_MIN` | `0.4` | minimum score of a candidate |
| `SPEED_AFTER_MIN_PX_PER_FRAME` | `8.0` | minimum shuttle speed after a candidate |
| `REACH_MAX_BODY_HEIGHTS` | `0.75` | maximum reach of the nearer player at a candidate |
| `MIN_HIT_GAP_FRAMES` | `8` | minimum frames between two hits |
| `RALLY_GAP_FRAMES` | `120` | hits further apart than this (4 s at 30 fps) are in different rallies |
| `REST_RADIUS_PX`, `REST_FRAME_COUNT` | `8.0`, `20` | the rest definition |
| `MATCH_TOLERANCE_FRAMES` | `5` | a detection within this many frames of a labelled hit counts as that hit |
| `HITS_DIR` | `Path("data/hits")` | output folder |
| `HIT_COLUMNS`, `RALLY_COLUMNS` | see above | output columns |

The thresholds were chosen on two sets of one broadcast match and checked on the
third (`docs/results/hits.md`); `HitDetectionSettings` overrides them per call.

```python
>>> from vision.hits import HIT_COLUMNS, HIT_SCORE_MIN, RALLY_COLUMNS, REACH_MAX_BODY_HEIGHTS, SPEED_AFTER_MIN_PX_PER_FRAME
>>> HIT_SCORE_MIN, SPEED_AFTER_MIN_PX_PER_FRAME, REACH_MAX_BODY_HEIGHTS
(0.4, 8.0, 0.75)
>>> HIT_COLUMNS, RALLY_COLUMNS
(['rally', 'frame', 'hitter', 'score', 'reach'], ['rally', 'start_frame', 'end_frame', 'hits'])

```

## Classes

### `HitDetectionSettings(velocity_window_frames=3, speed_floor_px_per_frame=2.0, hit_score_min=0.4, speed_after_min_px_per_frame=8.0, reach_max_body_heights=0.75, min_hit_gap_frames=8)`
The thresholds of `detect_hits`, as a frozen dataclass with the constants as defaults.

### `FrameTracks(frames, shuttle_points, shuttle_valid, near_boxes, far_boxes, main_view=None, shuttle_inpainted=None)`
One frame range's tracks, all arrays of the same length `N`: `frames` (`int`, video
frame indices, increasing), `shuttle_points` (`(N, 2)` pixels), `shuttle_valid`
(`bool`: shuttle visible and the frame is the main camera view), `near_boxes` and
`far_boxes` (`(N, 4)` `left, top, right, bottom` pixels of `me` and `opp`; `NaN` where
the player wasn't found). The two optional `bool` arrays, `main_view` and
`shuttle_inpainted` (the shuttle position came from InpaintNet), are only used to
explain errors in `error_breakdown`; `frame_tracks` fills them in.

The examples use a synthetic rally: the shuttle flies from the near player's box to
the far player's box in 20 frames, is hit there, and flies back.

```python
>>> import numpy as np
>>> import pandas as pd
>>> from vision.hits import FrameTracks
>>> toward_far = np.column_stack([150 + 19 * np.arange(21), 450 - 9.5 * np.arange(21)])
>>> back = np.column_stack([530 - 19 * np.arange(1, 20), 260 + 9.5 * np.arange(1, 20)])
>>> points = np.vstack([toward_far, back])
>>> near_boxes = np.tile([100.0, 400.0, 180.0, 560.0], (40, 1))
>>> far_boxes = np.tile([500.0, 200.0, 560.0, 320.0], (40, 1))
>>> tracks = FrameTracks(np.arange(40), points, np.ones(40, dtype=bool), near_boxes, far_boxes)

```

## Detection

### `window_velocities(points, valid, window) -> (before, after, defined)`
Mean velocity (px/frame, `(N, 2)`) over the `window` frames before and after each
frame; `defined` (`(N,)` bool) is `True` only where the frame and both window ends
are valid, so the first and last `window` frames are never defined.

```python
>>> from vision.hits import window_velocities
>>> before, after, defined = window_velocities(points, np.ones(40, dtype=bool), 3)
>>> before[20].tolist(), after[20].tolist()
([19.0, -9.5], [-19.0, 9.5])
>>> defined[:3].tolist(), bool(defined[20])
([False, False, False], True)

```

### `hit_scores(before, after, defined, speed_floor) -> ndarray`
The hit score per frame (0 where not defined).

```python
>>> from vision.hits import hit_scores
>>> scores = hit_scores(before, after, defined, 2.0)
>>> scores[17:24].round(3).tolist()
[0.0, 0.467, 0.934, 0.955, 0.934, 0.467, 0.0]

```

The score is spread over the window on both sides of the hit; `detect_hits` keeps
the peak.

### `distance_to_box(point, box) -> float`
Pixel distance from a point to a box's edge, 0 inside.

```python
>>> from vision.hits import distance_to_box
>>> distance_to_box((190, 480), (100, 400, 180, 560)), distance_to_box((150, 450), (100, 400, 180, 560))
(10.0, 0.0)
>>> round(distance_to_box((50, 380), (100, 400, 180, 560)), 2)
53.85

```

### `reach_distances(points, boxes) -> ndarray`
`distance_to_box` divided by the box height, per frame; `inf` where the box is
missing (`NaN`) or has no height.

```python
>>> from vision.hits import reach_distances
>>> reach_distances(points, near_boxes)[[0, 20]].round(3).tolist()
[0.0, 2.356]
>>> reach_distances(points, far_boxes)[[0, 20]].round(3).tolist()
[3.111, 0.0]
>>> reach_distances([[1.0, 1.0]], [[np.nan, np.nan, np.nan, np.nan]]).tolist()
[inf]

```

### `peaks_with_minimum_gap(candidate_indices, scores, min_gap) -> list[int]`
Keeps candidates in order of decreasing score, dropping any within `min_gap` of one
already kept; returns the kept indices sorted.

```python
>>> from vision.hits import peaks_with_minimum_gap
>>> peaks_with_minimum_gap([10, 12, 30], np.array([0.0] * 10 + [0.9, 0.0, 0.8] + [0.0] * 17 + [0.7]), 8)
[10, 30]

```

### `nearest_player(near_reach_now, far_reach_now) -> str`
`"me"` if the near player's reach is smaller or equal, else `"opp"`.

### `detect_hits(tracks, settings=HitDetectionSettings()) -> DataFrame`
Columns `frame`, `hitter`, `score`, `reach` (the hitter's), one row per hit, in frame
order.

```python
>>> from vision.hits import detect_hits
>>> detect_hits(tracks)
   frame hitter  score  reach
0     20    opp  0.955    0.0

```

## Rallies

### `is_at_rest(points, valid, rest_radius=8.0) -> bool`
`True` when all points are valid and within `rest_radius` of the first.

```python
>>> from vision.hits import is_at_rest
>>> is_at_rest([[300, 400], [305, 403], [299, 402]], [True, True, True]), is_at_rest([[300, 400], [320, 400]], [True, True])
(True, False)

```

### `shuttle_rest_frame(tracks, from_frame, to_frame, rest_radius=8.0, rest_frame_count=20) -> int | None`
The first frame after `from_frame` (up to `to_frame`) that starts a resting window.

```python
>>> from vision.hits import shuttle_rest_frame
>>> resting = np.tile([300.0, 400.0], (30, 1)) + (np.arange(30) % 3)[:, None]
>>> rest_tracks = FrameTracks(np.arange(60), np.vstack([points[:30], resting]), np.ones(60, dtype=bool),
...                           np.tile(near_boxes[0], (60, 1)), np.tile(far_boxes[0], (60, 1)))
>>> shuttle_rest_frame(rest_tracks, 0, 59), shuttle_rest_frame(tracks, 0, 39)
(30, None)

```

### `last_valid_frame_before(tracks, from_frame, to_frame) -> int`
The last frame in `[from_frame, to_frame]` where the shuttle is valid; `from_frame`
if there is none.

```python
>>> from vision.hits import last_valid_frame_before
>>> partly_lost = FrameTracks(np.arange(10), np.zeros((10, 2)), np.array([True] * 6 + [False] * 4),
...                           np.zeros((10, 4)), np.zeros((10, 4)))
>>> last_valid_frame_before(partly_lost, 2, 9)
5

```

### `rally_numbers(hit_frames, tracks, rally_gap_frames=120) -> list[int]`
A 1-based rally number per hit: a new rally starts after a gap longer than
`rally_gap_frames` or when the shuttle rested between two hits.

```python
>>> from vision.hits import rally_numbers
>>> rally_numbers([100, 130, 160, 400, 430], tracks, 120)
[1, 1, 1, 2, 2]

```

### `rally_end_frame(last_hit_frame, next_rally_start, tracks) -> int`
The rest frame after the last hit and before the next rally, else the last valid
frame before it. A shuttle that leaves the top of the picture and comes back is not
an end.

### `segment_rallies(hits, tracks, rally_gap_frames=120) -> (hits with rally, rallies)`
Adds `rally` to the hits (`HIT_COLUMNS`) and returns the rally table (`RALLY_COLUMNS`).

```python
>>> from vision.hits import segment_rallies
>>> numbered, rallies = segment_rallies(detect_hits(tracks), tracks)
>>> rallies
   rally  start_frame  end_frame  hits
0      1           20         39     1

```

Here the shuttle never rests and the range ends while it is still tracked, so the
rally ends at the last frame.

## Evaluation against ShuttleSet

### `match_hits(detected_frames, labelled_frames, tolerance=5) -> list[(int, int)]`
One-to-one matching, closest pairs first, within `tolerance` frames; sorted by the
labelled frame.

```python
>>> from vision.hits import match_hits, precision_recall
>>> match_hits([100, 131, 200], [103, 128, 300])
[(100, 103), (131, 128)]
>>> precision_recall(2, 3, 3)
(0.6666666666666666, 0.6666666666666666)

```

### `precision_recall(matched_count, detected_count, labelled_count) -> (float, float)`
0 when the respective count is 0.

### `side_of_hitter(homography, hitter_pixel) -> str`
`"me"` if the hitter's foot pixel maps to the near half of the court, else `"opp"`.

```python
>>> from vision.hits import side_of_hitter
>>> homography = np.array([[0.0253, 0.00874, -13.09], [-0.000222, -0.0704, 46.5], [8.6e-6, 0.00286, 1.0]])
>>> side_of_hitter(homography, (640, 500)), side_of_hitter(homography, (640, 350))
('me', 'opp')

```

### `labelled_hits(shots, homography, start_frame, end_frame, label_scale=1.0) -> DataFrame`
ShuttleSet's hits in the range: `rally`, `frame`, `hitter` (from
`player_location_x/y`, scaled by `label_scale` if the video isn't 1280 wide; `None`
when the location is missing, as for some serves), `type`.

```python
>>> from vision.hits import labelled_hits
>>> shots = pd.DataFrame({"rally": [1, 1], "frame_num": [100.0, 125.0], "player_location_x": [640.0, 640.0],
...                       "player_location_y": [500.0, 350.0], "type": ["lob", "drop"]})
>>> labelled_hits(shots, homography, 90, 130)
   rally  frame hitter  type
0      1    100     me   lob
1      1    125    opp  drop

```

### `evaluation_lines(hits, labels, tolerance=5) -> list[str]`
The four printed lines: counts and matches, precision and recall, hitter accuracy
over matched hits whose label has a hitter (a matched hit with a missing labelled
hitter, as for most serves, is not judged), and the mean and median of detected
minus labelled frame.

```python
>>> from vision.hits import evaluation_lines
>>> detected = pd.DataFrame({"frame": [20, 101], "hitter": ["opp", "me"], "score": 0.9, "reach": 0.0})
>>> labels = pd.DataFrame({"rally": [1, 1], "frame": [21, 100], "hitter": ["opp", None], "type": ["lob", "serve"]})
>>> for line in evaluation_lines(detected, labels):
...     print(line)
labelled hits: 2, detected: 2, matched within ±5 frames: 2
precision 100.0%, recall 100.0%
hitter correct: 1/1 = 100.0%
frame offset detected - labelled: mean +0.00, median +0.0

```

### `miss_cause(label_frame, is_first_in_rally, tracks, settings, tolerance) -> str`, `false_positive_cause(frame, labelled_rally_spans, tracks) -> str`, `is_inpainted_at(tracks, frame) -> bool`
The cause of one miss: in order, `frame not tracked`, `not main view`, `velocity
window undefined` (no frame within `tolerance` has one), `score below threshold`,
`speed after the hit below threshold`, `reach above limit` (no frame within
`tolerance` passes that test, given the earlier ones), else `suppressed by a stronger
candidate within the minimum gap`; prefixed with `serve: ` for the first hit of a
labelled rally. The cause of one false positive: `before the first labelled hit`,
`after the last hit of a rally` (between a rally's last labelled hit and the next
rally's first), `within a rally, on an InpaintNet-filled frame`, else `within a
rally, no labelled hit nearby`.

### `error_breakdown(hits, labels, tracks, settings=HitDetectionSettings(), tolerance=5) -> DataFrame`
One row per unmatched labelled hit (`miss`) and unmatched detection
(`false positive`): `kind`, `frame`, `cause`, `inpainted`.

```python
>>> from vision.hits import error_breakdown
>>> detections = pd.DataFrame({"frame": [20, 36], "hitter": ["opp", "me"], "score": 0.9, "reach": 0.0})
>>> labels = pd.DataFrame({"rally": [1, 1], "frame": [5, 20], "hitter": [None, "opp"], "type": ["serve", "lob"]})
>>> error_breakdown(detections, labels, tracks)
             kind  frame                          cause  inpainted
0  false positive     36  after the last hit of a rally      False
1            miss      5   serve: score below threshold      False

```

## Files

### `boxes_by_frame(boxes, frames, player) -> ndarray`, `frame_tracks(shuttle, boxes, main_view_frames) -> FrameTracks`
Build `FrameTracks` from the shuttle positions table (`vision/shuttle.py` columns),
the boxes table (`vision/players.py` `BOX_COLUMNS`) and the main-view frame indices;
a frame outside the main view or with the shuttle not visible is invalid.

```python
>>> from vision.hits import frame_tracks
>>> shuttle = pd.DataFrame({"frame": [10, 11, 12], "x": [100, 110, 0], "y": [50, 55, 0], "visibility": [1, 1, 0], "inpainted": 0})
>>> boxes = pd.DataFrame({"frame": [10, 11, 10], "player": ["me", "me", "opp"],
...                       "left": [0.0, 1.0, 50.0], "top": [0.0, 1.0, 50.0], "right": [10.0, 11.0, 60.0], "bottom": [20.0, 21.0, 70.0]})
>>> built = frame_tracks(shuttle, boxes, [10, 12])
>>> built.shuttle_valid.tolist(), built.near_boxes[1].tolist(), built.far_boxes[0].tolist()
([True, False, False], [1.0, 1.0, 11.0, 21.0], [50.0, 50.0, 60.0, 70.0])

```

### `require_track_file(path, command) -> Path`, `load_frame_tracks(video_path, start_frame, end_frame) -> FrameTracks`
Read the three track files of the range (error names the command that writes a
missing one) and run the view check on the video.

```python
>>> from pathlib import Path
>>> from vision.hits import require_track_file
>>> require_track_file(Path("data/shuttle/none/shuttle_1_2.csv"), "python -m vision.shuttle none.mp4 --start 1 --end 2")
Traceback (most recent call last):
FileNotFoundError: track file not found: data/shuttle/none/shuttle_1_2.csv (run: python -m vision.shuttle none.mp4 --start 1 --end 2)

```

### `output_paths(video_path, start_frame, end_frame, hits_dir=HITS_DIR) -> (Path, Path)`

```python
>>> from vision.hits import output_paths
>>> [path.as_posix() for path in output_paths(Path("data/match.mp4"), 14201, 14947)]
['data/hits/match/hits_14201_14947.csv', 'data/hits/match/rallies_14201_14947.csv']

```

### `main(arguments) -> int`
Command entry point: loads the tracks, detects and segments, writes both files,
prints the summary (and the evaluation) and returns the exit code.
