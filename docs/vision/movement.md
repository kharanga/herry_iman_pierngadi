# vision/movement.py

Hit frames, hitters and rallies from the two players' movement alone: it reads the
player tracking files (`positions_*.csv` and `boxes_*.csv`) and nothing else, no shuttle
track and no video frames except for the optional check sheet. Output files and columns
are the same as `vision/hits.py` writes (`hits_<start>_<end>.csv`,
`rallies_<start>_<end>.csv`), so `vision.features` and `scout.run` read them unchanged;
`reach` is NaN and `score` is the movement score.

## How it works

1. **Per-frame motion features** for each player (`motion_features`): court speed and
   acceleration from the smoothed foot position, a direction-change term (the dot product
   of the heading a few frames before and after), the rise and fall of the box height and
   width (an overhead raises the box top, a lunge widens it), the court `y` and its
   change, and whether the player was detected. The same six features of the other
   player are appended (`hit_feature_matrix`), because the opponent reacts to a hit.
2. **A per-frame hit score** for each player from a small gradient-boosted classifier
   (`fit_hit_model`, `hit_scores`): the probability that this player hits within
   ±2 frames of this frame. It is trained on the labelled ShuttleSet ranges of
   `data/match.mp4` (`--train`) and stored under `data/models/` like the shot-type
   classifier; the module holds no weights.
3. **Rally rhythm** (`alternating_hit_chain`): local maxima of the scores are candidates;
   a dynamic programme picks the chain of candidates with the largest total
   `score - hit_score_min` such that hitters alternate, consecutive hits are between
   `min_hit_gap_frames` and `max_hit_gap_frames` apart (a gap outside the usual
   15-55 frames at 30 fps costs `unusual_gap_penalty_per_10_frames`), and a gap longer
   than `max_hit_gap_frames` starts a new rally.
4. **Rallies** (`split_into_rallies`, `trim_rally_ends`, `rally_end_frame`): the chain is
   cut at long gaps; weak hits at either end of a rally (score below
   `rally_end_score_min`) are dropped, as are rallies left with one hit; a rally ends at
   the first frame after its last hit where both players stay below the walking speed
   for `still_frames`, else `max_hit_gap_frames` after the last hit (or at the next
   rally's start or the range end).
5. **Inferred serves** (`with_inferred_serves`): the detector never sees a serve (the
   server flicks from standing, the receiver waits), so without help a rally's first
   row would be the return at `ball_round` 1, which the shot-type classifier reads as a
   serve. Each rally therefore gets a row for the other player than its first hitter,
   `serve_to_return_seconds` (0.76 s, ShuttleSet's median from serve to return) before
   the first detected hit, with `score` NaN so it is visibly inferred; the detected hits
   start at `ball_round` 2 in `vision.features`. `--no-serves` turns it off.

All frame counts and the gap penalty scale with the video's fps and pixel features with
its width (`settings_for_video`), as in `vision/hits.py`. The detector functions are
pure: arrays and tables in, tables out. Only `load_player_tracks`, `load_hit_model`,
`save_hit_model`, `labelled_range_data`, `write_spot_check` and `main` touch files.

## Commands

```
.venv/bin/python -m vision.movement <video> --start N --end N [--evaluate MATCH_FOLDER SET_NO] [--spot-check N]
.venv/bin/python -m vision.movement <video> --train --match MATCH_FOLDER --range START END SET_NO [...] [--model FILE]
.venv/bin/python -m vision.movement <video> --cross-validate --match MATCH_FOLDER --range START END SET_NO [...]
```

| Argument | Meaning |
|---|---|
| `--start`, `--end` | the tracked range (inclusive); needs `python -m vision.players <video> --start N --end N` to have run |
| `--model` | the hit model file (default `data/models/movement_hits.joblib`) |
| `--evaluate` | compare with ShuttleSet's hit frames for this match folder and set, within ±0.25 s and ±0.50 s (`vision.hits.MATCH_TOLERANCE_SECONDS`), as frames at the video's own fps, each line headed "±0.25 s (±N frames at F fps)" (`--tolerance S`, repeatable, changes the windows); then one line per miss (with the labelled shot type) and per false positive (inside or outside the labelled rallies) |
| `--evaluate-labels CSV` | the same comparison against hand labels from `python -m vision.label` (`rally, frame_num, player, type_group, source`; rows in the range); with `--database` and `--match-id` also the shot-type agreement: among matched detected hits that have a `type_pred` in the own rows of that match and a labelled group other than `unknown`, how many agree |
| `--tolerance S` | a matching window in seconds for `--evaluate`, `--evaluate-labels` and `--cross-validate`, turned into frames at the video's fps; repeatable (default 0.25 and 0.5) |
| `--hits-dir` | root of the output files (default `data/hits`) |
| `--players-dir` | root of the player tracking files (default `data/players`); the tracking record next to the positions file gives the frame stride, and the settings are scaled to the effective rate `video fps / stride` |
| `--spot-check N` | draw N random detected hits (seed 1; inferred serves excluded), each as the frame 6 frames before (12 at 60 fps), the hit frame and 6 after, cropped around the hitter's box, both boxes drawn (`me` blue, `opp` red), to `data/checks/hits/<video id>_movement_spot_check_<start>_<end>.jpg` |
| `--no-serves` | don't add an inferred serve row to each rally |
| `--in-play-model MODEL` | rallies are the in-play stretches this model finds (`python -m vision.in_play`, `docs/vision/in_play.md`) instead of the gap rule. Hits are the hit chain inside each stretch. The inferred serve sits at the stretch's start unless the first detected hit is within 0.3 s of it. Off by default |
| `--train` | fit the model on the `--range` ranges (labelled through `--match`) and save it to `--model` |
| `--cross-validate` | hold out each `--range` in turn, train on the others, and print precision, recall and hitter accuracy per range and in total |
| `--range START END SET_NO`, `--match` | a labelled range of `<video>` and the ShuttleSet match folder; the range needs its player tracking files |

Output under `data/hits/<video id>/` (`--hits-dir`): `hits_<start>_<end>.csv`
(`HIT_COLUMNS`: `rally`, `frame`, `hitter`, `score` (NaN on an inferred serve), `reach`
= NaN) and `rallies_<start>_<end>.csv` (`RALLY_COLUMNS`: `rally`, `start_frame` (the
serve row when there is one), `end_frame`, `hits` (rows, serve included)). Printed:
`rallies: N, hits: N, inferred serves: N`, the evaluation lines (detected hits only,
then one line per tolerance on the inferred serves, `serve_evaluation_line`), then the
saved paths. Exit
codes: `0`; `1` with `error: ...` for a missing video, tracking file (the message names
the command), model, calibration or set CSV, a missing `--start`/`--end`, or `--train`
without `--match`/`--range`; `2` for invalid arguments.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `MODEL_PATH` | `data/models/movement_hits.joblib` | the shipped hit model (gitignored, rebuilt with `--train`) |
| `SERVE_TO_RETURN_SECONDS` | `0.76` | median serve-to-return time in ShuttleSet (3,502 rallies; quartiles 0.64 and 1.00 s), see `docs/results/no_shuttle.md` |
| `SPEED_CAP_M_PER_S` | `12.0` | speeds above this are tracking jumps and are clipped |
| `SELF_FEATURE_NAMES`, `OTHER_FEATURE_NAMES`, `FEATURE_NAMES` | 14, 6, 20 names | the feature columns; the model file stores them and `load_hit_model` refuses a mismatch |
| `HIT_MODEL_PARAMETERS` | depth 4, rate 0.05, 200 trees, balanced classes, seed 0 | `HistGradientBoostingClassifier` settings |
| `SPOT_CHECK_DIR`, `SPOT_CHECK_SEED`, `SPOT_CHECK_OFFSET_FRAMES` | `data/checks/hits`, `1`, `6` | the check sheet |
| `TILE_WIDTH`, `TILE_HEIGHT` | `480`, `360` | one tile of the sheet |

```python
>>> from vision.movement import FEATURE_NAMES, SELF_FEATURE_NAMES, OTHER_FEATURE_NAMES
>>> len(FEATURE_NAMES), FEATURE_NAMES[0], FEATURE_NAMES[-1]
(20, 'self_speed', 'other_y_move')
>>> len(SELF_FEATURE_NAMES), len(OTHER_FEATURE_NAMES)
(14, 6)

```

## Classes

### `MovementSettings(...)`
Frozen dataclass of every tuned number, at 30 fps and 1280 px unless scaled:
`position_smoothing_frames=5`, `speed_smoothing_frames=3`, `direction_smoothing_frames=7`,
`context_frames=5` (the "before" and "after" offset of the features),
`score_smoothing_frames=3`, `interpolation_limit_frames=10` (longer gaps in a track are
held at the last position), `label_tolerance_frames=2` (training target width),
`hit_score_min=0.5`, `rally_end_score_min=0.7`, `min_hit_gap_frames=10`,
`max_hit_gap_frames=70`, `usual_gap_frames=(15, 55)`,
`unusual_gap_penalty_per_10_frames=0.3`, `still_speed_m_per_s=0.8`, `still_frames=15`,
`infer_serves=True`, `serve_to_return_seconds=0.76`, `fps=30.0`, `pixel_scale=1.0`.

### `settings_for_video(fps, frame_width, base=MovementSettings()) -> MovementSettings`
Scales every frame count by `fps / 30` (`vision.hits.frames_for_fps`, at least 1), the
gap penalty per 10 frames by `30 / fps`, and sets `pixel_scale = 1280 / frame_width`
(box sizes are measured as if the frame were 1280 wide). At 30 fps and 1280 px the
result equals the defaults.

```python
>>> from vision.movement import MovementSettings, settings_for_video
>>> settings_for_video(30.0, 1280) == MovementSettings()
True
>>> scaled = settings_for_video(60.0, 1920)
>>> (scaled.context_frames, scaled.min_hit_gap_frames, scaled.max_hit_gap_frames, scaled.usual_gap_frames, scaled.still_frames)
(10, 20, 140, (30, 110), 30)
>>> round(scaled.unusual_gap_penalty_per_10_frames, 3), round(scaled.pixel_scale, 4), scaled.fps
(0.15, 0.6667, 60.0)

```

### `PlayerTracks(frames, near_points, far_points, near_boxes, far_boxes)`
One row per frame of the range: court positions (metres, NaN where the player wasn't
found) and pixel boxes of `me` (near) and `opp` (far). `points_of(player)` and
`boxes_of(player)` pick a side.

### `HitCandidate(frame, hitter, score)`
A local maximum of one player's hit score.

### `LabelledRange(video_path, start_frame, end_frame, set_no, match_folder, players_dir=PLAYERS_DIR)`
A tracked range with ShuttleSet labels, for training and cross-validation; `name` is
`"<video id> <start>-<end> set <n>"`.

## Tracks and features

### `rows_by_frame(table, frames, player, columns) -> ndarray`, `player_tracks(positions, boxes, start_frame, end_frame, frame_stride=1) -> PlayerTracks`
Reindex the player files onto every sampled frame of the range (`start_frame`,
`start_frame + stride`, ...; NaN rows where a player has no row). With a stride the
rows are one effective frame apart, so the features treat them as a video at
`fps / stride` (`settings_for_tracks`).

```python
>>> import numpy as np, pandas as pd
>>> from vision.movement import player_tracks
>>> positions = pd.DataFrame({"frame": [10, 11, 10], "player": ["me", "me", "opp"], "x_m": [3.0, 3.1, 2.0], "y_m": [2.0, 2.2, 10.0]})
>>> boxes = pd.DataFrame({"frame": [10, 11, 10], "player": ["me", "me", "opp"], "left": [600, 610, 620], "top": [300, 300, 200], "right": [700, 710, 660], "bottom": [500, 505, 300]})
>>> tracks = player_tracks(positions, boxes, 10, 12)
>>> tracks.frames
array([10, 11, 12])
>>> tracks.near_points
array([[3. , 2. ],
       [3.1, 2.2],
       [nan, nan]])
>>> tracks.far_boxes[0], np.isnan(tracks.far_boxes[1]).all()
(array([620., 200., 660., 300.]), np.True_)
>>> tracks.points_of("opp") is tracks.far_points
True
>>> player_tracks(positions, boxes, 10, 12, frame_stride=2).frames
array([10, 12])

```

### `filled(values, interpolation_limit) -> ndarray`, `moving_average(values, window) -> ndarray`, `shifted(values, offset) -> ndarray`
Helpers of the feature step: `filled` fills the first `interpolation_limit` frames of
a gap by linear interpolation towards the next known value and the rest of a longer
gap with that next known value (a jump, not a hold; leading NaNs take the first known
value, trailing ones the last; all zeros for an all-NaN track); `moving_average` is a centred
box filter that holds the end values beyond the ends (the input unchanged for a window
of 1); `shifted(v, k)` is `v[i + k]`
clamped at the ends, so a positive offset looks ahead.

```python
>>> from vision.movement import filled, moving_average, shifted
>>> filled([1.0, np.nan, 3.0, np.nan, np.nan, np.nan], interpolation_limit=1)
array([1., 2., 3., 3., 3., 3.])
>>> filled([np.nan, 1.0, np.nan, np.nan, np.nan, 5.0], interpolation_limit=1)
array([1., 1., 2., 5., 5., 5.])
>>> moving_average([0, 0, 3, 0, 0], 3)
array([0., 1., 1., 1., 0.])
>>> moving_average([6, 0, 0], 3)
array([4., 2., 0.])
>>> shifted(np.array([1, 2, 3, 4]), 2), shifted(np.array([1, 2, 3, 4]), -1)
(array([3, 4, 4, 4]), array([1, 1, 2, 3]))

```

### `motion_features(points, boxes, settings) -> dict[str, ndarray]`
The 14 `SELF_FEATURE_NAMES` arrays for one player (speed in m/s, acceleration in m/s²,
box changes in pixels at 1280 width). A track of fewer than 2 frames gives zeros.

```python
>>> from vision.movement import motion_features
>>> frames = 40
>>> x = np.full(frames, 3.0)
>>> y = np.concatenate([np.linspace(1.0, 3.0, 20), np.linspace(3.0, 1.0, 20)])
>>> boxes = np.tile([600.0, 300.0, 700.0, 500.0], (frames, 1))
>>> features = motion_features(np.column_stack([x, y]), boxes, MovementSettings())
>>> sorted(features) == sorted(SELF_FEATURE_NAMES)
True
>>> round(float(features["speed"][10]), 2)
3.16
>>> bool(features["direction_change"][20] < 0 < features["direction_change"][10])
True
>>> features["detected"].sum(), features["box_height_rise"].max()
(np.float64(40.0), np.float64(0.0))

```

The player runs up the court at 3.16 m/s (2 m in 20 frames at 30 fps) and turns at
frame 20, where the direction change is negative.

### `other_player(player) -> str`, `hit_feature_matrix(tracks, player, settings) -> ndarray`
The `(frames, 20)` matrix for one player: its own 14 features then the other player's 6
(`OTHER_FEATURE_NAMES`), in `FEATURE_NAMES` order.

```python
>>> from vision.movement import hit_feature_matrix, other_player
>>> other_player("me"), other_player("opp")
('opp', 'me')
>>> hit_feature_matrix(tracks, "me", MovementSettings()).shape
(3, 20)

```

### `hit_frame_labels(frames, hit_frames, tolerance) -> ndarray`
Training target: 1 within `tolerance` frames of a labelled hit of this player, else 0.

```python
>>> from vision.movement import hit_frame_labels
>>> hit_frame_labels(np.arange(100, 110), [104], tolerance=2)
array([0., 0., 1., 1., 1., 1., 1., 0., 0., 0.])

```

## Scores and the hit chain

### `fit_hit_model(feature_matrix, labels) -> HistGradientBoostingClassifier`, `hit_scores(model, feature_matrix, settings) -> ndarray`
Fit with `HIT_MODEL_PARAMETERS`; the scores are the model's hit probabilities smoothed
over `score_smoothing_frames` (an empty matrix gives an empty array).

### `local_maxima(values, min_gap, floor) -> list[int]`
Indices of the local maxima above `floor`, greedily from the highest, keeping any two
at least `min_gap` apart.

```python
>>> from vision.movement import local_maxima
>>> local_maxima([0.1, 0.9, 0.8, 0.2, 0.7, 0.1, 0.6], min_gap=2, floor=0.5)
[1, 4, 6]

```

### `hit_candidates(frames, scores_by_player, settings, floor=0.05) -> list[HitCandidate]`
The local maxima of each player's scores as candidates, in frame order.

```python
>>> from vision.movement import HitCandidate, hit_candidates
>>> frames = np.arange(100, 110)
>>> scores = {"me": np.array([0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0.0]), "opp": np.array([0, 0, 0, 0, 0, 0.8, 0, 0, 0, 0.0])}
>>> hit_candidates(frames, scores, MovementSettings(min_hit_gap_frames=2))
[HitCandidate(frame=101, hitter='me', score=0.9), HitCandidate(frame=105, hitter='opp', score=0.8)]

```

### `gap_penalty(gap, settings) -> float`, `alternating_hit_chain(candidates, settings) -> list[HitCandidate]`
The dynamic programme described above. Each candidate weighs `score - hit_score_min`;
within a rally consecutive hits alternate hitters, are at least `min_hit_gap_frames`
apart and pay `gap_penalty` outside the usual gap; after more than `max_hit_gap_frames`
any candidate may follow (a new rally). Returns the best chain in frame order.

```python
>>> from vision.movement import alternating_hit_chain, gap_penalty
>>> settings = MovementSettings(hit_score_min=0.5, min_hit_gap_frames=10, max_hit_gap_frames=70, usual_gap_frames=(15, 55))
>>> gap_penalty(30, settings), gap_penalty(5, settings), round(gap_penalty(65, settings), 2)
(0.0, 0.3, 0.3)
>>> candidates = [HitCandidate(100, "me", 0.9), HitCandidate(104, "opp", 0.6), HitCandidate(130, "opp", 0.8), HitCandidate(160, "me", 0.7), HitCandidate(400, "opp", 0.9)]
>>> [(c.frame, c.hitter) for c in alternating_hit_chain(candidates, settings)]
[(100, 'me'), (130, 'opp'), (160, 'me'), (400, 'opp')]

```

The candidate at 104 is dropped: it is too close to 100 and the same side as 130.

### `split_into_rallies(chain, max_gap) -> list[list[HitCandidate]]`, `trim_rally_ends(rally, end_score_min) -> list[HitCandidate]`

```python
>>> from vision.movement import split_into_rallies, trim_rally_ends
>>> chain = alternating_hit_chain(candidates, settings)
>>> [[c.frame for c in rally] for rally in split_into_rallies(chain, 70)]
[[100, 130, 160], [400]]
>>> [c.frame for c in trim_rally_ends(chain[:3], end_score_min=0.75)]
[100, 130]

```

### `both_still_from(frame_index, near_speed, far_speed, settings) -> bool`, `rally_end_frame(last_hit_frame, next_rally_start, frames, near_speed, far_speed, settings) -> int`
A rally ends at the first frame after its last hit from which both players stay below
`still_speed_m_per_s` for `still_frames`; otherwise `max_hit_gap_frames` after the last
hit, capped at the frame before the next rally and at the last frame.

```python
>>> from vision.movement import rally_end_frame
>>> frames = np.arange(0, 200)
>>> near = np.where(frames < 120, 3.0, 0.1)
>>> far = np.where(frames < 110, 2.0, 0.2)
>>> rally_end_frame(100, 1000, frames, near, far, MovementSettings())
120
>>> rally_end_frame(100, 1000, frames, np.full(200, 3.0), far, MovementSettings())
170
>>> rally_end_frame(100, 150, frames, np.full(200, 3.0), far, MovementSettings())
149

```

### `serve_gap_frames(settings) -> int`, `inferred_serve(rally, first_frame, settings) -> HitCandidate | None`, `with_inferred_serves(rallies, first_frame, settings) -> list[list[HitCandidate]]`
The movement detector can't see a serve (nobody moves at one), so each rally gets an
inferred serve: a candidate for the other player than the first detected hitter,
`serve_to_return_seconds` (0.76 s, the ShuttleSet median from serve to return over
3,502 rallies; 23 frames at 30 fps, 46 at 60) before the first detected hit, with score
NaN. No serve is added when that frame is before `first_frame` (the rally began before
the tracked range), or with `infer_serves=False` (`--no-serves`).

```python
>>> from vision.movement import inferred_serve, serve_gap_frames, with_inferred_serves
>>> serve_gap_frames(MovementSettings()), serve_gap_frames(settings_for_video(60.0, 1920))
(23, 46)
>>> rally = [HitCandidate(100, "me", 0.9), HitCandidate(130, "opp", 0.8)]
>>> inferred_serve(rally, 0, MovementSettings())
HitCandidate(frame=77, hitter='opp', score=nan)
>>> inferred_serve(rally, 90, MovementSettings()) is None
True
>>> [[c.frame for c in r] for r in with_inferred_serves([rally, [HitCandidate(400, "opp", 0.9), HitCandidate(430, "me", 0.7)]], 0, MovementSettings())]
[[77, 100, 130], [377, 400, 430]]
>>> with_inferred_serves([rally], 0, MovementSettings(infer_serves=False)) == [rally]
True

```

### `detected_hits_only(hits) -> DataFrame`, `inferred_serves_only(hits) -> DataFrame`, `serve_evaluation_line(hits, labels, tolerance) -> str`
Split a hits table by `score` (NaN marks an inferred serve; a table without a `score`
column counts as all detected). The evaluation functions score detected hits only;
`serve_evaluation_line` matches the inferred serves to the labelled first shot of each
rally and reports how many are within `tolerance` frames, how often the server is the
labelled hitter (when the label has a position) and the mean offset.

```python
>>> from vision.movement import detected_hits_only, inferred_serves_only, serve_evaluation_line
>>> table = pd.DataFrame({"rally": [1, 1, 1], "frame": [77, 100, 130], "hitter": ["opp", "me", "opp"], "score": [np.nan, 0.9, 0.8], "reach": np.nan})
>>> detected_hits_only(table)["frame"].tolist(), inferred_serves_only(table)["frame"].tolist()
([100, 130], [77])
>>> labels = pd.DataFrame({"rally": [1, 1, 1], "frame": [80, 102, 131], "hitter": ["opp", "me", "opp"], "type": ["serve", "net", "lift"]})
>>> serve_evaluation_line(table, labels, 5)
'inferred serves: 1, labelled first shots: 1, within ±5 frames: 1, server right: 1/1, offset mean -3.0'

```

### `rally_tables(rallies, tracks, settings) -> (hits, rallies)`, `detect_hits(tracks, model, settings) -> (hits, rallies)`
`detect_hits` runs the whole chain on `PlayerTracks` with a fitted model and returns the
two tables in `HIT_COLUMNS` and `RALLY_COLUMNS` (rallies numbered from 1 within the
range, `reach` NaN). `rally_tables` adds the inferred serves and builds the tables from
already chosen rallies; in the example the first hit is at the range start, so no serve
fits.

```python
>>> from vision.movement import rally_tables
>>> rallies = [[HitCandidate(10, "me", 0.9), HitCandidate(11, "opp", 0.8)]]
>>> hits, table = rally_tables(rallies, tracks, MovementSettings())
>>> hits
   rally  frame hitter  score  reach
0      1     10     me    0.9    NaN
1      1     11    opp    0.8    NaN
>>> table
   rally  start_frame  end_frame  hits
0      1           10         12     2

```

### `sample_hit_scores(tracks, model, settings) -> dict[str, ndarray]`, `hits_from_scores(tracks, scores, settings) -> (hits, rallies)`
`detect_hits` in two steps, so scores computed once can be run through the rhythm step
with other thresholds (as `vision.hit_training` does when it tunes them).
`sample_hit_scores` gives each player's hit score per tracked sample; `hits_from_scores`
runs the chain, rallies and tables on them. `detect_hits(tracks, model, settings)` is
`hits_from_scores(tracks, sample_hit_scores(tracks, model, settings), settings)`.

```python
>>> from vision.movement import hits_from_scores, sample_hit_scores
>>> class EveryFrameHalf:
...     def predict_proba(self, features):
...         return np.tile([0.5, 0.5], (len(features), 1))
>>> scores = sample_hit_scores(tracks, EveryFrameHalf(), MovementSettings())
>>> {player: values.round(2).tolist() for player, values in scores.items()}
{'me': [0.5, 0.5, 0.5], 'opp': [0.5, 0.5, 0.5]}
>>> quiet = {player: np.zeros(3) for player in scores}
>>> [len(table) for table in hits_from_scores(tracks, quiet, MovementSettings())]
[0, 0]

```

## Model file and tracks

### `save_hit_model(model, model_path=MODEL_PATH, trained_on=None) -> Path`, `load_hit_model(model_path=MODEL_PATH)`
joblib file holding the model, `FEATURE_NAMES` and the names of the training ranges.
`load_hit_model` raises `FileNotFoundError` (the message gives the training command) if
the file is missing and `ValueError` if its feature names differ from the module's.

```python
>>> import tempfile
>>> from pathlib import Path
>>> from vision.movement import load_hit_model, save_hit_model
>>> load_hit_model(Path("no_such_folder/missing.joblib"))
Traceback (most recent call last):
FileNotFoundError: movement hit model not found: no_such_folder/missing.joblib (train with: python -m vision.movement --train ...)
>>> folder = Path(tempfile.mkdtemp())
>>> import joblib
>>> _ = joblib.dump({"model": None, "feature_names": ["speed"]}, folder / "old.joblib")
>>> try:
...     load_hit_model(folder / "old.joblib")
... except ValueError as error:
...     print(str(error).split(" was ")[1])
trained on other features; retrain with python -m vision.movement --train
>>> saved = save_hit_model("a model", folder / "new.joblib", trained_on=["match 1-2 set 1"])
>>> load_hit_model(saved)
'a model'

```

### `load_player_tracks(video_path, start_frame, end_frame, players_dir=PLAYERS_DIR) -> PlayerTracks`, `track_frame_stride(video_path, start_frame, end_frame, players_dir=PLAYERS_DIR) -> int`, `settings_for_tracks(video_path, frame_stride, base=MovementSettings()) -> MovementSettings`
Reads the two player files of the range (`FileNotFoundError` naming the
`vision.players` command when one is missing) with the stride from their tracking
record (1 without one); `settings_for_tracks` is `settings_for_video` at the effective
rate `video fps / stride` and the video's width. `--train` and `--cross-validate`
refuse ranges tracked with different strides.

### `labelled_range_data(labelled_range, settings) -> (tracks, labels, features, targets)`, `stacked_training_data(range_data) -> (X, y)`
Training inputs of one labelled range (tracks, `vision.hits.labelled_hits` table,
feature matrix and target per player) and their stacking over ranges and players.

## Evaluation

### `evaluation_summary(hits, labels, tolerance) -> dict`, `summary_line(name, totals, tolerance) -> str`, `add_totals(totals, summary) -> dict`
Counts (`labelled`, `detected`, `matched`, `hitter_correct`, `hitter_judged`) with
`vision.hits.match_hits` over the detected hits (inferred serves left out), their sum
over ranges and the printed line.

```python
>>> from vision.movement import add_totals, evaluation_summary, summary_line
>>> detected = pd.DataFrame({"frame": [100, 130, 200], "hitter": ["me", "opp", "me"]})
>>> labels = pd.DataFrame({"frame": [102, 131, 160], "hitter": ["me", "me", "opp"]})
>>> summary = evaluation_summary(detected, labels, 5)
>>> summary
{'labelled': 3, 'detected': 3, 'matched': 2, 'hitter_correct': 1, 'hitter_judged': 2}
>>> summary_line("example", add_totals({}, summary), "±0.25 s (±8 frames at 30 fps)")
'example ±0.25 s (±8 frames at 30 fps): labelled 3, detected 3, matched 2, precision 66.7%, recall 66.7%, hitter correct 1/2'

```

### `cross_validation_lines(labelled_ranges, settings, seconds=MATCH_TOLERANCE_SECONDS) -> list[str]`
Holds out each range in turn, trains on the others, detects on the held-out one and
reports a line per range and tolerance plus the totals.

### `labels_from_file(labels_file, start_frame, end_frame) -> DataFrame`, `type_agreement_line(hits, labels, predictions, tolerance) -> str`, `detail_agreement_line(hits, labels, predictions, tolerance) -> str`, `own_predictions(database_path, match_id) -> DataFrame`
Hand labels as the `rally, frame, hitter, type, detail` table the evaluation functions
take. A `player` outside `me`/`opp` becomes a missing hitter, as for a ShuttleSet serve
without a position, and a missing or empty `type_detail` becomes a missing detail.
The functions give the shot-type agreement line (groups) and the detail agreement line.
The detail line compares `type_detail_pred` with the labelled detail, where both exist
and the label isn't `unknown`. `own_predictions` reads the `frame_num, type_pred` rows
of a match, plus `type_detail_pred` when the `shots` table has that column.
`--evaluate-labels` with `--database` prints both lines per tolerance.
On a labels file built from a ShuttleSet range, `--evaluate-labels` prints exactly what
`--evaluate` prints for that range (checked on 14201-14947: 25 matched, 100% / 96.2%,
hitter 25/25, shot type 22/24).

```python
>>> labels_file = Path(tempfile.mkdtemp()) / "clip.csv"
>>> _ = pd.DataFrame({"rally": [1, 1, 1], "frame_num": [100, 130, 900], "player": ["", "me", "opp"], "type_group": ["serve", "net", "lift"], "source": "hand"}).to_csv(labels_file, index=False)
>>> from vision.movement import labels_from_file, type_agreement_line
>>> labels_from_file(labels_file, 0, 500)
   rally  frame hitter   type detail
0      1    100    NaN  serve    NaN
1      1    130     me    net    NaN
>>> found = pd.DataFrame({"rally": [1, 1], "frame": [101, 131], "hitter": ["opp", "me"], "score": [0.9, 0.8], "reach": np.nan})
>>> type_agreement_line(found, labels_from_file(labels_file, 0, 500), pd.DataFrame({"frame_num": [101, 131], "type_pred": ["serve", "drive"]}), 5)
'shot type: 1/2 matched hits with a prediction and a labelled type agree = 50.0%'
>>> from vision.movement import detail_agreement_line
>>> detailed_file = Path(tempfile.mkdtemp()) / "detailed.csv"
>>> _ = pd.DataFrame({"rally": 1, "frame_num": [100, 130], "player": ["opp", "me"], "type_group": ["serve", "net"], "type_detail": ["serve_short", "return_net"], "source": "hand"}).to_csv(detailed_file, index=False)
>>> detail_agreement_line(found, labels_from_file(detailed_file, 0, 500), pd.DataFrame({"frame_num": [101, 131], "type_pred": ["serve", "net"], "type_detail_pred": ["serve_short", "net"]}), 5)
'shot detail: 1/2 matched hits with a predicted and a labelled detail agree = 50.0%'
>>> detail_agreement_line(found, labels_from_file(detailed_file, 0, 500), pd.DataFrame({"frame_num": [101, 131], "type_pred": ["serve", "net"]}), 5)
'shot detail: no detail predictions in the database'

```

### `miss_lines(hits, labels, tolerance) -> list[str]`
One line per unmatched label (`miss at <frame>: <type>`, marking a rally's first shot)
and per unmatched detection (inside or outside the labelled rallies); inferred serves
are left out.

```python
>>> from vision.movement import miss_lines
>>> labels = pd.DataFrame({"rally": [1, 1, 1], "frame": [102, 131, 160], "hitter": ["me", "me", "opp"], "type": ["serve", "net", "lift"]})
>>> for line in miss_lines(detected, labels, 5):
...     print(line)
miss at 160: lift
false positive at 200 (me): outside the labelled rallies

```

## Check sheet

### `spot_check_path(video_path, start_frame, end_frame, checks_dir=SPOT_CHECK_DIR) -> Path`, `spot_check_sample(hits, hit_count, seed=1) -> DataFrame`, `box_row_at(boxes, player, frame_index)`, `box_centre(box_row) -> (float, float)`, `crop_around(frame, centre, caption, width=480, height=360) -> ndarray`, `hit_tile(frame, frame_index, boxes, centre, caption) -> ndarray`, `write_spot_check(video_path, hits, boxes, hit_count, output_path, fps) -> Path`
The `--spot-check` sheet: `hit_count` random hits (without replacement, in frame order),
three tiles each, cropped around the hitter's box centre (`(0, 0)` without a box), boxes
drawn in the players' colours. Frames past the video end are black tiles.

```python
>>> from vision.movement import box_centre, box_row_at, crop_around, spot_check_path, spot_check_sample
>>> from pathlib import Path
>>> spot_check_path(Path("data/recordings/stand_in/set1.mp4"), 0, 3599)
PosixPath('data/checks/hits/stand_in_set1_movement_spot_check_0_3599.jpg')
>>> many = pd.DataFrame({"rally": 1, "frame": range(0, 300, 10), "hitter": "me", "score": 0.9, "reach": np.nan})
>>> sample = spot_check_sample(many, 5)
>>> len(sample), sample["frame"].is_monotonic_increasing, sample["frame"].tolist() == spot_check_sample(many, 5)["frame"].tolist()
(5, True, True)
>>> box_table = pd.DataFrame({"frame": [10], "player": ["opp"], "left": [620], "top": [200], "right": [660], "bottom": [300]})
>>> box_centre(box_row_at(box_table, "opp", 10)), box_centre(None), box_row_at(box_table, "me", 10)
((640.0, 250.0), (0.0, 0.0), None)
>>> crop_around(np.zeros((720, 1280, 3), dtype=np.uint8), (640, 360), "caption").shape
(360, 480, 3)

```

### `main(arguments) -> int`
Command entry point (see Commands).
