# vision/in_play.py

A per-frame model of whether a rally is in play. It's an option next to the gap rule of
`vision.movement`, which decides rally ends from the hit chain (no hit-like candidate
for `max_hit_gap_frames`, end score at least 0.7). With the in-play model:

- rallies are the stretches where the model says "in play";
- hits are the hit chain inside each stretch;
- the inferred serve sits at the stretch's start.

The results are in `docs/results/rally_in_play.md`.

## What it learns

**Labels, from ShuttleSet's hit frames.** A frame is in play from a rally's first hit
(the serve) to its last hit plus `LANDING_SECONDS` (1.0 s, the time for the last shot to
come down). Every other tracked frame is out of play. The frames are those of the
`vision.hit_training` matches, which are tracked only around the labelled rallies (2 s
of padding each side; `docs/vision/label_offset.md`).

**Features (43 per frame):**

- per player:
  - the distance from the net now;
  - over centred windows of 1, 3 and 8 s: mean and max speed, mean absolute
    acceleration, the share of frames with the player detected, and mean distance from
    the net;
  - the max hit score over each window;
- the number of hit-score peaks (local maxima above `hit_score_min`, at least
  `min_hit_gap_frames` apart) in each window;
- the seconds since the last peak and until the next one, capped at 10 s.

The hit scores come from the movement hit scorer (`vision.movement`). For training,
each match's scores come from a hit model trained without that match, so they look like
scores on unseen video.

**Model:** the hit scorer's model family (`HistGradientBoostingClassifier`, depth 4,
200 iterations, no class weights). It's trained on every second frame.

**From probabilities to rallies:**

1. The probability is smoothed over 1 s.
2. Frames at 0.5 or more are in play.
3. Gaps shorter than 1 s inside play are closed.
4. Stretches shorter than 2 s are dropped.
5. Each remaining stretch becomes a rally if its hit chain (hit candidates from 0.2 s
   before its start to 0.2 s after its end) holds at least one detected hit.
6. The rally's inferred serve (score NaN, hitter = the other player of the first
   detected hit) is the stretch's first frame. If the first detected hit is within 0.3 s
   of the start, that hit is taken as the serve and no row is added. `--no-serves` turns
   the inferred serve off as before.

## Commands

Train and measure, held out by match, next to the rule:

```
.venv/bin/python -m vision.in_play data/hit_training/matches.txt --held-out [--save data/models/in_play_nine_matches.joblib] [--out data/in_play]
```

- `--held-out`: each match is scored by an in-play model trained on the other matches
  and by the rule, on the same hit scores. It writes `held_out.csv` in `--out` (one row
  per match and method) and prints the totals per method.
- `--save MODEL`: trains on every listed match and saves the model.
- `--gaps`: only prints, per match, the minutes tracked and out of play, the median gap
  between rallies, and the mean share of each gap inside the tracked stretches
  (`gap_coverage`). It loads no tracks and trains nothing.
- The leave-one-match-out hit scores are cached in `--out/scores/<match>.joblib`; a
  cache is reused only if it was made from the same training matches.

The run exits `0`; `1` with `error: ...` for a missing file or no usable match; `2`
without `--held-out`, `--save` or `--gaps`.

Detect with it:

```
.venv/bin/python -m vision.movement <video> --start N --end N --model data/models/movement_hits_ten_matches.joblib --in-play-model data/models/in_play_nine_matches.joblib [--hits-dir D]
```

Score the rally bounds against labelled rallies with
`python -m vision.label_scores ... --fps F` (`docs/vision/label_scores.md`).

## Constants

| Name | Value | Meaning |
|---|---|---|
| `IN_PLAY_MODEL_PATH` | `data/models/in_play.joblib` | default model path of `load_in_play_model` |
| `LANDING_SECONDS` | `1.0` | in play after the last hit |
| `WINDOW_SECONDS` | `(1.0, 3.0, 8.0)` | feature windows |
| `PEAK_DISTANCE_CAP_SECONDS` | `10.0` | cap of the seconds since / until a peak |
| `PROBABILITY_SMOOTHING_SECONDS` | `1.0` | smoothing of the probability |
| `IN_PLAY_PROBABILITY_MIN` | `0.5` | in-play threshold |
| `MIN_IN_PLAY_SECONDS`, `MIN_OUT_OF_PLAY_SECONDS` | `2.0`, `1.0` | shortest stretch kept, shortest gap kept |
| `SERVE_WINDOW_SECONDS` | `0.3` | a first hit this close to the start is the serve |
| `SPAN_PADDING_SECONDS` | `0.2` | hit candidates taken this far outside a stretch |
| `TRAINING_SAMPLE_STEP` | `2` | every second frame in training |

## Functions

### `frames_for_seconds(seconds, fps) -> int`, `centred_window(values, frames, stat) -> ndarray`, `seconds_since(mask, fps, cap_seconds=10.0) -> ndarray`, `seconds_until(mask, fps, cap_seconds=10.0) -> ndarray`

```python
>>> import numpy as np, pandas as pd
>>> from vision.in_play import centred_window, frames_for_seconds, seconds_since, seconds_until
>>> frames_for_seconds(0.3, 30.0), frames_for_seconds(0.001, 30.0)
(9, 1)
>>> centred_window(np.array([1.0, 2, 3, 4, 5]), 3, "max")
array([2., 3., 4., 5., 5.])
>>> peaks = np.array([0, 1, 0, 0, 1, 0], dtype=bool)
>>> seconds_since(peaks, 2.0, 2.0), seconds_until(peaks, 2.0, 2.0)
(array([2. , 0. , 0.5, 1. , 0. , 0.5]), array([0.5, 0. , 1. , 0.5, 0. , 2. ]))

```

### `player_signals(tracks, player, settings) -> dict`, `score_peaks(scores, settings) -> ndarray`, `in_play_feature_matrix(tracks, scores, settings) -> ndarray`, `IN_PLAY_FEATURE_NAMES`
The features above, one row per tracked frame, in the order of `IN_PLAY_FEATURE_NAMES`.

```python
>>> from vision.in_play import IN_PLAY_FEATURE_NAMES
>>> len(IN_PLAY_FEATURE_NAMES), IN_PLAY_FEATURE_NAMES[:3], IN_PLAY_FEATURE_NAMES[-2:]
(43, ('me_net_distance', 'opp_net_distance', 'me_speed_mean_1s'), ('seconds_since_peak', 'seconds_to_peak'))

```

### `labelled_rally_spans(labels, fps, landing_seconds=1.0) -> list[(int, int)]`, `in_play_targets(frames, rally_spans) -> ndarray`
Each labelled rally (by `set_no` and `rally`) from its first hit to its last hit plus the
landing time; and 1 for the frames inside any span.

```python
>>> from vision.in_play import in_play_targets, labelled_rally_spans
>>> labelled_rally_spans(pd.DataFrame({"set_no": 1, "rally": [1, 1, 2, 2], "frame": [100, 160, 400, 430]}), 30.0)
[(100, 190), (400, 460)]
>>> in_play_targets(np.arange(10), [(2, 4), (7, 8)])
array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])

```

### `runs_of(mask) -> list[(int, int)]`, `in_play_spans(probability, fps, threshold=0.5, min_in_play_seconds=2.0, min_gap_seconds=1.0) -> list[(int, int)]`
Runs of true values as inclusive sample-index spans. Then the in-play stretches of a
smoothed probability: gaps under 1 s inside play are closed, and stretches under 2 s
are dropped.

```python
>>> from vision.in_play import in_play_spans, runs_of
>>> runs_of(np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool))
[(1, 2), (4, 4), (7, 7)]
>>> probability = np.array([0.1] * 5 + [0.9] * 70 + [0.2] * 10 + [0.8] * 80 + [0.3] * 40 + [0.9] * 70 + [0.1] * 5)
>>> in_play_spans(probability, 30.0)
[(5, 164), (205, 274)]

```

### `rally_of_span(candidates, span, settings) -> list[HitCandidate]`, `hits_in_play(tracks, scores, probability, settings) -> (hits, rallies)`, `detect_hits_in_play(tracks, scores, in_play_model, settings) -> (hits, rallies)`
The hit chain inside a stretch, with the inferred serve at its start unless the first
hit is within 0.3 s of it. The tables have the columns of `vision.movement`, in video
frames.

```python
>>> from vision.in_play import rally_of_span
>>> from vision.movement import HitCandidate, settings_for_video
>>> settings = settings_for_video(30.0, 1280.0)
>>> candidates = [HitCandidate(20, "opp", 0.9), HitCandidate(45, "me", 0.8), HitCandidate(70, "opp", 0.7)]
>>> [(hit.frame, hit.hitter) for hit in rally_of_span(candidates, (0, 90), settings)]
[(0, 'me'), (20, 'opp'), (45, 'me'), (70, 'opp')]
>>> [(hit.frame, hit.hitter) for hit in rally_of_span(candidates, (15, 90), settings)]
[(20, 'opp'), (45, 'me'), (70, 'opp')]

```

### `SpanComparison`, `overlap(first, second) -> int`, `compare_spans(labelled, found, fps) -> SpanComparison`
Labelled against found rally spans:

- **matched**: one-to-one pairs, largest overlap first;
- **merged**: found spans that overlap two or more labelled ones;
- **split**: labelled spans that two or more found ones overlap;
- **missed** / **extra**: spans that overlap nothing on the other side;
- **start_errors** / **end_errors**: found minus labelled, in seconds, for the matched
  pairs.

```python
>>> from vision.in_play import compare_spans
>>> comparison = compare_spans([(0, 99), (150, 249), (400, 499)], [(5, 260), (410, 480), (600, 650)], 30.0)
>>> comparison.matched, comparison.merged, comparison.split, comparison.missed, comparison.extra
(2, 1, 0, 0, 1)
>>> [round(error, 2) for error in comparison.start_errors], [round(error, 2) for error in comparison.end_errors]
([-4.83, 0.33], [0.37, -0.63])

```

### `gap_coverage(rally_spans, stretches, fps) -> dict`, `gap_lines(match_list) -> list[str]`
How much out-of-play time tracked stretches hold:

- the minutes tracked and out of play, and the out-of-play share;
- the median gap between consecutive rally spans (the frames strictly between one
  span's end and the next span's start, in seconds);
- the mean share of a gap's frames inside a stretch.

`gap_lines` prints it per match of a match list, as a Markdown table.

```python
>>> from vision.in_play import gap_coverage
>>> coverage = gap_coverage([(10, 19), (40, 49), (90, 99)], [(5, 24), (35, 54), (85, 104)], 10.0)
>>> round(coverage["tracked_minutes"] * 60, 1), round(coverage["out_of_play_share"], 3), coverage["median_gap_seconds"], round(coverage["gap_tracked_share"], 3)
(6.0, 0.5, 3.0, 0.375)

```

### `fit_in_play_model(feature_matrix, targets)`, `in_play_probability(model, feature_matrix, settings) -> ndarray`, `save_in_play_model(model, model_path, trained_on)`, `load_in_play_model(model_path)`
Fit, smoothed probability, and the saved file (`model`, `feature_names`,
`landing_seconds`, `trained_on`). A missing file is a `FileNotFoundError` naming the
training command, and a file made with other features is a `ValueError`.

### `first_row_on_serve(hits, rallies, labels, tolerance) -> (int, int)`, `stretch_truth`, `in_play_training_rows`, `video_spans`, `frame_agreement`, `method_row`, `held_out_match_rows`, `cached_scores`, `summary_lines`, `run`, `main`
The training and held-out run. `first_row_on_serve` counts the labelled serves (first
hit of each labelled rally) that a detected rally's first row matches within the
tolerance (the held-out run uses ±0.50 s, in frames at the match's fps). It's the
condition for the classifier to see the serve at `ball_round` 1. The held-out table
records hit recall and precision per window as `labelled_0.25s`, `matched_0.5s`, and
so on.
