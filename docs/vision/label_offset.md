# vision/label_offset.py

Finds the frame offset between a ShuttleSet match's labels and a downloaded copy of its
video. A broadcast cuts to close-ups and replays between rallies and back to the main
camera for play, so the frames the main camera shows (the court-line view check of
`vision.players`) should coincide with the labelled rallies (first to last labelled hit).
The tool samples the view check through the whole video, then slides the labelled
rallies over it and takes the offset with the highest correlation. Labels made at a
different frame rate than the video are first scaled by `video fps / label fps`.

A video frame `f` corresponds to the label frame `(f - offset) / label_scale`; with
`label_scale` 1 and offset 0 (Axelsen vs Ginting) the labels apply as they are.

This **coarse** offset is only good to about 2 s: the broadcast often joins the main view
after the serve and leaves it after the last hit, so the correlation peak is a broad
hill and can sit off zero (Axelsen vs Ginting: +48 frames, 1.6 s, where the true offset is
0). The **fine** offset comes after tracking the labelled rallies: the movement hit
scorer's score of the labelled hitter, averaged over all labelled hits shifted by each
offset within ±3 s of the coarse one (`fine_offset_scores`), peaks sharply at the true
offset. On Axelsen vs Ginting's 107 tracked stretches with the shipped model,
`--fine` prints 0.670 at 0 and at most 0.266 more than 10 frames away (clarity 0.404).

## Commands

The three steps of aligning a match, in order (tracking in between):

```
.venv/bin/python -m vision.label_offset <video> --match <ShuttleSet folder> --label-fps <25|30> [--stride 3] [--max-offset-seconds 30]   # coarse
.venv/bin/python -m vision.label_offset <video> --stretches                                                                        # what to track
.venv/bin/python -m vision.players <video> --stretches data/alignment/<video id>_stretches.csv [--batch-frames 4]                    # tracking (GPU)
.venv/bin/python -m vision.label_offset <video> --fine [--model data/models/movement_hits.joblib]                                  # fine
```

**Coarse** (the first command):
Needs the video's court calibration (`python -m vision.court <video> --shuttleset-match <folder>`).
Writes `data/alignment/<video id>.json` (`match`, `offset_frames`, `label_scale`, `score`,
`runner_up_score`, `clarity`, `clear`, `stride`) and `data/checks/alignment/<video id>_offset.png`
(correlation against offset, the best offset marked, a grey line at 0). Prints one line:
offset, correlation, runner-up, clarity and `clear` or `UNCLEAR`. Exit codes: `0`; `1` with
`error: ...` for a missing video, calibration or set CSV, or when `--match` / `--label-fps` is missing.

**`--stretches`** reads the coarse record and writes `data/alignment/<video id>_stretches.csv`
(`start_frame, end_frame`): every labelled rally shifted by the coarse offset, padded by
2 s each side (the coarse offset is only good to about 2 s), and merged with the next
when less than 1 s apart, so no two stretches overlap. Only the labelled rallies are
tracked: nothing outside them has labels. It prints the count, the frames to track and
the tracking command.

**`--fine`** needs the stretches tracked. It scores every stretch's frames with the hit
model, pools them, searches ±3 s around the coarse offset, adds `fine_offset_frames`,
`fine_score`, `fine_runner_up_score`, `fine_clarity` and `fine_clear` to the record,
and draws `data/checks/alignment/<video id>_fine_offset.png`. It prints one line, for
example `fine offset 0 frames (coarse 48), mean hitter score 0.670, runner-up 0.266,
clarity 0.404, clear` for Axelsen vs Ginting. The labelled hitter comes from
`player_location` through the video's homography (pixels scaled by
`video width / 1280`). Exit code `1` with `error: ...` when the record, the stretch list,
a tracking file or the model is missing.

**Clarity** is the best score minus the best one far enough away from it: more than 10 s
for the coarse offset (the right hill against any other), more than 10 frames for the
fine one. A match whose coarse clarity is below `MIN_PEAK_CLARITY` (0.1) or whose fine
clarity is below `MIN_FINE_CLARITY` (0.2) is reported unclear and left out.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `VIEW_SAMPLE_STRIDE` | `3` | view check on every 3rd frame |
| `MAX_OFFSET_SECONDS` | `30.0` | search range either way |
| `CLARITY_EXCLUSION_SECONDS` | `10.0` | how far the coarse runner-up must be from the best offset |
| `MIN_PEAK_CLARITY` | `0.1` | below this the coarse offset is unclear |
| `FINE_SEARCH_SECONDS`, `FINE_EXCLUSION_FRAMES`, `MIN_FINE_CLARITY` | `3.0`, `10`, `0.2` | the fine search range around the coarse offset, the fine runner-up distance, the fine clarity threshold |
| `STRETCH_PADDING_SECONDS`, `STRETCH_MERGE_SECONDS` | `2.0`, `1.0` | padding of each labelled rally, the gap under which two stretches merge |
| `LABEL_FRAME_WIDTH` | `1280.0` | the frame width ShuttleSet's pixel coordinates refer to |
| `OFFSETS_DIR`, `CHECKS_DIR` | `data/alignment`, `data/checks/alignment` | outputs |

## Functions

### `rally_spans(shots, frame_scale=1.0) -> ndarray`, `rally_indicator(spans, frames, offset=0) -> ndarray`
First and last labelled hit frame of every rally (per set and rally), scaled to the video's
frame rate; 1.0 on the frames inside a rally shifted by `offset`, else 0.0.

```python
>>> import numpy as np, pandas as pd
>>> from vision.label_offset import rally_indicator, rally_spans
>>> shots = pd.DataFrame({"set_no": [1, 1, 1, 1], "rally": [1, 1, 2, 2], "frame_num": [10.0, 20.0, 50.0, 60.0]})
>>> rally_spans(shots).tolist(), rally_spans(shots, 1.2).tolist()
([[10, 20], [50, 60]], [[12, 24], [60, 72]])
>>> rally_indicator(rally_spans(shots), np.array([5, 10, 25, 55, 61]), offset=1).tolist()
[0.0, 0.0, 0.0, 1.0, 1.0]

```

### `offset_score(view, spans, frames, offset) -> float`, `offset_scores(view, spans, frames, offsets) -> ndarray`
Pearson correlation of the shifted rally indicator with the main-view signal (0.0 when
either is constant).

### `peak_clarity(offsets, scores, exclusion) -> (best, score, runner_up, clarity)`, `offset_from_view(view, frames, spans, max_offset, step, exclusion, frame_scale=1.0) -> OffsetResult`, `is_clear(result, min_clarity=0.1) -> bool`
A synthetic broadcast: four rallies, the main view on during each rally shifted by 6
frames (the labels are 6 frames early), close-ups in between.

```python
>>> from vision.label_offset import is_clear, offset_from_view, peak_clarity
>>> spans = np.array([[100, 400], [700, 900], [1300, 1700], [2000, 2300]])
>>> frames = np.arange(0, 2600, 3)
>>> view = rally_indicator(spans, frames, offset=6)
>>> result = offset_from_view(view, frames, spans, max_offset=300, step=3, exclusion=60)
>>> result.offset_frames, round(result.score, 3), is_clear(result)
(6, 1.0, True)
>>> peak_clarity([-3, 0, 3, 90], [0.2, 0.9, 0.5, 0.4], exclusion=10)
(0, 0.9, 0.4, 0.5)
>>> is_clear(offset_from_view(np.ones(len(frames)), frames, spans, 300, 3, 60))
False

```

### `fine_offset_scores(label_frames, label_hitters, scores_by_player, frames, offsets) -> ndarray`, `fine_offset(label_frames, label_hitters, scores_by_player, frames, centre, half_width, exclusion=10) -> OffsetResult`
For each offset, the mean over labelled hits of the labelled hitter's per-frame hit score
at the shifted frame (hits shifted outside the tracked frames are skipped); the best
offset within `centre ± half_width` with its clarity.

```python
>>> from vision.label_offset import fine_offset, fine_offset_scores
>>> frames = np.arange(1000, 1200)
>>> me_scores, opp_scores = np.zeros(200), np.zeros(200)
>>> me_scores[[30, 90, 150]] = 1.0
>>> opp_scores[[60, 120]] = 1.0
>>> labels, hitters = [1027, 1057, 1087, 1117, 1147], ["me", "opp", "me", "opp", "me"]
>>> fine_offset_scores(labels, hitters, {"me": me_scores, "opp": opp_scores}, frames, [0, 3, 30]).round(2).tolist()
[0.0, 1.0, 0.0]
>>> result = fine_offset(labels, hitters, {"me": me_scores, "opp": opp_scores}, frames, centre=0, half_width=40)
>>> result.offset_frames, result.score, round(result.clarity, 2)
(3, 1.0, 1.0)

```

### `tracking_stretches(spans, fps, padding_seconds=2.0, merge_seconds=1.0) -> list[(int, int)]`
The frame ranges to track for rally spans already shifted by the offset: sorted, padded
(never before frame 0), merged when the gap is at most `merge_seconds`.

```python
>>> from vision.label_offset import tracking_stretches
>>> tracking_stretches(np.array([[500, 900], [100, 300], [950, 1000]]), fps=30.0)
[(40, 360), (440, 1060)]
>>> tracking_stretches(np.array([[10, 20]]), fps=25.0, padding_seconds=1.0)
[(0, 45)]

```

### `labelled_hitters(shots, homography, frame_scale, offset, pixel_scale=1.0) -> DataFrame`, `pooled_scores(scored_stretches) -> (frames, scores_by_player)`
Every shot as `set_no, rally, frame, hitter, type`: the frame scaled and shifted to the
video, the hitter `me` (near side) or `opp` from `player_location`, missing (NaN) without one.
`pooled_scores` joins the per-stretch frames and per-player scores into one series for
`fine_offset_scores`.

```python
>>> from vision.label_offset import labelled_hitters, pooled_scores
>>> identity = np.eye(3)
>>> shots = pd.DataFrame({"set_no": [1, 1], "rally": [1, 1], "frame_num": [100.0, 130.0], "player_location_x": [3.0, np.nan], "player_location_y": [10.0, np.nan], "type": ["smash", "lob"]})
>>> labelled_hitters(shots, identity, 1.2, 5).to_dict("records")
[{'set_no': 1, 'rally': 1, 'frame': 125, 'hitter': 'opp', 'type': 'smash'}, {'set_no': 1, 'rally': 1, 'frame': 161, 'hitter': nan, 'type': 'lob'}]
>>> frames, scores = pooled_scores([(np.array([10, 11]), {"me": np.array([0.1, 0.2]), "opp": np.array([0.3, 0.4])}), (np.array([50]), {"me": np.array([0.5]), "opp": np.array([0.6])})])
>>> frames.tolist(), scores["me"].tolist(), scores["opp"].tolist()
([10, 11, 50], [0.1, 0.2, 0.5], [0.3, 0.4, 0.6])

```

### `stretches_path(video_path)`, `fine_check_image_path(video_path)`, `read_offset_record(video_path)`, `write_stretches(video_path)`, `read_stretch_list(video_path)`, `match_labels(video_path, record, offset)`, `find_fine_offset(video_path, model_path=MODEL_PATH, search_seconds=3.0) -> (OffsetResult, offsets, profile)`
The file side of `--stretches` and `--fine`. `read_offset_record` and `read_stretch_list`
raise `FileNotFoundError` naming the command that writes the missing file.

```python
>>> from pathlib import Path
>>> from vision.label_offset import fine_check_image_path, read_stretch_list, stretches_path
>>> stretches_path(Path("data/shuttleset_videos/x.mp4")).as_posix(), fine_check_image_path(Path("data/match.mp4")).as_posix()
('data/alignment/shuttleset_videos_x_stretches.csv', 'data/checks/alignment/match_fine_offset.png')
>>> read_stretch_list(Path("data/shuttleset_videos/missing.mp4"))
Traceback (most recent call last):
FileNotFoundError: no stretch list data/alignment/shuttleset_videos_missing_stretches.csv (run: python -m vision.label_offset data/shuttleset_videos/missing.mp4 --stretches)

```

### `score_profile(scores, offsets, width=900, height=300, score_name="correlation") -> ndarray`, `match_shots(match_folder, set_dir=SHUTTLESET_SET_DIR) -> DataFrame`, `view_signal(video_path, calibration, stride=3, end_frame=None)`, `video_fps(video_path)`, `offset_record_path(video_path)`, `check_image_path(video_path)`, `find_offset(video_path, match_folder, label_fps, stride=3, max_offset_seconds=30.0)`, `main(arguments)`
The check image, all sets of a match's labels (`FileNotFoundError` when none exist), the
sampled view signal, and the file side.

```python
>>> from pathlib import Path
>>> from vision.label_offset import check_image_path, offset_record_path, score_profile
>>> score_profile(np.array([0.1, 0.8, 0.3]), np.array([-3, 0, 3])).shape
(300, 900, 3)
>>> offset_record_path(Path("data/shuttleset_videos/x.mp4")).as_posix(), check_image_path(Path("data/match.mp4")).as_posix()
('data/alignment/shuttleset_videos_x.json', 'data/checks/alignment/match_offset.png')

```
