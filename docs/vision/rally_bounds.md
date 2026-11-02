# vision/rally_bounds.py

Scores the rallies of several detectors against a video's hand-marked rally bounds,
using only the rallies whose bounds the user checked (`bounds_checked` in the rallies
file from `python -m vision.label`). Each detector is a hit model, its thresholds, and
either the gap rule or an in-play model (`docs/vision/in_play.md`). All of them run on
the same tracks.

## Command

```
.venv/bin/python -m vision.rally_bounds <video> --rallies <labels stem>_rallies.csv \
    [--method NAME HIT_MODEL HIT_SCORE_MIN RALLY_END_SCORE_MIN IN_PLAY_MODEL|rule ...] \
    [--found NAME HITS_CSV ...] [--end N] [--labels LABELS_CSV] [--players-dir D]
```

`--method` runs a detector; `--found` scores a saved `vision.movement` run (its
`rallies_*.csv` sits next to the hits file). At least one of them is needed. `--end N` is
the last labelled frame: later frames are not judged, and detected rallies that start
after it are left out.

The video must be tracked in full (frame 0 to its last frame). It prints how many checked
rallies are scored, then one Markdown row per method:
- frames right, among the frames outside every unchecked rally: in play by the detector
  (inside one of its rallies, `start_frame` to `end_frame`) and in play by the user
  (inside a checked rally), or out of play by both;
- rallies found, then matched, merged, split, missed and extra against the checked
  rallies (`vision.in_play.compare_spans`). A detected rally that overlaps only
  unchecked rallies is left out, so it isn't counted as extra;
- the start and end error of the matched rallies, median absolute and mean, in seconds.

With `--labels` (the hand labels CSV), a second table follows, one row per method:
- serve rows: each detected rally's first row;
- serves found at ±0.25 s and ±0.50 s: serve rows matched one to one with the labelled
  serves (`type_group = serve`, up to `--end`); fake serves are the serve rows left over;
- detected hits out of play: detected hits (not inferred serves) up to the last judged
  frame that fall outside every hand-marked rally, checked or not.

Exit codes: `0`; `1` with `error: ...` for a missing file or column; `2` for bad
arguments.

## Functions

### `found_spans(rallies)`, `outside_unchecked_only(found, checked, unchecked)`, `frames_right(found, frames, checked, unchecked) -> (int, int)`

```python
>>> import numpy as np, pandas as pd
>>> from vision.rally_bounds import found_spans, frames_right, outside_unchecked_only
>>> found_spans(pd.DataFrame({"start_frame": [0, 30], "end_frame": [9, 39]}))
[(0, 9), (30, 39)]
>>> outside_unchecked_only([(0, 9), (30, 39), (60, 69)], checked=[(2, 8)], unchecked=[(31, 35)])
[(0, 9), (60, 69)]
>>> frames_right([(0, 9)], np.arange(0, 20), checked=[(2, 8)], unchecked=[(15, 19)])
(12, 15)

```

### `bound_row(name, found, frames, checked, unchecked, fps) -> dict`, `bound_lines(rows) -> list[str]`, `method_tables(...)`, `saved_tables(hits_file, video_path, players_dir)`, `main(arguments) -> int`
One method's numbers, the printed table, and a method's hits and rallies with the
track frames: run on the whole video (`vision.movement.detected_tables`) or read from a
saved run.

```python
>>> from vision.rally_bounds import bound_lines, bound_row
>>> row = bound_row("rule", [(0, 9), (60, 69)], np.arange(0, 80), checked=[(2, 8)], unchecked=[(30, 40)], fps=10.0)
>>> row["matched"], row["extra"], row["start_error_median_abs_s"], row["end_error_mean_s"]
(1, 1, 0.2, 0.1)
>>> bound_lines([row])[2]
'| rule | 56/69 = 81.2% | 2 | 1 of 1 | 0 | 0 | 0 | 1 | 0.20 | 0.10 | -0.20 | +0.10 |'

```

### `up_to(end_frame, hits, rallies, frames)`, `serve_frames(hits)`, `out_of_play_hits(hits, checked, unchecked, last_judged)`, `serve_row(...)`, `serve_lines(rows)`, `labelled_serves_of(labels_file, end_frame)`

```python
>>> from vision.rally_bounds import out_of_play_hits, serve_frames, serve_lines, serve_row, up_to
>>> hits = pd.DataFrame({"rally": [1, 1, 2, 2], "frame": [5, 20, 52, 60], "hitter": ["me", "opp", "opp", "me"], "score": [np.nan, 0.9, 0.8, 0.7]})
>>> rallies = pd.DataFrame({"rally": [1, 2], "start_frame": [5, 52], "end_frame": [25, 65]})
>>> serve_frames(hits)
[5, 52]
>>> kept_hits, kept_rallies, kept_frames = up_to(55, hits, rallies, np.arange(0, 80))
>>> len(kept_hits), len(kept_rallies), int(kept_frames.max())
(3, 2, 55)
>>> out_of_play_hits(hits, checked=[(4, 22)], unchecked=[], last_judged=79)
2
>>> row = {"method": "stance", **serve_row(hits, [6, 40], [(4, 22)], [], np.arange(0, 80), fps=10.0)}
>>> serve_lines([row])[2]
'| stance | 2 | 1 of 2 | 1 | 1 of 2 | 1 | 2 |'

```
