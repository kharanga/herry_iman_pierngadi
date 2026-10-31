# vision/hit_training.py

Trains and measures the movement hit scorer (`vision/movement.py`) on whole ShuttleSet
matches instead of a few hand-picked ranges, and scores every match with a model that
never saw it. The results on nine matches are in `docs/results/hits_ten_matches.md`.

## Before it runs

Every match in the list must be aligned and tracked, with the three steps of
`docs/vision/label_offset.md`:

```
.venv/bin/python -m vision.court <video> --shuttleset-match <folder>
.venv/bin/python -m vision.label_offset <video> --match <folder> --label-fps <fps>
.venv/bin/python -m vision.label_offset <video> --stretches
.venv/bin/python -m vision.players <video> --stretches data/alignment/<video id>_stretches.csv --batch-frames 4
.venv/bin/python -m vision.label_offset <video> --fine
```

A match whose coarse clarity is below 0.1, or whose fine clarity is below 0.2, or with
no fine offset yet, is left out with a printed reason (`exclusion_reason`).

## Command

```
.venv/bin/python -m vision.hit_training <match list> [--held-out [--tune] [--by-player]] [--learning-curve] [--baseline MODEL] [--save MODEL [--exclude FOLDER ...]] [--out data/hit_training]
```

The match list is a text file with one match per line: the ShuttleSet match folder and the
video path, separated by a space (`#` lines are skipped).

| Option | What it does | Table in `--out` |
|---|---|---|
| `--held-out` | each match scored by a model trained on all the others, at the shipped thresholds | `held_out.csv` |
| `--tune` | also tunes the hit score and rally-end score thresholds for each held-out match. A grid of 5 x 4 values, best F1 at ±5 over the training matches, each scored by a model trained without it, so the held-out match plays no part | rows with `thresholds` = `tuned` |
| `--by-player` | again, but also leaving out every training match that shares a player with the held-out one (players from ShuttleSet's `match.csv`) | `held_out_by_player.csv` |
| `--learning-curve` | each match scored by models trained on 1, 3, 5 and all other matches (random subsets, seed 2026) | `learning_curve.csv` |
| `--baseline MODEL` | every match scored by an existing model | `baseline.csv` |
| `--save MODEL` | trains on every listed match except `--exclude` ones and saves the model | |

Every table has one row per match and tolerance (±5 and ±10 frames): `labelled`,
`detected`, `matched`, `hitter_correct`, `hitter_judged` (the counts of
`vision.movement.evaluation_summary`). The command prints the totals; `--held-out`
also prints them by fps. Labels are every shot of every set at
`frame_num x label_scale + fine offset`, so serves are labelled hits too; detections
leave out the inferred serve rows. Training uses the targets of the shipped model:
frames within ±2 of a hit of that player, on frames where at least one player was
found. Exit code `1` with `error: ...` when the list, a record, a stretch list, a tracking
file or the baseline model is missing, or no match is usable.

Run time on the Mac's CPU for nine matches (9,424 hits): about 1 min to load, 5 s per
fit; `--held-out --tune --by-player --learning-curve` together take about 20 min.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `RESULTS_DIR` | `data/hit_training` | default `--out` |
| `TOLERANCES`, `TUNING_TOLERANCE` | `(5, 10)`, `5` | match tolerances reported, the one tuned on |
| `HIT_SCORE_GRID`, `RALLY_END_GRID` | `(0.35, 0.45, 0.5, 0.55, 0.65)`, `(0.5, 0.6, 0.7, 0.8)` | thresholds tried by `--tune` |
| `LEARNING_CURVE_SIZES`, `LEARNING_CURVE_SEED` | `(1, 3, 5)`, `2026` | subset sizes (plus all others), the subset seed |

## Functions

### `read_match_list(list_file) -> list[(str, Path)]`, `exclusion_reason(record) -> str | None`

```python
>>> import tempfile
>>> from pathlib import Path
>>> from vision.hit_training import exclusion_reason, read_match_list
>>> folder = Path(tempfile.mkdtemp())
>>> _ = (folder / "matches.txt").write_text("# folder video\nMatch_A data/a.mp4\n\nMatch_B data/b.mp4\n")
>>> [(name, path.as_posix()) for name, path in read_match_list(folder / "matches.txt")]
[('Match_A', 'data/a.mp4'), ('Match_B', 'data/b.mp4')]
>>> _ = (folder / "bad.txt").write_text("Match_A\n")
>>> read_match_list(folder / "bad.txt")  # doctest: +ELLIPSIS
Traceback (most recent call last):
ValueError: .../bad.txt: every line needs a ShuttleSet match folder and a video path
>>> exclusion_reason({"clarity": 0.026})
'coarse offset unclear (clarity 0.026)'
>>> exclusion_reason({"clarity": 0.5})
'no fine offset yet (run python -m vision.label_offset <video> --fine)'
>>> exclusion_reason({"clarity": 0.5, "fine_offset_frames": 3, "fine_clarity": 0.12})
'fine offset unclear (clarity 0.120)'
>>> exclusion_reason({"clarity": 0.5, "fine_offset_frames": 0, "fine_clarity": 0.4}) is None
True

```

### `players_of_matches(names, set_dir=SHUTTLESET_SET_DIR) -> dict[str, set]`, `training_matches(held_out, names, players=None) -> list[str]`
The training matches for a held-out one: all the others, or with `players` only those
that share no player with it.

```python
>>> from vision.hit_training import players_of_matches, training_matches
>>> _ = (folder / "match.csv").write_text("video,winner,loser\nA,Ann,Bea\nB,Cat,Ann\nC,Dee,Eve\n")
>>> players = players_of_matches(["A", "B", "C"], folder)
>>> sorted(players["B"])
['Ann', 'Cat']
>>> training_matches("A", ["A", "B", "C"]), training_matches("A", ["A", "B", "C"], players)
(['B', 'C'], ['C'])

```

### `f1_of(totals) -> float`, `totals_line(name, rows) -> str`, `summary_lines(title, rows, group_columns=()) -> list[str]`

```python
>>> import pandas as pd
>>> from vision.hit_training import f1_of, summary_lines
>>> round(f1_of({"labelled": 100, "detected": 80, "matched": 60}), 3)
0.667
>>> rows = pd.DataFrame({"match": ["A", "B", "A", "B"], "fps": [25.0, 30.0, 25.0, 30.0], "tolerance": [5, 5, 10, 10], "labelled": [10, 20, 10, 20], "detected": [10, 18, 10, 18], "matched": [8, 15, 9, 17], "hitter_correct": [8, 15, 9, 16], "hitter_judged": [8, 15, 9, 17]})
>>> print("\n".join(summary_lines("held out", rows)))
held out, tolerance 5: labelled 30, detected 28, matched 23, precision 82.1%, recall 76.7%, hitter 23/23 = 100.0%
held out, tolerance 10: labelled 30, detected 28, matched 26, precision 92.9%, recall 86.7%, hitter 25/26 = 96.2%
>>> summary_lines("held out", rows, ("fps",))[0]
'held out, fps 25.0, tolerance 5: labelled 10, detected 10, matched 8, precision 80.0%, recall 80.0%, hitter 8/8 = 100.0%'

```

### `MatchData(name, video_path, fps, settings, tracks, labels)`, `load_match(folder, video_path) -> MatchData`, `loaded_matches(match_list) -> dict[str, MatchData]`
One aligned, tracked match: its settings for its fps (`settings_for_video`), the tracks
of every stretch, and the labels at the fine offset (`vision.label_offset.match_labels`).

### `training_rows(match) -> (X, y)`, `fit_on(names, rows_by_match)`, `match_scores(model, match) -> list`, `detected_hits(match, scores, settings) -> DataFrame`, `threshold_settings(match, thresholds) -> MovementSettings`, `best_thresholds(matches, scores_by_match) -> (float, float)`
The pieces of each evaluation: the feature rows and targets of a match, a model fitted on
the named matches (rows stacked in the given order), per-stretch scores, the hits
detected from scores (`vision.movement.hits_from_scores`, no inferred serves), and the
threshold grid search.

### `held_out_rows(matches, rows_by_match, tune=False, players=None)`, `learning_curve_rows(matches, rows_by_match, sizes=(1, 3, 5), seed=2026)`, `baseline_rows(matches, model)`, `count_rows(match, hits, **columns)`, `write_table(rows, output_path)`, `main(arguments)`
The tables of the command, one row per match and tolerance.
