# vision/label_scores.py

Scores hits files against hand labels that cover only part of a video's hits, such as
the club footage labelled with `vision.label_ui` (`docs/vision/label_ui.md`). Unlabelled
hits exist, so a detection without a label is not counted as a false positive.
Instead:

- **recall**, **hitter** and **shot type** are measured on the labelled hits only;
- **precision** is counted only inside the labelled rallies. It is a lower bound,
  because a detection there can still be a real hit that was not labelled.

Every number is reported for the detected hits alone, and again with the inferred serve
rows (`score` NaN, `docs/vision/movement.md`) counted as detections.

## Command

```
.venv/bin/python -m vision.label_scores --labels data/labels/<video id>.csv --match-id <id> \
    --run NAME HITS_CSV DATABASE [--run ...] --fps F [--rallies CSV] [--tolerance S ...] [--breakdown-tolerance 0.5]
```

| Option | What it does |
|---|---|
| `--labels` | labels CSV with `rally, frame_num, player, type_group, source` (`source` = `hand` or `accepted`; missing means `hand`) |
| `--rallies` | the labels' rallies CSV (`rally, start_frame, end_frame`); default `<labels stem>_rallies.csv` next to the labels |
| `--run NAME HITS_CSV DATABASE` | one model: its hits file and the shots database holding the rows built from it and predicted (`vision.features`, then `shots.predict`); repeatable, one column group per run |
| `--match-id` | the own rows' match id in each database |
| `--fps F` | the video's frame rate, required. It turns the matching windows into frames, and times the comparison of each run's rallies file (`rallies_*.csv` next to its hits file) with the labelled rallies |
| `--tolerance S` | a matching window in seconds for the headline table, as frames at `--fps` (`vision.hits.MATCH_TOLERANCE_SECONDS`, default 0.25 and 0.5); repeatable. Headers read "±0.25 s (±15 frames at 60 fps)" |
| `--breakdown-tolerance S` | the window in seconds of the serve table, the breakdowns and the missed list (default 0.5) |

It prints Markdown:

- the headline table: recall, hitter, shot type and precision per run and tolerance;
- recall per label source when the labels have more than one;
- the labelled serves, found by an inferred serve row or by a detected hit;
- the labelled serves left out, per run and tolerance: matched with the inferred serve
  rows in, then scored on the labels whose group isn't `serve`. It gives non-serves
  matched, shot type agreement, and matched non-serves typed `serve`;
- recall, hitter and shot type by labelled type and by labelled player;
- one labelled-vs-predicted shot type table per run;
- shot detail agreement per run: `type_detail_pred` against the labelled `type_detail`,
  on matched hits where both exist and the label isn't `unknown`. It prints "no label
  has a type_detail" for older label files, and "no detail predictions" for a database
  without `type_detail_pred`;
- the rally bounds against the labelled rallies: rallies matched, merged
  (one detected rally over two labelled ones), split, missed and extra, and the start
  and end errors in seconds;
- every labelled hit that any run missed, with the offset of that run's nearest detected
  hit.

Exit codes: `0`; `1` with `error: ...` for a missing file or a missing column; `2` for
bad arguments.

Matching is the one used everywhere else (`vision.hits.match_hits`): one to one,
closest pairs first. A labelled hit's shot type is compared with the `type_pred` of the
row at the matched frame. Labels typed `unknown` and rows with no prediction are left
out of the shot-type counts.

## Functions

### `read_labels(labels_file) -> DataFrame`, `labelled_rally_spans(label_rallies, labels) -> list[(int, int)]`
The labels as `rally, frame, hitter, type, source`, ordered by frame. A player other than
`me`/`opp` becomes NaN. The spans are the start and end frames of the rallies that hold
at least one label.

```python
>>> import pandas as pd, numpy as np
>>> from vision.label_scores import labelled_rally_spans
>>> labels = pd.DataFrame({"rally": [1, 1, 1, 2], "frame": [100, 160, 220, 900], "hitter": ["me", "opp", "me", "opp"], "type": ["serve", "lift", "clear", "serve"], "source": "hand"})
>>> labelled_rally_spans(pd.DataFrame({"rally": [1, 2, 3], "start_frame": [80, 880, 1500], "end_frame": [260, 990, 1700]}), labels)
[(80, 260), (880, 990)]

```

### `label_outcomes(hits, labels, predictions, tolerance, with_serves=False) -> DataFrame`, `agreement_counts(outcomes) -> dict`, `group_names(table, column) -> Series`, `counts_by(outcomes, column) -> dict`
One row per label, with:

- `detected_frame`: the matched detection, `<NA>` when missed;
- `detected_hitter`;
- `inferred_serve`: the match is an inferred serve row;
- `type_pred`: the prediction of the row at that frame.

`with_serves` also matches the inferred serve rows; without it, only the detected hits.
`agreement_counts` gives the labelled, matched, hitter and shot-type counts.
`counts_by` groups them by a column through `group_names`. A label with no hitter (a
player other than `me`/`opp`) is grouped as `unknown hitter`, and one with no type as
`no type`. Both are counted for recall; the first is left out of hitter agreement.

```python
>>> from vision.label_scores import agreement_counts, label_outcomes
>>> hits = pd.DataFrame({"rally": 1, "frame": [96, 158, 231, 600], "hitter": ["me", "opp", "opp", "me"], "score": [np.nan, 0.9, 0.8, 0.7]})
>>> predictions = pd.DataFrame({"frame_num": [96, 158, 231, 600], "type_pred": ["serve", "lift", "drop", "net"]})
>>> outcomes = label_outcomes(hits, labels, predictions, 10)
>>> outcomes["detected_frame"].tolist()
[<NA>, 158, <NA>, <NA>]
>>> served = label_outcomes(hits, labels, predictions, 15, with_serves=True)
>>> served["detected_frame"].tolist(), served["inferred_serve"].tolist()
([96, 158, 231, <NA>], [True, False, False, False])
>>> agreement_counts(served)
{'labelled': 4, 'matched': 3, 'hitter_right': 2, 'hitter_judged': 3, 'type_right': 2, 'type_judged': 3, 'detail_right': 0, 'detail_judged': 0}
>>> from vision.label_scores import counts_by, group_names
>>> no_player = served.assign(hitter=[np.nan, "opp", "me", "opp"])
>>> group_names(no_player, "hitter").tolist()
['unknown hitter', 'opp', 'me', 'opp']
>>> {name: (counts["labelled"], counts["hitter_judged"]) for name, counts in counts_by(no_player, "hitter").items()}
{'me': (1, 1), 'opp': (2, 1), 'unknown hitter': (1, 0)}

```

### `precision_in_labelled_rallies(hits, outcomes, spans, tolerance, with_serves=False) -> (int, int)`, `inside_spans(frame, spans, tolerance) -> bool`
The matched detections and all detections within `tolerance` frames of a labelled
rally. Detections outside every labelled rally are not counted at all.

```python
>>> from vision.label_scores import inside_spans, precision_in_labelled_rallies
>>> precision_in_labelled_rallies(hits, served, [(80, 260), (880, 990)], 15, with_serves=True)
(3, 3)
>>> precision_in_labelled_rallies(hits, outcomes, [(80, 260)], 10)
(1, 2)
>>> inside_spans(600, [(80, 260)], 10), inside_spans(265, [(80, 260)], 10)
(False, True)

```

### `detail_lines(runs, labels, tolerance) -> list[str]`, `rally_bound_lines(runs, spans, fps) -> list[str]`, `rallies_file_of_hits(hits_file) -> Path | None`, `rallies_of_hits(hits_file) -> DataFrame | None`
The detail table and the rally-bound table. The rally bounds are compared with
`vision.in_play.compare_spans` (`docs/vision/in_play.md`). The rallies file of a hits
file `hits_<range>.csv` is `rallies_<range>.csv` next to it (`rallies_of_hits` reads it).
A hits file with another name has none, and that run gets a "no rallies file" row.

```python
>>> from pathlib import Path
>>> from vision.label_scores import ScoredRun, detail_lines, rally_bound_lines, rallies_file_of_hits
>>> rallies_file_of_hits(Path("data/hits/v/hits_0_999.csv")).as_posix(), rallies_file_of_hits(Path("hits.csv"))
('data/hits/v/rallies_0_999.csv', None)
>>> detailed = labels.assign(detail=["serve_short", "lift", "clear", np.nan])
>>> with_detail = predictions.assign(type_detail_pred=["serve_long", "lift", "drop", "net"])
>>> detail_lines([ScoredRun("model", hits, with_detail)], detailed, 15)[2]
'| model | 3 | 3 | 1/3 = 33.3% |'
>>> detail_lines([ScoredRun("model", hits, predictions)], detailed, 15)[2], detail_lines([ScoredRun("model", hits, predictions)], labels, 15)
('| model | 3 | no detail predictions | |', ['no label has a type_detail; detail agreement not scored'])
>>> rallies = pd.DataFrame({"rally": [1, 2], "start_frame": [90, 880], "end_frame": [250, 1000], "hits": [3, 1]})
>>> rally_bound_lines([ScoredRun("model", hits, predictions, rallies)], [(80, 260), (880, 990)], 30.0)[2]
'| model | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0.17 | 0.33 | +0.17 | +0.00 |'

```

### `share(numerator, denominator) -> str`, `nearest_offset(frame, hits) -> int | None`
The table cells (`3/4 = 75.0%`), and the signed offset of the nearest detected hit
(inferred serves left out) for the missed list.

```python
>>> from vision.label_scores import nearest_offset, share
>>> share(3, 4), share(0, 0)
('3/4 = 75.0%', '0/0')
>>> nearest_offset(220, hits), nearest_offset(100, hits.iloc[:1])
(11, None)

```

### `headline_table(...)`, `serve_table(...)`, `non_serve_table(runs, labels, windows, fps)`, `source_lines(...)`, `breakdown_table(...)`, `type_confusion_table(...)`, `missed_table(...)`, `report_lines(...)`, `read_predictions(database_path, match_id)`, `main(arguments)`
These build the printed tables from the functions above. `read_predictions` reads
`frame_num, type_pred` of the match's own rows, read-only. A missing database is a
`FileNotFoundError`.
