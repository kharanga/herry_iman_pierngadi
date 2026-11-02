# vision/flight_types.py

Shot types from the feet alone against feet + shuttle flight. Both classifiers are
trained on hand-labelled club hits and type the own rows of another video. This is the
classifier part of the shuttle-flight experiment (`docs/results/shuttle_features.md`);
nothing here is used by the pipeline.

## Command

Write the labelled hits of each training video as hits and rallies files, for
`vision.features`:

```
.venv/bin/python -m vision.flight_types --train VIDEO LABELS DATABASE END [--train ...] --hits-out data/flight/hits
```

Then build each training video's own rows from them, using `vision.features` with
`--hits-dir data/flight/hits` on a copy of `data/scout.duckdb`. Then train and type:

```
.venv/bin/python -m vision.flight_types --train VIDEO LABELS DATABASE END [--train ...] --predict VIDEO DATABASE END OUT_DIR [--predict ...]
```

- `--train`: a labelled video, its labels CSV, the database holding its own rows built
  from the labelled hits, and the last shuttle-tracked frame.
- `--predict`: a video whose own rows in DATABASE (from detected hits) are typed. Each
  classifier writes a copy of the database with its predictions: `OUT_DIR/feet_only.duckdb`
  and `OUT_DIR/feet_flight.duckdb`. The copy clears this match's own `type_pred`,
  `type_detail_pred` and, when the table has it, `type_conf`, and sets the first two on
  the typed rows, in one transaction. Score the
  copies with `vision.label_scores` against the same hits file.
- Rows without both a hitter point and a landing point aren't typed, as in
  `shots.predict`.

Exit codes: `0`; `1` with `error: ...` for a missing file or column; `2` for bad
arguments.

## Features and classifiers

The feet features are the shipped classifier's (`shots.features`, the feet set). The
flight features come from the smooth detections between a hit and the next shot of its
rally (or 2 s later for the last shot). Distances are turned into metres using the
court's pixel size at the hitter's feet: the mean pixel length of 1 m across and 1 m
along the court there, from the calibration.

| Feature | Meaning |
|---|---|
| `flight_peak_m` | rise of the highest detection above the first one before the next hit |
| `flight_start_rising` | 1 if the shuttle moves up over the first 0.3 s |
| `flight_start_speed_m_per_s` | speed between the first and last detection in the first 0.3 s |
| `flight_seen_share` | detected frames over the frames until the next hit |

All four are missing when the hitter's feet aren't found within 5 frames, when there are
fewer than 3 detections in the first 0.3 s, when the first one comes later than 0.15 s
after the hit, or when the flight's track jumps (`vision.flight.has_jump`). The
classifiers are gradient-boosted trees (seed 0) over the 8 shot details, and they handle
missing values themselves. So a hit without flight is typed from its feet features alone.

```python
>>> import pandas as pd
>>> from vision.flight_types import flight_features, labels_as_hits
>>> seen = pd.DataFrame({"frame": [101, 102, 103, 104, 110, 120], "x": [500, 505, 510, 515, 540, 560], "y": [400, 390, 380, 372, 340, 330]})
>>> features = flight_features(100, 130, seen, scale=50.0, fps=60, frame_width=1280)
>>> round(features["flight_peak_m"], 2), features["flight_start_rising"], round(features["flight_start_speed_m_per_s"], 2)
(1.4, 1.0, 9.61)
>>> late = seen[seen["frame"] >= 110]
>>> pd.isna(flight_features(100, 130, late, 50.0, 60, 1280)["flight_peak_m"])
True
>>> labels = pd.DataFrame({"rally": [1, 1], "frame_num": [100, 160], "player": ["me", "?"], "type_group": ["serve", "lift"]})
>>> labels_as_hits(labels).values.tolist()
[[1, 100, 'me', 1.0, nan]]

```
