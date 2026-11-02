# vision/prelabel.py

Writes the model's hits and shot types as a labels file, for the user to confirm or
correct in `vision.label_ui --review` (`docs/vision/label_ui.md`, "Review mode").

## Command

```
.venv/bin/python -m vision.prelabel <video> --hits HITS_CSV --database DATABASE --match-id ID [--labels LABELS] [--force]
```

- `--hits`: the detector's hits file. Its `rallies_*.csv` sits next to it.
- `--database`: holds the video's own rows built from those hits and typed:
  `type_pred`, `type_detail_pred`, and `type_conf` as the score (for example the
  `average.duckdb` of `vision.club_classifier`).
- **The labels file** (default `data/labels/<video id>.csv`) gets one row per hit:
  - the hit's rally, frame and hitter;
  - the predicted group and detail, with `source = model`;
  - `score`, `model_frame` (= the frame), `model_type_detail` (= the detail) and
    `model_player` (= the hitter).
  A hit with no prediction (no landing point: the last shot of a rally) gets type
  `unknown` and no score.
- **The rallies file** gets the detector's rallies that hold hits, with
  `bounds_checked = False`.
- **Overwriting:** an existing labels file is never overwritten without `--force`,
  because it may hold the user's review.

Exit codes: `0`; `1` with `error: ...` for an existing labels file without `--force`, a
missing file or a missing column; `2` for bad arguments.

## Functions

```python
>>> import pandas as pd
>>> from vision.prelabel import model_labels, model_rallies
>>> hits = pd.DataFrame({"rally": [1, 1, 1], "frame": [100, 130, 160], "hitter": ["opp", "me", "opp"], "score": [float("nan"), 0.9, 0.8], "reach": float("nan")})
>>> typed = pd.DataFrame({"frame_num": [100, 130], "type_pred": ["serve", "net"], "type_detail_pred": ["serve_short", "net"], "type_conf": [0.9, 0.6]})
>>> labels = model_labels(hits, typed)
>>> labels[["frame_num", "player", "type_detail", "source", "score", "model_frame", "model_type_detail"]].values.tolist()
[[100, 'opp', 'serve_short', 'model', 0.9, 100, 'serve_short'], [130, 'me', 'net', 'model', 0.6, 130, 'net'], [160, 'opp', 'unknown', 'model', nan, 160, 'unknown']]
>>> model_rallies(pd.DataFrame({"rally": [1, 2], "start_frame": [90, 400], "end_frame": [200, 500], "hits": [3, 0]}), labels).values.tolist()
[[1, 90, 200, False]]

```
