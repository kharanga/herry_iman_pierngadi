# vision/review_report.py

How often the model was right on a game the user reviewed in `vision.label_ui --review`.
Results: `docs/results/club_review.md`.

## Command

```
.venv/bin/python -m vision.review_report <video> --labels LABELS
```

It reads the reviewed labels file, its `_rejected.csv` and its `_rallies.csv`. The video
is used only for its fps. Read-only. It prints Markdown:
- **Status:** model hits reviewed (kept plus rejected), and how many were kept exactly
  (frame, type and hitter), type-corrected, hitter-corrected, moved, rejected and added.
  Unreviewed rows (`source = model`) are counted and left out of everything else.
- **Shot type:** the model's group against the user's, on kept model hits.
- **Hitter:** hitter right on kept model hits, per model hitter.
- **Moved hits:** the lag, model frame minus the user's frame (positive = the model came
  later), listed and as median and mean, overall, per hitter and per type.
- **Rejected and added by type:** with re-timed pairs separated out. A re-timed pair is
  a rejected model hit with a user-added hit within 1 s (`RETIMED_REACH_SECONDS`), in any
  rally, nearest first, each added hit used once. That gives false hits (rejected, not
  re-timed) and missed hits (added, not re-timed). A sensitivity line repeats this at
  1.5 s and 2 s.

Exit codes: `0`; `1` with `error: ...` for a missing file or column; `2` for bad
arguments.

## Functions

```python
>>> import pandas as pd
>>> from vision.review_report import retimed_rejections, split_rows
>>> kept = pd.DataFrame({"rally": [1, 1, 1, 2], "frame_num": [100, 140, 200, 400], "player": ["opp", "me", "opp", "me"], "type_group": ["serve", "net", "lift", "clear"], "type_detail": ["serve_short", "net", "lift", "clear"],
...                      "source": ["confirmed", "model", "hand", "hand"], "model_frame": [100, 140, None, None], "model_type_detail": ["serve_short", "net", "", ""], "model_player": ["opp", "me", "", ""]})
>>> reviewed, added, unreviewed = split_rows(kept)
>>> len(reviewed), len(added), len(unreviewed)
(1, 2, 1)
>>> rejected = pd.DataFrame({"rally": [1, 3], "frame_num": [190, 900], "player": ["opp", "me"], "model_type_detail": ["drop", "net"]})
>>> retimed_rejections(rejected, added, reach_frames=30).values.tolist()
[[190, 200, 'drop', 'lift', True]]

```

`model_group` maps a model detail to its group (`unknown` when empty);
`status_lines`, `confusion_lines`, `hitter_lines`, `lag_lines`, `by_type_lines` and
`report_lines` build the tables.
