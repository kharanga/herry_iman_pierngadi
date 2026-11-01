# vision/track_check.py

Counts the player-tracking errors that matter on club footage, and checks the tracks
against hand labels. It reads only the saved positions and boxes, so the same command
gives the numbers before and after a change to player choice
(`docs/results/club_tracks_clean.md`).

## Command

```
.venv/bin/python -m vision.track_check <video> --end N [--start 0] [--labels CSV | --rallies CSV] [--players-dir D] [--sheets [--tag before]]
```

| Option | What it does |
|---|---|
| `--end`, `--start` | the tracked range whose `positions_<start>_<end>.csv` and `boxes_` files are read |
| `--labels` | hand labels (`vision.label_ui`): their hits for the label check, and their `_rallies.csv` for the gaps |
| `--rallies` | a rallies CSV (`start_frame`, `end_frame`) for the gaps instead, e.g. the detector's `data/hits/<id>/rallies_0_N.csv` |
| `--sheets` | draw up to 12 sample frames per error type, spread over the video, to `data/checks/track_check/<video id>_<type>[_<tag>].jpg` |

It prints a Markdown table and exits `0`; `1` with `error: ...` for a missing video or
track file; `2` for bad arguments.

## What is counted

| Error | Rule (constants) |
|---|---|
| still stand-in | a player's position more than 0.3 m outside the court lines (`OUTSIDE_MARGIN_M`) and slower than 0.3 m/s (`STILL_SPEED_M_PER_S`, median over 5 frames) for 2 s or more (`STILL_SECONDS`): a line judge, umpire or spectator standing in for a player who left the view. Counted in runs and player-frames, and how many of those frames fall inside rallies |
| passer-by | the player's box touches the image's side edge (`SIDE_EDGE_PX`, 4 px) and is more than 1.6 times that player's median box height (`PASSER_HEIGHT_RATIO`): someone walking right in front of the camera |
| swap | `me` on the far half while `opp` is on the near half |
| gap inside rallies | a player missing for 0.5 s or more (`GAP_SECONDS`) inside a rally |
| label check | for each labelled hit: the hitter tracked at that frame, on its own half, and faster than 1 m/s (`MOVING_SPEED_M_PER_S`) somewhere within 1 s of the hit (`HIT_WINDOW_SECONDS`) |

A still stand-in can't be told apart, from the saved rows alone, from a real player who
stopped outside the lines. The sample sheets show which it is.

## Functions

### `outside_lines(x_m, y_m, margin=0.3) -> ndarray`, `runs(mask) -> list[(int, int)]`, `speeds(rows, fps, smoothing=5) -> ndarray`

```python
>>> import numpy as np, pandas as pd
>>> from vision.track_check import outside_lines, runs, speeds
>>> outside_lines(np.array([3.0, 6.7, -0.1]), np.array([2.0, 12.8, 5.0])).tolist()
[False, True, False]
>>> runs(np.array([0, 1, 1, 0, 1], dtype=bool))
[(1, 2), (4, 4)]
>>> rows = pd.DataFrame({"frame": [0, 1, 2, 4], "x_m": [0.0, 0.1, 0.2, 0.4], "y_m": 0.0})
>>> speeds(rows, 10.0, smoothing=1).round(2).tolist()
[1.0, 1.0, 1.0, 1.0]

```

### `player_rows(positions, boxes, player)`, `still_stand_in_runs(rows, fps)`, `passer_by_runs(rows, frame_width)`, `swap_frames(positions)`, `rally_gap_runs(positions, rallies, fps)`, `diagnosis(positions, boxes, rallies, fps, frame_width) -> dict`

```python
>>> from vision.track_check import diagnosis, still_stand_in_runs, swap_frames
>>> frames = np.arange(100)
>>> positions = pd.concat([
...     pd.DataFrame({"frame": frames, "player": "me", "x_m": 3.0, "y_m": 2.0 + 0.05 * (frames % 20)}),
...     pd.DataFrame({"frame": frames, "player": "opp", "x_m": 6.8, "y_m": 12.8}),
... ])
>>> boxes = positions[["frame", "player"]].assign(left=100.0, top=100.0, right=150.0, bottom=300.0)
>>> found = diagnosis(positions, boxes, pd.DataFrame({"start_frame": [10], "end_frame": [60]}), 25.0, 1280.0)
>>> found["still_stand_in"], found["passer_by"], len(found["swap"]), found["rally_gap"]
([('opp', 0, 99)], [], 0, [])
>>> swap_frames(pd.DataFrame({"frame": [5, 5], "player": ["me", "opp"], "x_m": 3.0, "y_m": [11.0, 2.0]}))
[5]

```

### `label_agreement(positions, boxes, labels, fps) -> dict`

```python
>>> from vision.track_check import label_agreement
>>> labels = pd.DataFrame({"frame_num": [30, 50], "player": ["me", "opp"]})
>>> label_agreement(positions, boxes, labels, 25.0)
{'labelled': 2, 'hitter_tracked': 2, 'right_half': 2, 'moving': 1, 'right_half_and_moving': 1}

```

### `frames_inside(found, rallies)`, `run_frames(found)`, `rally_frames(rallies)`, `summary_lines(...)`, `sample_frames(found, error_type, count=12)`, `write_sheet(...)`, `sheet_path(...)`, `rallies_for(labels_file, rallies_file)`, `main(arguments)`
The table lines, the sample picks (spread evenly over the found runs or frames), the
sheets, and the command.
