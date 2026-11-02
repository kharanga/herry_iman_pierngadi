# vision/flight.py

How often the shuttle tracker (`vision/shuttle.py`) sees the shuttle after each labelled
hit: the coverage check of the shuttle-flight experiment
(`docs/results/shuttle_features.md`).

## Command

```
.venv/bin/python -m vision.flight <video> --labels <labels CSV> --end N [--start 0] [--shuttle-dir data/shuttle] [--sheet]
```

It reads every `shuttle_<start>_<end>.csv` stretch of the video that overlaps the range,
and the labels with their `_rallies.csv`. Labels after `--end` are left out. It prints:
- per hitter (all, `me`, `opp`): labelled hits, then the shuttle seen in the 0.5 s after
  the hit, usable, usable and smooth, and the share of rally frames with the shuttle seen;
- the same usable shares per labelled type.

With `--sheet` it also draws 20 random hits (seed 1). Each tile shows the frame a
quarter-second after the hit with the 0.5 s of detections drawn, marked "jump" when the
track jumps. It is saved to `data/checks/shuttle/<video id>_flight_sheet.jpg`.

A labels file that still holds unreviewed model labels (`source = model`, from `vision.prelabel`) is refused with `error: ... review them in python -m vision.label_ui --review first` (`vision.label.reviewed_rows_only`).

Exit codes: `0`; `1` with `error: ...` for a missing file or no shuttle track in the
range; `2` for bad arguments.

## Rules

- **A detection** is a frame with `visibility` 1 and `inpainted` 0: TrackNet saw the
  shuttle, and InpaintNet didn't fill it in.
- **Seen:** at least one detection in the 0.5 s after the hit (`SEEN_WINDOW_SECONDS`, in
  frames at the video's fps).
- **Usable:** detections in at least half of those frames (`USABLE_SHARE`).
- **Smooth:** usable, with no step between detections longer than 15% of the frame width
  per frame (`JUMP_SHARE_OF_WIDTH_PER_FRAME`). Such a step means the track hopped to a
  different shuttle.
- **Rally frames** are attributed to the player whose shot is in flight: the last
  labelled hitter at or before the frame.

```python
>>> import numpy as np, pandas as pd
>>> from vision.flight import frames_seen_after, has_jump, hit_coverage, in_flight_hitter
>>> frames_seen_after(100, np.array([101, 102, 103, 140]), window_frames=30), frames_seen_after(103, np.array([101, 102, 103, 140]), 30)
(3, 0)
>>> points = pd.DataFrame({"frame": [10, 11, 12], "x": [100.0, 110.0, 400.0], "y": [50.0, 52.0, 54.0]})
>>> has_jump(points.iloc[:2], frame_width=1280), has_jump(points, frame_width=1280)
(False, True)
>>> labels = pd.DataFrame({"rally": [1, 1], "frame_num": [100, 160], "player": ["me", "opp"], "type_group": ["serve", "lift"]})
>>> seen = pd.DataFrame({"frame": list(range(101, 120)), "x": 500.0, "y": 300.0})
>>> hit_coverage(labels, seen, fps=60, frame_width=1280)[["seen_frames", "usable", "smooth"]].values.tolist()
[[19, True, True], [0, False, False]]
>>> in_flight_hitter(np.array([90, 100, 150, 170]), labels).tolist()
['me', 'me', 'me', 'opp']

```

## Functions

`stretch_bounds`, `overlaps_range` and `shuttle_track` gather the stretch files.
`detected_points` keeps the detections. `rally_visibility`, `coverage_lines`,
`type_lines`, `sheet_tile`, `write_sheet` and `main` build the outputs.
