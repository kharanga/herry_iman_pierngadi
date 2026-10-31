# vision/overlay.py

Draws what the pipeline found back onto the video, from the saved files only (no
tracking): both players' boxes and feet, the current rally and shot count, each
detected hit flashed on the hitter, inferred serves marked apart, and frames where a
player is missing said so. It also draws both players' paths on a top-down court
diagram, one colour per rally. For checking a whole set by eye.

The per-frame drawing (`draw_tracking_overlay`) is a pure function: a frame and the
numbers for that frame in (`FrameState`), a frame out. `frame_state` builds those
numbers from the four tables (positions, boxes, hits, rallies). Only
`load_overlay_inputs`, `write_tracking_video` and `main` touch files.

## Command

```
.venv/bin/python -m vision.overlay <video> --start N --end N [--players-dir D] [--hits-dir D] [--output FILE] [--width 960] [--every 2] [--no-video] [--diagram]
```

| Argument | Meaning |
|---|---|
| `--start`, `--end` | the tracked range (its `positions_*.csv`, `boxes_*.csv`, `hits_*.csv` and `rallies_*.csv` must exist) |
| `--players-dir`, `--hits-dir` | roots of the tracking and hit files (defaults `data/players`, `data/hits`) |
| `--output` | the video file (default `data/checks/players/<video id>_tracking.mp4`) |
| `--width` | output width in pixels (default 960; the height follows) |
| `--every N` | write every Nth video frame (default 2: a 60 fps set becomes a 30 fps check video); the drawn numbers are those of the nearest tracked frame at or before it (tracks with a frame stride) |
| `--no-video` | skip the video (with `--diagram`) |
| `--diagram` | also write the paths diagram to `data/checks/players/<video id>_player_paths.png` |

The video is written with OpenCV, then re-encoded to H.264 with `ffmpeg` (`libx264`,
`yuv420p`, CRF 23) when `ffmpeg` is on the path, so QuickTime opens it; without
`ffmpeg` the `mp4v` file is kept. The raw file lives in a temporary directory that is
removed after every run, also when `ffmpeg` fails, so nothing but the output is left on
disk. Exit codes: `0`; `1` with `error: ...` for a missing
video or track file, or an `ffmpeg` failure.

In the picture: `me` blue, `opp` red, a filled dot at the foot point, the label
`<player> (<x>, <y>) m` from the positions file; top left `f<frame>  rally <n> of
<count>  shot <k>` (or `between rallies`), and `not detected: me` / `opp` in orange
when a player has no box on that frame. For `HIT_FLASH_FRAMES` (10) frames from each
hit the hitter's box is thick green with `hit f<frame>` under it; an inferred serve
(score NaN in the hits file) is thick magenta with `serve (inferred) f<frame>`.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `PLAYER_COLOR_BGR` | `me` blue, `opp` red | box colours |
| `HIT_FLASH_COLOR_BGR`, `SERVE_FLASH_COLOR_BGR` | green, magenta | the flashes |
| `HIT_FLASH_FRAMES` | `10` | frames a hit stays flashed |
| `DEFAULT_OUTPUT_WIDTH_PX`, `DEFAULT_OUTPUT_STRIDE` | `960`, `2` | `--width`, `--every` defaults |
| `DIAGRAM_PX_PER_M`, `DIAGRAM_MARGIN_PX` | `60`, `40` | the court diagram's scale and border |
| `RALLY_COLORS_BGR` | 8 colours | cycled per rally on the diagram |

## Classes

### `HitFlash(hitter, frame, inferred)`, `FrameState(frame_index, boxes, positions, rally_label, shot_count, flash)`
The numbers for one frame: the box row and position row per player (`None` when
missing), the rally label, the shots so far in the rally (`None` between rallies) and
the active flash.

## Functions

### `rows_by_player(table, frame_index) -> dict`, `rally_at(frame_index, rallies)`, `hit_flash_at(frame_index, hits, flash_frames=10) -> HitFlash | None`, `frame_state(frame_index, positions, boxes, hits, rallies, flash_frames=10) -> FrameState`

```python
>>> import numpy as np, pandas as pd
>>> from vision.overlay import FrameState, HitFlash, frame_state, hit_flash_at, rally_at, rows_by_player
>>> positions = pd.DataFrame({"frame": [100, 100, 101], "player": ["me", "opp", "me"], "x_m": [3.0, 2.5, 3.1], "y_m": [2.0, 10.0, 2.1], "track_id": 1, "confidence": 0.9})
>>> boxes = pd.DataFrame({"frame": [100, 100, 101], "player": ["me", "opp", "me"], "left": [600, 620, 605], "top": [300, 200, 300], "right": [700, 660, 705], "bottom": [500, 300, 500]})
>>> hits = pd.DataFrame({"rally": [1, 1, 1], "frame": [77, 100, 130], "hitter": ["opp", "me", "opp"], "score": [np.nan, 0.9, 0.8], "reach": np.nan})
>>> rallies = pd.DataFrame({"rally": [1], "start_frame": [77], "end_frame": [160], "hits": [3]})
>>> rows_by_player(boxes, 101)["opp"] is None, int(rows_by_player(boxes, 101)["me"]["left"])
(True, 605)
>>> int(rally_at(100, rallies)["rally"]), rally_at(200, rallies) is None
(1, True)
>>> hit_flash_at(105, hits), hit_flash_at(80, hits), hit_flash_at(115, hits) is None
(HitFlash(hitter='me', frame=100, inferred=False), HitFlash(hitter='opp', frame=77, inferred=True), True)
>>> state = frame_state(101, positions, boxes, hits, rallies)
>>> state.rally_label, state.shot_count, state.flash, state.boxes["opp"] is None
('rally 1 of 1', 2, HitFlash(hitter='me', frame=100, inferred=False), True)
>>> frame_state(200, positions, boxes, hits, rallies).rally_label, frame_state(200, positions, boxes, hits, rallies).shot_count
('between rallies', None)

```

### `put_text(frame, text, origin, scale=0.7, color=TEXT_COLOR_BGR) -> None`, `scaled_box(box_row, scale) -> (int, int, int, int)`, `draw_tracking_overlay(frame, state, scale=1.0) -> ndarray`
`draw_tracking_overlay` draws a `FrameState` on a frame; `scale` maps the box pixels of
the original video onto a resized frame. The input frame is not changed.

```python
>>> from vision.overlay import draw_tracking_overlay, scaled_box
>>> scaled_box(boxes.iloc[0], 0.5)
(300, 150, 350, 250)
>>> frame = np.zeros((540, 960, 3), dtype=np.uint8)
>>> drawn = draw_tracking_overlay(frame, state, scale=0.5)
>>> drawn.shape == frame.shape, int(frame.sum()), bool(drawn[150:250, 300:352].any())
(True, 0, True)

```

### `load_overlay_inputs(video_path, start_frame, end_frame, players_dir=PLAYERS_DIR, hits_dir=HITS_DIR) -> (positions, boxes, hits, rallies)`, `tracking_video_path(video_path, checks_dir=CHECKS_DIR) -> Path`, `encode_h264(raw_path, output_path) -> Path`, `write_tracking_video(video_path, start_frame, end_frame, output_path, players_dir=PLAYERS_DIR, hits_dir=HITS_DIR, output_width=960, output_stride=2, flash_frames=10) -> Path`
The file side: read the four tables (`FileNotFoundError` for a missing one), the default
output name, the `ffmpeg` re-encode, and the loop over the video.

```python
>>> from pathlib import Path
>>> from vision.overlay import tracking_video_path
>>> tracking_video_path(Path("data/recordings/stand_in/set1.mp4")).as_posix()
'data/checks/players/stand_in_set1_tracking.mp4'

```

### `court_to_diagram(points, px_per_m=60, margin=40) -> ndarray`, `blank_court_diagram(px_per_m=60, margin=40) -> ndarray`, `rally_of_frame(frame_index, rallies) -> int | None`, `court_paths_diagram(positions, rallies, px_per_m=60, margin=40) -> ndarray`, `paths_diagram_path(video_path, checks_dir=CHECKS_DIR) -> Path`
Court metres to diagram pixels (the near baseline at the bottom), the empty court with
its lines, and both players' paths within each rally in that rally's colour.

```python
>>> from vision.overlay import blank_court_diagram, court_paths_diagram, court_to_diagram, paths_diagram_path, rally_of_frame
>>> court_to_diagram([[0.0, 0.0], [6.1, 13.4]]).tolist()
[[40.0, 844.0], [406.0, 40.0]]
>>> blank_court_diagram().shape
(884, 446, 3)
>>> rally_of_frame(100, rallies), rally_of_frame(500, rallies)
(1, None)
>>> court_paths_diagram(positions, rallies).shape
(884, 446, 3)
>>> paths_diagram_path(Path("data/recordings/stand_in/set1.mp4")).as_posix()
'data/checks/players/stand_in_set1_player_paths.png'

```

### `main(arguments) -> int`
Command entry point (see Command).
