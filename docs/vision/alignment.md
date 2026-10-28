# vision/alignment.py

Checks that ShuttleSet shot labels line up with a match video. It picks 4 shots
spread across a set, seeks to each shot's hit frame (plus an offset), draws a red
circle at the labeled hit point (the hitter's feet) and a caption, and saves the
frames as JPEGs under `data/checks/<match_folder>/set<n>/`. If the circle sits at
the hitter's feet at the moment of contact, the offset is right.

The caption is `rally <r> shot <b> <shot type>`, with the shot type in English.
The set CSVs hold Chinese shot types, which OpenCV's fonts can't draw, so they are
translated with the shared map in `ingest/label_translations.py`
(`translate_label("type", ...)`, see `docs/analytics/label_translations.md`); this
module keeps no map of its own. A label missing from that map doesn't stop the
check: the caption says `unmapped shot type`, so the image is still written and the
gap is visible. An empty or missing label says `no shot type`. ShuttleSet's own
`未知球種` label is a real label and shows as `unknown`.

Per-frame helpers (`label_scale`, `scaled_point`, `draw_hit_marker`, ...) are pure:
numbers or arrays in, numbers or arrays out. Only `load_set_shots`, `open_video`,
`read_frame` and `write_alignment_checks` touch files.

## Command

```
.venv/bin/python -m vision.alignment <match_folder> <set_no> <video> [--offset N]
```

| Argument | Meaning |
|---|---|
| `match_folder` | ShuttleSet folder name under `data/CoachAI-Projects/ShuttleSet/set/` |
| `set_no` | 1, 2 or 3 |
| `video` | path to the match video |
| `--offset` | integer frames added to every label `frame_num` (default 0) |

Example (Axelsen vs Ginting, offset 0):

```
.venv/bin/python -m vision.alignment Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals 1 data/match.mp4
video fps=30.00, scale=1.000, offset=0
saved data/checks/Viktor_Axelsen_..._SemiFinals/set1/rally1_shot2_frame12359.jpg
...
```

Exit codes: `0` when at least one image was written; `1` with `error: ...` on stderr
when the set CSV or video is missing, the video can't be opened, the set has no
shots with `hit_x` and `hit_y` (`error: the set has no shots with hit_x and hit_y to check`),
or no frame could be read (`error: no frames could be read; check the video and offset`,
for example an offset past the end of the video); `2` for invalid arguments (for
example `set_no` 4). Frames that can't be read are reported and skipped.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `SHUTTLESET_SET_DIR` | `Path("data/CoachAI-Projects/ShuttleSet/set")` | ShuttleSet `set/` folder, relative to the working directory; default `set_dir` |
| `CHECKS_DIR` | `Path("data/checks")` | root folder for check images, relative to the working directory; default `checks_dir` |
| `LABEL_FRAME_WIDTH` | `1280` | frame width (pixels) that ShuttleSet label coordinates refer to |
| `MARKER_COLOR_BGR` | `(0, 0, 255)` | red, in OpenCV's BGR order, for the circle and caption |
| `MISSING_SHOT_TYPE_CAPTION` | `"no shot type"` | caption text when a shot has no type label |
| `UNMAPPED_SHOT_TYPE_CAPTION` | `"unmapped shot type"` | caption text when the label isn't in the shared English map |

```python
>>> from vision.alignment import CHECKS_DIR, LABEL_FRAME_WIDTH, MARKER_COLOR_BGR, SHUTTLESET_SET_DIR
>>> SHUTTLESET_SET_DIR.as_posix(), CHECKS_DIR.as_posix(), LABEL_FRAME_WIDTH, MARKER_COLOR_BGR
('data/CoachAI-Projects/ShuttleSet/set', 'data/checks', 1280, (0, 0, 255))

```

## Functions

### `set_csv_path(set_dir, match_folder, set_no) -> Path`
Path of a set's shot CSV. Inputs: `set_dir` (`Path`, ShuttleSet `set/` folder),
`match_folder` (`str`), `set_no` (`int`). No errors.

```python
>>> from pathlib import Path
>>> from vision.alignment import set_csv_path
>>> set_csv_path(Path("data/CoachAI-Projects/ShuttleSet/set"), "Some_Match", 2).as_posix()
'data/CoachAI-Projects/ShuttleSet/set/Some_Match/set2.csv'

```

### `check_image_dir(checks_dir, match_folder, set_no) -> Path`
Folder for one set's check images: `<checks_dir>/<match_folder>/set<n>`. No errors.

```python
>>> from vision.alignment import check_image_dir
>>> check_image_dir(Path("data/checks"), "Some_Match", 3).as_posix()
'data/checks/Some_Match/set3'

```

### `check_image_name(rally, ball_round, frame_num) -> str`
File name of one check image. All inputs `int` (label values, `frame_num` without offset).

```python
>>> from vision.alignment import check_image_name
>>> check_image_name(17, 8, 25747)
'rally17_shot8_frame25747.jpg'

```

### `label_scale(frame_width) -> float`
Factor from ShuttleSet label pixels (1280-wide frame) to the video's pixels.
Input: `frame_width` (`float`, video width in pixels). Output: `frame_width / 1280`.

```python
>>> from vision.alignment import label_scale
>>> label_scale(1920.0)
1.5
>>> label_scale(1280.0)
1.0

```

### `scaled_point(label_x, label_y, scale) -> tuple[int, int]`
Label pixel position scaled to video pixels and rounded to integers.

```python
>>> from vision.alignment import scaled_point
>>> scaled_point(740.0, 508.0, 1.5)
(1110, 762)

```

### `shots_with_hit_point(shots) -> DataFrame`
Rows of a set CSV that have both `hit_x` and `hit_y` (serves usually don't), index reset.
Input: `shots` (`pandas.DataFrame` with `hit_x`, `hit_y`). Errors: `KeyError` if a column is missing.

```python
>>> import pandas as pd
>>> from vision.alignment import shots_with_hit_point
>>> shots = pd.DataFrame({"rally": [1, 1, 1], "ball_round": [1, 2, 3],
...                       "hit_x": [None, 740.0, 475.0], "hit_y": [None, 508.0, 313.0]})
>>> shots_with_hit_point(shots)["ball_round"].tolist()
[2, 3]

```

### `spread_positions(row_count, pick_count) -> list[int]`
`pick_count` row positions spread evenly from first to last row, without duplicates.
Inputs: `row_count` (`int`, number of rows), `pick_count` (`int`, how many to pick).
Returns all positions if `row_count <= pick_count`, `[0]` if `pick_count` is 1,
and `[]` if `row_count` or `pick_count` is 0 or negative. No errors.

```python
>>> from vision.alignment import spread_positions
>>> spread_positions(10, 4)
[0, 3, 6, 9]
>>> spread_positions(3, 4)
[0, 1, 2]
>>> spread_positions(10, 1)
[0]
>>> spread_positions(0, 4)
[]

```

### `pick_spread_shots(shots, pick_count=4) -> DataFrame`
`shots_with_hit_point` followed by `spread_positions`: up to `pick_count` shots
spread across the set, first and last included (only the first when `pick_count` is 1).
Inputs: `shots` (`pandas.DataFrame` with `hit_x`, `hit_y`), `pick_count` (`int`).
Errors: `KeyError` if `hit_x` or `hit_y` is missing.

```python
>>> from vision.alignment import pick_spread_shots
>>> shots = pd.DataFrame({"ball_round": range(1, 12),
...                       "hit_x": [None] + [500.0] * 10, "hit_y": [None] + [400.0] * 10})
>>> pick_spread_shots(shots)["ball_round"].tolist()
[2, 5, 8, 11]
>>> pick_spread_shots(shots, pick_count=1)["ball_round"].tolist()
[2]

```

### `english_shot_type(raw_shot_type) -> str`
English shot type for a caption. Input: `raw_shot_type` (the CSV's `type` value:
a Chinese label as `str`, an already-English label, or `NaN`/`None` when empty).
Output: the English label from the shared map (surrounding spaces ignored);
`"unmapped shot type"` for a label the map doesn't know; `"no shot type"` for
`NaN`, `None` or blank text. Never raises, so one odd label can't stop a check run.

```python
>>> from vision.alignment import english_shot_type
>>> english_shot_type("發長球")
'long service'
>>> english_shot_type("過度切球"), english_shot_type("過渡切球")
('passive drop', 'passive drop')
>>> english_shot_type("未知球種")
'unknown'
>>> english_shot_type("smash")
'smash'
>>> english_shot_type("新球種")
'unmapped shot type'
>>> english_shot_type(float("nan")), english_shot_type(None), english_shot_type("  ")
('no shot type', 'no shot type', 'no shot type')

```

### `shot_caption(rally, ball_round, shot_type) -> str`
Text drawn on the image. `shot_type` should already be English (see `english_shot_type`).

```python
>>> from vision.alignment import shot_caption
>>> shot_caption(13, 10, "net shot")
'rally 13 shot 10 net shot'

```

### `draw_hit_marker(frame, point, caption) -> ndarray`
Copy of `frame` (`uint8` BGR array, H x W x 3) with a red circle (radius 12) at
`point` (`(x, y)` ints) and `caption` in red at the top left. The input frame is unchanged.

```python
>>> import numpy as np
>>> from vision.alignment import draw_hit_marker
>>> frame = np.zeros((100, 200, 3), dtype=np.uint8)
>>> marked = draw_hit_marker(frame, (100, 60), "rally 1 shot 2 lob")
>>> marked.shape, int(frame.sum())
((100, 200, 3), 0)
>>> marked[60, 112].tolist()
[0, 0, 255]

```

### `load_set_shots(set_dir, match_folder, set_no) -> DataFrame`
Reads one set CSV with pandas. Errors: `FileNotFoundError` naming the missing path
if the CSV doesn't exist.

```python
>>> import tempfile
>>> from vision.alignment import load_set_shots
>>> temporary_folder = tempfile.TemporaryDirectory()
>>> set_dir = Path(temporary_folder.name)
>>> (set_dir / "Some_Match").mkdir()
>>> _ = (set_dir / "Some_Match" / "set1.csv").write_text("rally,ball_round,frame_num,hit_x,hit_y,type\n1,1,120,640.0,500.0,lob\n")
>>> load_set_shots(set_dir, "Some_Match", 1)["frame_num"].tolist()
[120]
>>> load_set_shots(set_dir, "Some_Match", 2)  # doctest: +ELLIPSIS
Traceback (most recent call last):
FileNotFoundError: set CSV not found: .../Some_Match/set2.csv (check the match folder and set number)

```

### `open_video(video_path) -> cv2.VideoCapture`
Opens a video. Errors: `FileNotFoundError` if the file doesn't exist, `OSError`
if OpenCV can't open it. Caller releases the capture.

```python
>>> from vision.alignment import open_video
>>> open_video(set_dir / "missing.mp4")  # doctest: +ELLIPSIS
Traceback (most recent call last):
FileNotFoundError: video not found: .../missing.mp4

```

### `read_frame(capture, frame_index) -> ndarray | None`
Seeks to `frame_index` and returns that frame as a `uint8` BGR array (H x W x 3),
or `None` if it can't be read. Inputs: `capture` (`cv2.VideoCapture`, an open video,
for example from `open_video`), `frame_index` (`int`, 0-based frame number).
No errors: a negative index or an index past the end gives `None`.

```python
>>> import cv2
>>> from vision.alignment import read_frame
>>> video_path = set_dir / "tiny.avi"
>>> writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
>>> for frame_index in range(5):
...     _ = writer.write(np.full((48, 64, 3), frame_index * 50, dtype=np.uint8))
>>> writer.release()
>>> capture = open_video(video_path)
>>> frame = read_frame(capture, 3)
>>> frame.shape, abs(int(frame.mean()) - 150) <= 5
((48, 64, 3), True)
>>> read_frame(capture, 5) is None, read_frame(capture, -1) is None
(True, True)
>>> capture.release()

```

### `write_alignment_checks(match_folder, set_no, video_path, offset=0, set_dir=SHUTTLESET_SET_DIR, checks_dir=CHECKS_DIR) -> list[Path]`
Does the whole check for one set and returns the paths of the images written
(skipped frames are left out; a set with no shots with hit points gives `[]`).
Inputs: `match_folder` (`str`), `set_no` (`int`), `video_path` (`Path`), `offset`
(`int`, frames added to each label `frame_num`), `set_dir` and `checks_dir` (`Path`).
Prints the video fps, scale and offset, each saved path, and each frame it can't read.
Errors: those of `load_set_shots` and `open_video`.
The CSV is checked before the video, so a wrong match folder is reported first.

```python
>>> from vision.alignment import write_alignment_checks
>>> write_alignment_checks("Some_Match", 1, set_dir / "missing.mp4", set_dir=set_dir, checks_dir=set_dir / "checks")  # doctest: +ELLIPSIS
Traceback (most recent call last):
FileNotFoundError: video not found: .../missing.mp4
>>> written_paths = write_alignment_checks("Some_Match", 1, video_path, set_dir=set_dir, checks_dir=set_dir / "checks")  # doctest: +ELLIPSIS
video fps=25.00, scale=0.050, offset=0
could not read frame 120 (rally 1 shot 1)
>>> written_paths
[]
>>> written_paths = write_alignment_checks("Some_Match", 1, video_path, offset=-118, set_dir=set_dir, checks_dir=set_dir / "checks")  # doctest: +ELLIPSIS
video fps=25.00, scale=0.050, offset=-118
saved .../checks/Some_Match/set1/rally1_shot1_frame120.jpg
>>> [path.relative_to(set_dir).as_posix() for path in written_paths]
['checks/Some_Match/set1/rally1_shot1_frame120.jpg']

```

### `main(arguments) -> int`
Command entry point: parses `arguments` (`list[str]`, without the program name),
runs `write_alignment_checks` with the default `SHUTTLESET_SET_DIR` and `CHECKS_DIR`
(relative to the working directory), prints errors to stderr, and returns the exit
code described above. Errors: invalid arguments raise `SystemExit(2)` from argparse.

```python
>>> import contextlib
>>> import io
>>> from vision.alignment import main
>>> error_output = io.StringIO()
>>> with contextlib.redirect_stderr(error_output):
...     main(["No_Such_Match", "1", "data/match.mp4"])
1
>>> error_output.getvalue()  # doctest: +ELLIPSIS
'error: set CSV not found: .../No_Such_Match/set1.csv (check the match folder and set number)\n'
>>> with contextlib.redirect_stderr(io.StringIO()):
...     main(["Some_Match", "4", "data/match.mp4"])
Traceback (most recent call last):
SystemExit: 2
>>> temporary_folder.cleanup()

```
