# vision/shuttle.py and vision/tracknet.py

Shuttle tracking: finds the shuttle in every frame of a video range and writes its
pixel position and whether it is visible. It uses the pretrained TrackNetV3 models
(Chen and Hsu, [github.com/qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3),
MIT license for code and checkpoints):

1. **TrackNet** looks at 8 consecutive frames plus a background image (the median
   of frames sampled across the range, so players and shuttle disappear from it)
   and outputs a heatmap per frame. Frames are resized to 512 x 288 RGB.
2. Each heatmap is thresholded at 0.5; the centre of the largest blob's bounding
   box, scaled back to the video size, is the shuttle position. No blob means not
   visible.
3. **InpaintNet** looks at 16 frames of positions at a time and fills gaps where
   the shuttle was lost mid-flight (for example behind a player), but not where it
   left the top of the picture.

With the default `weighted` ensemble every window of 8 frames is evaluated (one per
frame), and each frame's heatmap is the weighted average of the 8 windows that
contain it (weights 1, 2, 3, 4, 4, 3, 2, 1, so a frame counts more when it is near
the middle of a window). `nonoverlap` evaluates windows side by side, 8 times
faster; on the measured sample both were equally accurate. InpaintNet always uses the weighted ensemble.

This follows the upstream `predict.py`, reimplemented here; nothing from the
upstream repository is imported or executed. `vision/tracknet.py` defines the two
networks with the same layer names, so the published checkpoints load into them with
a strict check of every weight name and shape. Checkpoints are read with
`torch.load(..., weights_only=True)`, which loads tensors only and never runs code
from the file. Differences from upstream: at the first and last frames of a range,
where fewer than 8 windows overlap, the weights are renormalised (upstream averages
them uniformly); the background is the median of up to 300 frames sampled from the
tracked range (upstream: up to 1800 from the whole video); coordinates are pixel
positions in the video (upstream: the same).

Positions are video pixels, not court meters: the shuttle flies above the floor, so
the court calibration doesn't apply to it.

The video can contain broadcast cutaways; the model still outputs positions there,
which are meaningless. Use the view check in `vision/players.py`
(`split_frames_by_view`) to know which frames are the main camera view.

## Setup

```
./vision/fetch_tracknetv3.sh
```

Clones the upstream repository at a pinned commit into `data/third_party/TrackNetV3`
(reference and license only; not used at run time) and downloads the checkpoint
zip from the Google Drive link in its README with `curl` into
`data/models/tracknetv3/` (`TrackNet_best.pt`, 136 MB; `InpaintNet_best.pt`, 6 MB).
Both files are checked against SHA-256 hashes; a re-run skips the download when
they match.

Google Drive sometimes answers with an HTML page instead of the file (daily download
quota, or its virus-scan warning for large files), still with HTTP status 200. The
script checks the download is a valid zip before unzipping; if it isn't, it exits 1
with `error: the file downloaded from Google Drive is an HTML page, not a zip` and
instructions. To download by hand, open
https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view in a browser,
save the zip as `data/models/tracknetv3/TrackNetV3_ckpts.zip` and run the script again:
it uses that file instead of downloading, and deletes it after installing the
checkpoints. A zip without both checkpoints, or with checkpoints that don't match the
hashes, also exits 1. The checkpoints are extracted and verified in a temporary folder
inside `data/models/tracknetv3/` and only then moved into place, so a failed run never
leaves a partial or wrong checkpoint behind. Everything stays under the
gitignored `data/`. Only packages already in `requirements.txt` are needed (torch,
Pillow through Ultralytics, OpenCV, numpy); the upstream `requirements.txt` must not
be installed (it pins old torch, numpy and OpenCV).

## Command

```
.venv/bin/python -m vision.shuttle <video> (--start N --end N | --shuttleset-rally MATCH SET RALLY) [--padding N] [--ensemble weighted|nonoverlap] [--overlay]
```

| Argument | Meaning |
|---|---|
| `video` | path to the video |
| `--start`, `--end` | first and last frame (inclusive) |
| `--shuttleset-rally` | ShuttleSet match folder, set number, rally number: tracks from the rally's first hit to its last, plus `--padding` frames each side (start clamped to 0) |
| `--padding` | frames before and after a ShuttleSet rally (default 30) |
| `--ensemble` | `weighted` (default, about 1.6 s per frame on CPU) or `nonoverlap` (about 0.25 s per frame, 6 times faster) |
| `--overlay` | also write `data/checks/shuttle/<video stem>_shuttle_<start>_<end>.mp4`: the last 8 visible positions as a yellow trail and the current one as a red circle |

Output: `data/shuttle/<video stem>/shuttle_<start>_<end>.csv`, one row per frame:

| Column | Type | Meaning |
|---|---|---|
| `frame` | int | frame index in the video |
| `x`, `y` | int | shuttle position in video pixels; `0, 0` when not visible |
| `visibility` | int | 1 if the shuttle was found, else 0 |
| `inpainted` | int | 1 if the frame was in a gap that InpaintNet filled (its position comes from InpaintNet, not from the heatmap) |

Printed: `frames: N, visible: N, inpainted: N` and the saved paths. Exit codes: `0`
on success; `1` with `error: ...` on stderr for a missing video, checkpoint or set
CSV, an unknown rally, a negative start, a reversed range, or a range with no
readable frame; `2` for invalid arguments.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `CHECKPOINT_DIR` | `Path("data/models/tracknetv3")` | where the fetch script puts the checkpoints |
| `TRACKNET_CHECKPOINT`, `INPAINTNET_CHECKPOINT` | `TrackNet_best.pt`, `InpaintNet_best.pt` in `CHECKPOINT_DIR` | default checkpoints |
| `MODEL_INPUT_WIDTH`, `MODEL_INPUT_HEIGHT` | `512`, `288` | TrackNet input size |
| `HEATMAP_THRESHOLD` | `0.5` | heatmap value counted as shuttle |
| `INVISIBLE_COORDINATE_THRESHOLD` | `50 / sqrt(288² + 512²)` ≈ 0.085 | InpaintNet output with both normalised coordinates below this counts as not visible |
| `INPAINT_MIN_HEIGHT_SHARE` | `0.05` | gaps are only filled if the shuttle was lower than 5% of the frame height (`y > 0.05 * height`) on both sides, so it didn't leave the top of the picture |
| `BACKGROUND_SAMPLE_COUNT` | `300` | frames sampled for the median background |
| `WINDOW_BATCH_SIZE` | `8` | TrackNet windows per forward pass |
| `ENSEMBLE_MODES` | `("weighted", "nonoverlap")` | allowed ensemble modes |
| `SHUTTLE_DIR`, `CHECKS_DIR` | `Path("data/shuttle")`, `Path("data/checks/shuttle")` | output folders |
| `POSITION_COLUMNS` | `["frame", "x", "y", "visibility", "inpainted"]` | output columns |

```python
>>> from vision.shuttle import ENSEMBLE_MODES, INVISIBLE_COORDINATE_THRESHOLD, POSITION_COLUMNS
>>> ENSEMBLE_MODES, round(INVISIBLE_COORDINATE_THRESHOLD, 4)
(('weighted', 'nonoverlap'), 0.0851)
>>> POSITION_COLUMNS
['frame', 'x', 'y', 'visibility', 'inpainted']

```

## vision/tracknet.py

### `TrackNet(in_channels, out_channels)`
U-Net: input `(N, in_channels, 288, 512)` floats in 0..1, output `(N, out_channels, 288, 512)`
heatmaps in 0..1. With the published checkpoint `in_channels = 27` (background + 8
frames, 3 channels each) and `out_channels = 8` (one heatmap per frame).

### `InpaintNet()`
1-D U-Net over a sequence: `forward(coordinates, inpaint_mask)` with coordinates
`(N, L, 2)` normalised to 0..1 by the frame size and mask `(N, L, 1)`; returns `(N, L, 2)`.

```python
>>> import torch
>>> from vision.tracknet import InpaintNet, TrackNet
>>> TrackNet(27, 8)(torch.zeros(1, 27, 32, 64)).shape
torch.Size([1, 8, 32, 64])
>>> InpaintNet()(torch.zeros(2, 16, 2), torch.zeros(2, 16, 1)).shape
torch.Size([2, 16, 2])

```

### `tracknet_input_channels(sequence_length, background_mode) -> int`
Input channels for a checkpoint's settings. Only the `concat` background mode (the
published checkpoint's) is implemented. Errors: `ValueError` for any other mode.

```python
>>> from vision.tracknet import tracknet_input_channels
>>> tracknet_input_channels(8, "concat")
27
>>> tracknet_input_channels(8, "subtract")
Traceback (most recent call last):
ValueError: unsupported TrackNet background mode: 'subtract' (only 'concat' is implemented)

```

### `require_checkpoint_file(checkpoint_path)`, `read_checkpoint(checkpoint_path) -> dict`
`read_checkpoint` loads a checkpoint dict on the CPU with `weights_only=True`.
Errors: `FileNotFoundError` naming the fetch script if the file is missing.

```python
>>> from pathlib import Path
>>> from vision.tracknet import read_checkpoint
>>> read_checkpoint(Path("data/models/tracknetv3/missing.pt"))
Traceback (most recent call last):
FileNotFoundError: checkpoint not found: data/models/tracknetv3/missing.pt (run ./vision/fetch_tracknetv3.sh)

```

### `load_tracknet(checkpoint_path) -> (TrackNet, int)`, `load_inpaintnet(checkpoint_path) -> (InpaintNet, int)`
Build the network from the checkpoint's settings, load its weights (strict: every
name and shape must match) and switch to evaluation mode. Return the network and
its sequence length (8 for TrackNet, 16 for InpaintNet with the published
checkpoints). Errors: `FileNotFoundError` for a missing file, `RuntimeError` from
torch if the weights don't match the architecture.

## vision/shuttle.py: per-frame functions

### `model_input_image(frame_bgr) -> ndarray`
BGR video frame to the model's input: RGB, resized to 512 x 288 with Pillow (as the
upstream code does), channels first, `uint8` `(3, 288, 512)`.

```python
>>> import numpy as np
>>> from vision.shuttle import model_input_image
>>> frame = np.zeros((720, 1280, 3), dtype=np.uint8)
>>> frame[..., 2] = 200
>>> image = model_input_image(frame)
>>> image.shape, image.dtype, image[:, 0, 0].tolist()
((3, 288, 512), dtype('uint8'), [200, 0, 0])

```

### `median_background(frames_bgr) -> ndarray`
Per-pixel median of the frames, then `model_input_image`. Anything that moves (the
shuttle, the players) is absent from it.

```python
>>> from vision.shuttle import median_background
>>> empty_court = np.full((72, 128, 3), 60, dtype=np.uint8)
>>> with_player = empty_court.copy()
>>> with_player[10:30, 50:60] = 255
>>> median_background([empty_court, with_player, empty_court])[:, 50, 200].tolist()
[60, 60, 60]

```

### `tracknet_window_input(background, window_images) -> ndarray`
Background followed by the window's images along the channel axis, as `float32`
in 0..1: `(3 * (1 + L), 288, 512)`.

```python
>>> from vision.shuttle import tracknet_window_input
>>> background = np.zeros((3, 288, 512), dtype=np.uint8)
>>> window = [np.full((3, 288, 512), 255, dtype=np.uint8)] * 8
>>> stacked = tracknet_window_input(background, window)
>>> stacked.shape, stacked.dtype, float(stacked[0, 0, 0]), float(stacked[3, 0, 0])
((27, 288, 512), dtype('float32'), 0.0, 1.0)

```

### `ensemble_weights(sequence_length) -> ndarray`
Positional weights for averaging overlapping windows, normalised to sum 1.

```python
>>> from vision.shuttle import ensemble_weights
>>> ensemble_weights(8).tolist()
[0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05]

```

### `check_ensemble_mode(ensemble_mode)`, `window_starts(frame_count, sequence_length, ensemble_mode) -> list[int]`
First frame (0-based, within the range) of every window. `weighted`: every frame
that has a full window after it. `nonoverlap`: every `sequence_length` frames, plus a
last window ending at the last frame. A range shorter than a window gets one window
(padded, see below). Errors: `ValueError` for an unknown mode.

```python
>>> from vision.shuttle import window_starts
>>> len(window_starts(20, 8, "weighted")), window_starts(20, 8, "nonoverlap"), window_starts(5, 8, "weighted")
(13, [0, 8, 12], [0])
>>> window_starts(20, 8, "fast")
Traceback (most recent call last):
ValueError: ensemble mode must be one of ('weighted', 'nonoverlap'), got 'fast'

```

### `padded_window_frame_indices(start, sequence_length, frame_count) -> list[int]`
Frames of one window; past the last frame the last frame is repeated. A repeated
frame shows a still shuttle, which TrackNet doesn't detect, so ranges shorter than 8
frames lose most detections (on a 3-frame range of a rally, 2 of 3 frames came out
not visible); track at least 16 frames.

```python
>>> from vision.shuttle import padded_window_frame_indices
>>> padded_window_frame_indices(0, 8, 5)
[0, 1, 2, 3, 4, 4, 4, 4]

```

### `heatmap_position(heatmap, frame_width, frame_height) -> (int, int, int)`
Shuttle position from one heatmap: threshold at 0.5, take the largest blob (by
bounding-box area), its box centre, scaled from heatmap to video pixels. Returns
`(x, y, visibility)`; `(0, 0, 0)` when nothing passes the threshold.

```python
>>> from vision.shuttle import heatmap_position
>>> heatmap = np.zeros((288, 512))
>>> heatmap[100:104, 200:206] = 0.9
>>> heatmap[10, 10] = 0.8
>>> heatmap_position(heatmap, 1280, 720)
(507, 255, 1)
>>> heatmap_position(np.zeros((288, 512)), 1280, 720)
(0, 0, 0)

```

### `inpaint_mask(y_values, visibility, height_threshold) -> ndarray`
Marks the frames InpaintNet should fill: runs of invisible frames with the shuttle
lower in the picture than `height_threshold` pixels (`y > height_threshold`) on
both sides, or a run at the very start followed by such a frame. A gap after which
the shuttle reappears near the top edge is treated as the shuttle having left the
picture and is not filled.

```python
>>> from vision.shuttle import inpaint_mask
>>> inpaint_mask([300, 310, 0, 0, 0, 330, 340], [1, 1, 0, 0, 0, 1, 1], 36).tolist()
[0, 0, 1, 1, 1, 0, 0]
>>> inpaint_mask([300, 310, 0, 0, 0, 20, 30], [1, 1, 0, 0, 0, 1, 1], 36).tolist()
[0, 0, 0, 0, 0, 0, 0]

```

### `coordinates_below_threshold_to_zero(normalized_coordinates) -> ndarray`
Copy where points with both coordinates below `INVISIBLE_COORDINATE_THRESHOLD`
become `(0, 0)` (not visible).

```python
>>> from vision.shuttle import coordinates_below_threshold_to_zero
>>> coordinates_below_threshold_to_zero(np.array([[0.05, 0.08], [0.5, 0.25], [0.5, 0.0]])).tolist()
[[0.0, 0.0], [0.5, 0.25], [0.5, 0.0]]

```

### `weighted_frame_average(window_values, starts, frame_count, weights) -> ndarray`
Per-frame weighted average of overlapping windows: `window_values` is
`(windows, L, ...)`, window `k` covers frames `starts[k]` to `starts[k] + L - 1`;
frames past `frame_count` (padding) are ignored. Weights are renormalised for frames
covered by fewer windows.

```python
>>> from vision.shuttle import weighted_frame_average
>>> window_values = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
>>> weighted_frame_average(window_values, [0, 1], 4, np.array([1.0, 2.0, 1.0])).astype(float).round(3).tolist()
[0.0, 1.667, 3.333, 5.0]

```

### `HeatmapEnsemble(frame_count, sequence_length, ensemble_mode)`
Streaming version of `weighted_frame_average` for heatmaps, so a rally's heatmaps
never all sit in memory. `add_window(start, heatmaps)` adds one window's
`(L, H, W)` heatmaps; `pop_frames_before(frame_limit)` returns and forgets
`(frame, averaged heatmap)` for every frame below `frame_limit`, in order. Call it
with the next window's start once all earlier windows are added. `nonoverlap` uses
equal weights.

```python
>>> from vision.shuttle import HeatmapEnsemble
>>> ensemble = HeatmapEnsemble(frame_count=3, sequence_length=2, ensemble_mode="weighted")
>>> ensemble.add_window(0, np.array([[[0.0]], [[1.0]]]))
>>> ensemble.add_window(1, np.array([[[3.0]], [[5.0]]]))
>>> [(frame, heatmap.tolist()) for frame, heatmap in ensemble.pop_frames_before(2)]
[(0, [[0.0]]), (1, [[2.0]])]
>>> [(frame, heatmap.tolist()) for frame, heatmap in ensemble.pop_frames_before(3)]
[(2, [[5.0]])]

```

### `sampled_frame_indices(start_frame, end_frame, sample_count) -> list[int]`
Up to `sample_count` frames evenly spread over the range, both ends included.

```python
>>> from vision.shuttle import sampled_frame_indices
>>> sampled_frame_indices(100, 109, 4)
[100, 103, 106, 109]

```

### `visible_trail(positions, frame_index, trail_length) -> list[(int, int)]`, `draw_shuttle_trail(frame, trail_points) -> ndarray`
The visible positions among the last `trail_length` frames up to `frame_index`, and
a copy of the frame with them joined by a yellow line and the last one circled in red.

```python
>>> import pandas as pd
>>> from vision.shuttle import draw_shuttle_trail, visible_trail
>>> positions = pd.DataFrame({"frame": [10, 11, 12, 13], "x": [100, 110, 0, 130], "y": [50, 55, 0, 65],
...                           "visibility": [1, 1, 0, 1], "inpainted": 0})
>>> visible_trail(positions, 13, 3)
[(110, 55), (130, 65)]
>>> drawn = draw_shuttle_trail(np.zeros((100, 200, 3), dtype=np.uint8), visible_trail(positions, 13, 3))
>>> drawn[65, 138].tolist()
[0, 0, 255]

```

## vision/shuttle.py: running the model

### `read_model_input_images(video_path, start_frame, end_frame) -> (list[ndarray], (int, int))`
`model_input_image` for every frame of the range (stops at the end of the video)
and the video's `(width, height)`. Errors: those of `open_video` and `frames_in_range`
(`vision/players.py`): negative start, reversed range, no readable frame.

### `read_background(video_path, start_frame, end_frame) -> ndarray`
`median_background` of up to 300 frames sampled evenly from the range.

### `tracknet_positions(images, frame_size, background, tracknet, sequence_length, ensemble_mode) -> DataFrame`
Runs TrackNet over the windows, ensembles the heatmaps and converts each to a
position. Columns `frame` (0-based within the range), `x`, `y`, `visibility`.

### `inpainted_positions(positions, frame_width, frame_height, inpaintnet, sequence_length) -> DataFrame`
Builds the inpaint mask, runs InpaintNet on 16-frame windows (weighted ensemble),
keeps the TrackNet position where the mask is 0, and returns `POSITION_COLUMNS`.

### `track_shuttle(video_path, start_frame, end_frame, ensemble_mode="weighted", tracknet_checkpoint=..., inpaintnet_checkpoint=...) -> DataFrame`
The whole pipeline for one range; `frame` is the video's frame index. Checks the
range, the mode and the video before loading anything. Errors: `ValueError` (range,
mode), `FileNotFoundError` (video, checkpoints), `OSError` (video can't be opened).

### `write_overlay_video(video_path, positions, output_path, trail_length=8) -> Path`
Writes an MP4 (`mp4v`) of the frames in `positions` with `draw_shuttle_trail`.

### `positions_path(...)`, `overlay_path(...)`, `save_positions(positions, output_path)`

```python
>>> from vision.shuttle import overlay_path, positions_path
>>> positions_path(Path("data/match.mp4"), 14201, 14947).as_posix()
'data/shuttle/match/shuttle_14201_14947.csv'
>>> overlay_path(Path("data/match.mp4"), 14201, 14947).as_posix()
'data/checks/shuttle/match_shuttle_14201_14947.mp4'

```

### `frames_to_track(parsed) -> (int, int)`, `main(arguments) -> int`
Range from the command arguments (as in `vision/players.py`), and the command entry
point returning the exit code.

```python
>>> import argparse
>>> from vision.shuttle import frames_to_track
>>> frames_to_track(argparse.Namespace(shuttleset_rally=None, start=10, end=20, padding=30))
(10, 20)

```
