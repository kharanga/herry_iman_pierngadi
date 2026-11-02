# vision/halve_fps.py

Carries a video's hand labels and court calibration over to a copy at half the frame rate,
made by keeping every second frame: frame k of the copy is frame 2k of the original.

Making the copy itself is one ffmpeg command. It keeps the even frames by decode order and
re-times them to half the rate:

```
ffmpeg -i <video>.mp4 -an -vf "select=not(mod(n\,2)),setpts=N/(30)/TB" -fps_mode passthrough \
  -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p <video>_30.mp4
```

Use `30000/1001` in place of `30` for a 59.94 fps original. Frame k of the copy is the
original's frame 2k counted in decode order. On an irregular original, OpenCV's seeking
can disagree with that order by a frame or two (see "Irregular originals" below).

## Command

```
.venv/bin/python -m vision.halve_fps <video> <halved video> [--labels LABELS OUTPUT]
```

- The calibration is copied from `data/calibration/<video id>/court.json` to the halved
  video's folder, with `frame_index` halved (rounded down) and the overlay image copied.
  The pixel homography is unchanged, because the copy keeps the frame size.
- With `--labels`, the labels CSV and its `_rallies.csv` are halved into OUTPUT and
  OUTPUT's rallies file. Nothing is written if two hits land on one frame, two rallies
  overlap or a hit falls outside its rally.
- Originals are never changed.

Exit codes: `0`; `1` with `error: ...` for a missing file, colliding hits, overlapping
rallies or a hit outside its rally; `2` for bad arguments.

## Rules

- **Hit frames:** frame f becomes f / 2, rounded half up, so an odd f becomes (f + 1) / 2.
  An odd frame sits between two kept frames, so the later one is used. The hit is at most
  half a 30 fps frame (17 ms) from where it was labelled.
- **Rally bounds:** starts round down and ends round up, so a rally only grows and a hit
  inside its rally stays inside. Two back-to-back rallies (one ending on an odd frame,
  the next starting on the frame after) would then share a frame. So an end that reaches
  the next rally's start is pulled back to the frame before it (`ends_before_next_start`).
  Overlapping rallies are refused like colliding hits.

```python
>>> import pandas as pd
>>> from vision.halve_fps import halved_hit_frames, halved_rally_ends, halved_rally_starts, label_problems
>>> halved_hit_frames([100, 101, 102]).tolist()
[50, 51, 51]
>>> halved_rally_starts([99, 100]).tolist(), halved_rally_ends([99, 100]).tolist()
([49, 50], [50, 50])
>>> labels = pd.DataFrame({"rally": [1, 1, 2], "frame_num": [51, 51, 90]})
>>> rallies = pd.DataFrame({"rally": [1, 2], "start_frame": [40, 95], "end_frame": [60, 120]})
>>> label_problems(labels, rallies)
['hits collide at frame 51', 'hit at frame 90 falls outside rally 2']
>>> from vision.halve_fps import halved_rallies, overlapping_rallies
>>> back_to_back = pd.DataFrame({"rally": [1, 2, 3], "start_frame": [11, 34, 511], "end_frame": [33, 510, 1268]})
>>> halved_rallies(back_to_back)[["start_frame", "end_frame"]].values.tolist()
[[5, 16], [17, 254], [255, 634]]
>>> overlapping_rallies(pd.DataFrame({"rally": [1, 2], "start_frame": [5, 17], "end_frame": [17, 254]}))
[(1, 2)]

```

## Irregular originals

`xBClOgrP1Hs` (59.94 fps) has 38,179 frames when read in order but reports 38,183, and
OpenCV's seeking lands up to 2 frames past in-order reading later in the video. Its
halved copy is regular, with frame k = in-order frame 2k. Labels made on such an
original need checking against the video before they are halved.
