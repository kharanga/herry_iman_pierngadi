# vision/label_ui.py

The hit labelling tool of `vision/label.py` in the browser: a small local server on the
Mac and one page. It uses the same state functions (`LabelState`, `handle_key`, ...) and
writes the same two files, so labels made in the browser load in the OpenCV window and
the other way round, and both score with `vision.hits --evaluate-labels`. The tool never
runs a model.

## Start

```
.venv/bin/python -m vision.label_ui <video> --match-id <id> [--labels FILE] [--port 8766] [--no-browser] [--players-dir D] [--hits-dir D] [--blank] [--segment-seconds 60]
```

Without `--blank`, it needs the video's hits and rallies files for the whole video
(`python -m vision.hits <video> --start 0 --end <last frame>`). It draws the tracking overlay when the player
files exist, and the `ffmpeg` binary on the path for the rally clips. It serves `http://127.0.0.1:8766/` (local only) and opens it in the default
browser; ctrl-c (or `kill`, SIGTERM) stops it and removes the clip folder. `--port 0` picks a free port. Exit codes: `0`; `1` with
`error: ...` for a missing video or hits file, or a `--segment-seconds` that isn't positive.

### Blank mode (`--blank`)

For labelling a whole video by hand, with nothing the detector inferred on screen:

- **Nothing inferred:** no detected rallies, no hit proposals, no inferred serves, and no
  hit flash or detector rally text in the overlay. No hits file is read.
- **Tracking stays:** the tracking boxes, `me`/`opp` and the foot dots stay, because
  they are measured, not inferred.
- **Segments:** the video is cut into back-to-back **segments** of `--segment-seconds`
  (default 60). `[` / `]` and the prev/next buttons move between segments. The timeline,
  the looping clip on the right and the hit list cover the current segment, as they
  cover a rally otherwise. Stepping with `,` `.` `<` `>` past a segment's edge moves into
  the next or previous segment. Segments are never written to a file.
  `--segment-seconds` also turns segments on without `--blank`.
- **Rallies:** you mark them yourself:
  - `b` (**rally starts here**) starts a rally at the current frame. The new rally runs
    to the frame before the next rally, or to the end of the video, until you end it.
    Pressed inside a rally, `b` moves that rally's start, unless the rally already has
    hits before the frame. In that case `b` splits the rally there, and the frame
    starts the next point. If the next rally follows right after and its start isn't
    checked yet, the boundary simply moves to the frame (this is how `b` fixes old
    back-to-back rallies). Pressed less than 3 s before an unchecked rally
    (`UNCHECKED_START_REACH_SECONDS`), `b` moves that rally's start back to the frame.
  - `e` (**rally ends here**) ends the rally holding the current frame there. If the
    rally has hits after the frame, `e` splits it, so those hits keep a rally. If the
    frame is in the head of a rally (no hits yet, and its start not checked) right after
    a rally whose end isn't checked, `e` moves that earlier rally's end to the frame
    instead. Pressed after a
    rally, `e` extends that rally to the current frame.
  - A rally counts as **checked** once both its start (`b`) and its end (`e`) have been
    pressed. Unchecked rallies are a brown block on the timeline, and the header counts
    them. Moving a bound with `r` unchecks that side.
  - Rallies never overlap, and they may cross segment edges.
  - `r` and `-` act only on the rally holding the current frame. `-` leaves the frame
    where it is. `u` undoes any of these.
  - The "accept proposal (a)", "rally start/end here (r)" and "add rally (+)" buttons are
    hidden, because `b` / `e` replace them and there are no proposals. The keys still
    work.
  - Your rallies are the dark green blocks on the timeline.
- **Labels:** a hit's rally is your rally that holds it. Mark the rally's start (the
  serve's contact, `docs/vision/label.md`) before its hits. A hit outside every rally is
  dropped at the next rally change.
- **Files:** labels and rallies keep video frame numbers and the same CSV columns, so one
  labels file and one rallies file hold the whole video.

The status line shows `BLANK`, the clock time, the segment, the rally holding the frame
and the number of rallies. The header adds how many serves still need short / long and
how many rallies are unchecked.

### Relabelling an older file

Stand-in set 1 was labelled before the serve lengths and the checked rally bounds
existed. To bring it to the same standard, keeping its 153 hits:

```
.venv/bin/python -m vision.label_ui data/recordings/stand_in/set1.mp4 --match-id stand_in_set1 --blank --port 8768
```

1. The header shows "24 serves need short/long, 26 rallies unchecked".
2. For each rally:
   - step to the serve's contact and press `b`;
   - step to the landing and press `e`.

   The old bounds move to those frames, and the rally turns from brown to green. A
   detected rally that holds two points splits where you press `b` at the second serve.
3. On each `serve ?` hit, press `h`, the hitter, then `s` and `s` or `l`.
4. Every change is saved at once. Rallies you haven't checked keep `bounds_checked`
   false, so they aren't used as in-play training data. Screenshot: `data/checks/label_ui/blank_segments.png` (Chong
v Ding, segment 2).

## The page

- **Left:** the current frame as a JPEG from the server (960 px, tracking boxes with
  `me`/`opp` and the foot dots), under it the rally **timeline** (yellow ticks = the
  detector's proposals, green bars = labelled hits, white line = the current frame;
  click anywhere on it to jump there), and the buttons.
- **Right:** the current rally as a looping, muted HTML5 video at 0.25x / 0.5x (default) /
  1x, with a strip under it showing its playhead (blue dot) and the left panel's frame
  (white line); then the list of the rally's labelled hits (frame, hitter, type, source):
  click a row to jump to it, `delete` to remove it.
- **Top:** a **frame box**, then the frame and rally, saved / unsaved, progress ("12 of
  36 rallies labelled, 97 hits"), and the pending hit prompt.
  - Type a frame number in the box and press Enter or `go`. The page sends the same
    `jump` action as a click on the timeline (`jump_to_frame`, so `at_frame`). A frame
    inside a rally selects that rally; in blank mode the segment follows.
  - Anything that isn't a whole number, or a frame past the video's last, gets a short
    message next to the box ("type a whole frame number", "frames run 0-N"), and nothing
    is sent.
  - While the box has the focus, the key shortcuts are off: the box keeps its own key
    presses, and the page ignores keys typed in any input. Esc leaves the box, and so
    does a jump.

Every action is a button and a key; the keys are the OpenCV tool's (`docs/vision/label.md`):

| Key | Button | Action |
|---|---|---|
| left / right arrow, `,` `.` | -1, +1 | one frame |
| shift+arrow, `<` `>` | -10, +10 | ten frames |
| `[` `]` | prev rally, next rally (prev / next segment with segments) | rally, or segment |
| `b`, `e` | rally starts here, rally ends here | mark your own rally's start and end (blank mode) |
| `h`, then `m`/`o`, then `c d k n l v ?` | hit, me, opp, clear ... unknown | label a hit at the current frame: clear, drop, smash, net, lift, drive, unknown |
| `h`, then `m`/`o`, then `s` and `s`/`l` | serve short (s s), serve long (s l) | a serve: after `s` a pink banner asks **SERVE, press s for SHORT or l for LONG**; inside it `l` means long, not lift. Each serve button sends both keys |
| `a` | accept proposal | the nearest proposal (within 60 frames) with its hitter, then a type |
| `x`, `u` | delete nearest, undo | |
| `r`, `+`, `-` | rally start/end here, add rally, remove rally | rally bounds |
| esc | cancel | cancel a pending hit |
| space, `1` `2` `3` | speed buttons | pause / play the loop, loop speed |

Only the keys of this table are sent to the server (`ACTION_KEYS`); any other key does
nothing, and a key pressed with cmd, ctrl or alt is left to the browser (cmd+R reloads the
page, it doesn't set a rally bound). There is no write or quit key: every change is
already saved. The help line under the buttons is `BROWSER_KEY_HELP`, the OpenCV tool's
`KEY_HELP` without its write / quit keys:

```python
>>> from vision.label_ui import ACTION_KEYS, BROWSER_KEY_HELP
>>> BROWSER_KEY_HELP[-1]
'+ add rally here   - delete rally   1/2/3 loop speed   esc cancel   every change is saved at once'
>>> sorted({"w", "q", "esc", "h"} & ACTION_KEYS)
['esc', 'h']

```

**Autosave:** after every change that alters labels or rallies the server writes both CSVs
atomically (`data/labels/<video id>.csv`: `rally, frame_num, player, type_group, type_detail, source`;
`data/labels/<video id>_rallies.csv`: `rally, start_frame, end_frame`). Reloading the page
or restarting the server restores the state from them.

**The rally clip** is cut on demand with `ffmpeg` (480 px wide, H.264, no audio) into a
temporary folder that holds only the current rally's clip and is removed on exit; it is
served with HTTP range requests, which Safari needs to play video. A range that starts past the
end is ignored and the whole clip is sent (`200`).

## HTTP interface

| Request | Answer |
|---|---|
| `GET /` | the page |
| `GET /api/state` | the state as JSON (`state_json`) |
| `POST /api/action` with a JSON object | applies it (`apply_action`), autosaves, answers the new state; `400` with `{"error": ...}` for any body `validated_action` refuses |
| `GET /frame.jpg?f=N` | frame N with the overlay, JPEG |
| `GET /clip.mp4?start=S&end=E` | the clip of frames S-E (E clamped to the last frame), with `Range` support (`206 Partial Content`); `400` when S is negative, after E or past the last frame (`checked_clip_span`); `500` when `ffmpeg` fails or is missing |

Actions: `{"type": "key", "key": "h"}` (any key of the table), `{"type": "jump", "frame": N}`,
`{"type": "rally", "index": I}` (0-based), `{"type": "delete", "frame": N}`. Every body is
checked in one place, `validated_action`, before anything runs. It must be a JSON object
with exactly these two fields. The type must be one of the four, the key a string of
`ACTION_KEYS`, and the frame or index a JSON number with no fraction (not a string, not
`true`). Anything else gets `400` and leaves the state alone. Any other error inside a
request is answered with `500` and the error, never by dropping the connection. A
browser that closes the connection mid-answer is not an error: stepping frames quickly
cancels the pending frame requests. `BrokenPipeError`, `ConnectionResetError` and
`ConnectionAbortedError` (`CLIENT_GONE_ERRORS`) are dropped quietly, the socket is
closed, and no `500` is attempted on it. The state and the label files are not touched.

## Functions

The request handling is a thin layer over pure functions:

```python
>>> import pandas as pd
>>> from vision.label import initial_state, HitLabel
>>> from vision.label_ui import apply_action, delete_label_at, go_to_rally_index, jump_to_frame, parse_byte_range, progress, state_json
>>> hits = pd.DataFrame({"rally": [1, 1, 2], "frame": [100, 130, 400], "hitter": ["me", "opp", "me"], "score": [0.9, 0.8, 0.9], "reach": float("nan")})
>>> rallies = pd.DataFrame({"rally": [1, 2], "start_frame": [90, 390], "end_frame": [200, 500], "hits": [2, 1]})
>>> state = initial_state(hits, rallies, last_frame=999, fps=30.0)
>>> jump_to_frame(state, 450).frame, jump_to_frame(state, 450).rally_index, jump_to_frame(state, 300).rally_index, jump_to_frame(state, 5000).frame
(450, 1, 0, 999)
>>> go_to_rally_index(state, 1).frame, go_to_rally_index(state, 9).rally.rally
(390, 2)
>>> for action in ({"type": "jump", "frame": 131}, {"type": "key", "key": "h"}, {"type": "key", "key": "o"}, {"type": "key", "key": "n"}):
...     state = apply_action(state, action)
>>> state.labels, progress(state)
((HitLabel(rally=1, frame=131, player='opp', type_group='net', source='hand', type_detail='net'),), {'rallies_labelled': 1, 'rallies': 2, 'hits_labelled': 1, 'serves_without_length': 0, 'unchecked_rallies': 2})
>>> view = state_json(state)
>>> view["rally"], [p["frame"] for p in view["proposals"]], view["labels"][0]["frame"], view["pending"], view["can_undo"]
({'rally': 1, 'start_frame': 90, 'end_frame': 200}, [100, 130], 131, None, True)
>>> delete_label_at(state, 131).labels, delete_label_at(state, 7) is state
((), True)
>>> apply_action(state, {"type": "fly"})
Traceback (most recent call last):
ValueError: unknown action: {'type': 'fly'}
>>> apply_action(state, [1, 2])
Traceback (most recent call last):
ValueError: an action is a JSON object, not [1, 2]
>>> apply_action(state, {"type": "rally", "index": None})
Traceback (most recent call last):
ValueError: not a whole number: None
>>> apply_action(state, {"type": "key", "key": ["a"]})
Traceback (most recent call last):
ValueError: unknown action: {'type': 'key', 'key': ['a']}
>>> from vision.label_ui import validated_action
>>> validated_action({"type": "jump", "frame": 120.0}), validated_action({"type": "key", "key": "h"})
(('jump', 120), ('key', 'h'))
>>> for body in ({"type": "jump", "frame": "120"}, {"type": "jump", "frame": 1.5}, {"type": "delete"}, {"type": "rally", "index": 1, "extra": 2}, {"type": ["key"]}):
...     try:
...         validated_action(body)
...     except ValueError as error:
...         print(error)
not a whole number: '120'
not a whole number: 1.5
a delete action needs 'frame': {'type': 'delete'}
unexpected fields in {'type': 'rally', 'index': 1, 'extra': 2}
unknown action: {'type': ['key']}
>>> parse_byte_range("bytes=0-99", 1000), parse_byte_range("bytes=900-", 1000), parse_byte_range("bytes=-100", 1000), parse_byte_range("bytes=2000-", 1000), parse_byte_range(None, 1000)
((0, 99), (900, 999), (900, 999), None, None)

```

### `clip_command(video_path, start_frame, end_frame, fps, output_path, width=480) -> list[str]`
The `ffmpeg` call for a rally clip: seek to `start_frame / fps`, cut the rally's
duration, scale, H.264 `ultrafast`, `+faststart`.

```python
>>> from pathlib import Path
>>> from vision.label_ui import clip_command
>>> command = clip_command(Path("set1.mp4"), 600, 899, 60.0, Path("clip.mp4"))
>>> command[command.index("-ss") + 1], command[command.index("-t") + 1], command[command.index("-vf") + 1]
('10.000', '5.000', 'scale=480:-2')

```

### `checked_clip_span(start_frame, end_frame, last_frame) -> (int, int)`
The frames a clip may cover: `end_frame` clamped to the last frame; `ValueError` for a
negative start, a start after the end or a start past the last frame.

```python
>>> from vision.label_ui import checked_clip_span
>>> checked_clip_span(20, 90, 999), checked_clip_span(950, 1200, 999)
((20, 90), (950, 999))
>>> checked_clip_span(90, 20, 999)
Traceback (most recent call last):
ValueError: no clip for frames 90-20: need 0 <= start <= end and start <= 999

```

### `LabelSession(video_path, labels_file, rallies_file, players_dir, hits_dir)`, `make_handler(session)`, `serve(session, port=8766, open_browser=True)`, `main(arguments)`
The session holds the state behind a lock (`act`, `snapshot`), renders frames
(`frame_jpeg`), cuts clips (`clip_path`) and removes its temporary folder (`close`);
`make_handler` builds the `http.server` request handler around it.
