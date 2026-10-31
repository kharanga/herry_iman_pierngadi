# vision/label.py

A window for labelling hits by hand on my own footage, rally by rally, starting from
the movement detector's proposals. Every number on the club video is otherwise
"agreement with another run"; the labels make precision, recall and shot-type
agreement measurable there. The tool never runs a model: it reads the video, the
tracking files (for the overlay) and the hits/rallies files (for the rallies and the
proposals), and writes two CSVs.

The labelling state (`LabelState`) and every change to it (`handle_key` and the
functions it calls) are pure: state in, state out, no files, no window. The OpenCV
loop (`label_video`) only reads frames, draws and forwards key names.

## Command

```
.venv/bin/python -m vision.label <video> --match-id <id> [--labels FILE] [--players-dir D] [--hits-dir D]
```

| Argument | Meaning |
|---|---|
| `video` | the set video; needs its hits and rallies files for the whole video (`python -m vision.hits <video> --start 0 --end <last frame>`), and shows the tracking overlay when its player files exist |
| `--match-id` | shown in the window title and the printed summary |
| `--labels` | the labels CSV to read on start and write on `w`/`q` (default `data/labels/<video id>.csv`); the rallies CSV sits next to it as `<name>_rallies.csv` |
| `--players-dir`, `--hits-dir` | roots of the tracking and hit files (defaults `data/players`, `data/hits`), e.g. to label on top of another tracking configuration |

The window (`compose_panels`): left the **stepping panel**, the current frame paused
at 960 px wide with the tracking overlay (`vision.overlay`), under it the rally's
timeline (`draw_timeline`: proposals as short yellow ticks, labels as green bars with
the hitter's and type's initials, the current frame as a white line) and a status line;
right the **playing panel**, the rally's slice looping at a slow speed (`draw_loop_panel`,
480 px wide, read once per rally into memory, at most 900 frames so a long rally is
subsampled), with a white marker at the stepping panel's frame and a yellow dot at the
loop's own position. The key map is printed on start and drawn under the panels.

| Key | Action |
|---|---|
| space | play / pause the left panel (one frame per screen refresh) |
| `,` `.` (or left / right arrow) | one frame back / forward |
| `<` `>` | ten frames back / forward |
| `[` `]` | previous / next rally (the left panel jumps to its start, the right panel reloads) |
| `h` | mark a hit at the current frame; then `m` or `o` for the hitter, then a type key |
| `s` `c` `d` `k` `n` `l` `v` `?` | after `h` or `a` and the hitter: serve, clear, drop, smash, net, lift, drive, unknown; the label is stored and shown |
| `a` | accept the nearest proposal within 60 frames: jumps to it, takes its hitter, waits for the type key |
| `x` | delete the labelled hit nearest to the current frame |
| `u` | undo the last change to labels or rallies (200 steps) |
| `r` | move the current rally's start (when the frame is in its first half) or end (second half) to the current frame |
| `+` | add a rally starting at the current frame (2 s long; adjust with `r`), when the frame is in no rally |
| `-` | delete the current rally and its labels |
| `1` `2` `3` | loop speed 0.25x, 0.5x (default), 1x |
| `w` | write both CSVs (atomically) |
| `q` | write and quit; `esc` quits without writing (or cancels a pending hit) |

Shift+arrow is not readable through OpenCV on every platform, hence `<` and `>`.
Labelling a frame that already has a label replaces it. A hit's rally is the rally
containing its frame; moving rally bounds relabels the hits and drops those left
outside every rally.

Output: `data/labels/<video id>.csv` with `rally, frame_num, player, type_group,
source` (`hand` for `h`, `accepted` for `a`) and `data/labels/<video id>_rallies.csv`
with `rally, start_frame, end_frame`, rallies numbered by start frame. On start the
tool loads both if they exist (the rallies file replaces the detector's rallies).
Scoring a hits file against the labels: `python -m vision.hits <video> --start N --end N
--evaluate-labels data/labels/<video id>.csv [--database data/scout.duckdb --match-id ID]`
(`docs/vision/movement.md`). Exit codes: `0`; `1` with `error: ...` for a missing video
or hits file.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `LABELS_DIR` | `data/labels` | where the CSVs go |
| `LABEL_COLUMNS`, `RALLY_COLUMNS` | see above | the two files' columns |
| `TYPE_KEYS`, `PLAYER_KEYS`, `STEP_KEYS` | key maps | `s c d k n l v ?` to groups, `m o` to players, `, . < >` to steps |
| `PROPOSAL_MAX_DISTANCE_FRAMES` | `60` | `a` accepts a proposal only this close |
| `NEW_RALLY_SECONDS` | `2.0` | length of a rally added with `+` |
| `LEFT_PANEL_WIDTH`, `RIGHT_PANEL_WIDTH`, `LOOP_MAX_FRAMES` | `960`, `480`, `900` | panel sizes and the loop's memory cap |

## Classes

`HitLabel(rally, frame, player, type_group, source)`, `RallySpan(rally, start_frame,
end_frame)`, `Proposal(frame, hitter)`, `PendingHit(frame, source, player=None)` and
`LabelState(labels, rallies, proposals, rally_index, frame, last_frame, fps, pending,
history, dirty)`, all frozen; `state.rally` is the current `RallySpan` or `None`.

## State functions

Every example starts from two detector rallies and three proposals:

```python
>>> import pandas as pd
>>> from vision.label import *
>>> hits = pd.DataFrame({"rally": [1, 1, 1, 2, 2], "frame": [100, 130, 160, 400, 430], "hitter": ["opp", "me", "opp", "me", "opp"], "score": [float("nan"), 0.9, 0.8, 0.9, 0.7], "reach": float("nan")})
>>> rallies = pd.DataFrame({"rally": [1, 2], "start_frame": [100, 400], "end_frame": [200, 500], "hits": [3, 2]})
>>> state = initial_state(hits, rallies, last_frame=999, fps=30.0)
>>> state.frame, state.rally, len(state.proposals)
(100, RallySpan(rally=1, start_frame=100, end_frame=200), 4)

```

The inferred serve (score NaN) is not a proposal.

### `renumbered(rallies)`, `rally_containing(rallies, frame)`, `labels_relabelled(labels, rallies)`
Rallies sorted and numbered by start frame; the index of the rally containing a frame;
labels re-assigned to the rally containing them (dropped when none does).

```python
>>> renumbered([RallySpan(5, 400, 500), RallySpan(9, 100, 200)])
(RallySpan(rally=1, start_frame=100, end_frame=200), RallySpan(rally=2, start_frame=400, end_frame=500))
>>> rally_containing(state.rallies, 450), rally_containing(state.rallies, 300)
(1, None)
>>> labels_relabelled([HitLabel(1, 450, "me", "net", "hand"), HitLabel(1, 300, "me", "net", "hand")], state.rallies)
(HitLabel(rally=2, frame=450, player='me', type_group='net', source='hand'),)

```

### `step(state, frames)`, `go_to_rally(state, rally_index)`, `clamp_frame(frame, last_frame)`

```python
>>> step(state, 10).frame, step(state, -500).frame, step(replace(state, frame=995), 10).frame
(110, 0, 999)
>>> go_to_rally(state, 1).frame, go_to_rally(state, 7).rally.rally, go_to_rally(state, -3).rally.rally
(400, 2, 1)

```

### `nearest_index(frames, frame)`, `nearest_label(state)`, `nearest_proposal(state, max_distance=60)`

```python
>>> nearest_index([100, 130, 160], 140), nearest_index([], 5)
(1, None)
>>> nearest_proposal(replace(state, frame=150)), nearest_proposal(replace(state, frame=300))
(Proposal(frame=160, hitter='opp'), None)

```

### `begin_hit`, `accept_nearest_proposal`, `set_pending_player`, `set_pending_type`, `cancel_pending`, `with_label`
The three-key sequence: a pending hit (frame and source), then the hitter, then the
type, which stores the label (replacing one on the same frame) and clears the pending
hit. `set_pending_type` before a hitter does nothing.

```python
>>> pending = begin_hit(replace(state, frame=131))
>>> pending.pending
PendingHit(frame=131, source='hand', player=None)
>>> set_pending_type(pending, "net").labels
()
>>> labelled = set_pending_type(set_pending_player(pending, "me"), "net")
>>> labelled.labels, labelled.pending, labelled.dirty
((HitLabel(rally=1, frame=131, player='me', type_group='net', source='hand'),), None, True)
>>> accepted = accept_nearest_proposal(replace(labelled, frame=158))
>>> accepted.frame, accepted.pending
(160, PendingHit(frame=160, source='accepted', player='opp'))
>>> two = set_pending_type(accepted, "lift")
>>> [(label.frame, label.player, label.type_group, label.source) for label in two.labels]
[(131, 'me', 'net', 'hand'), (160, 'opp', 'lift', 'accepted')]
>>> cancel_pending(accepted).pending is None
True

```

### `delete_nearest_label`, `undo`

```python
>>> fewer = delete_nearest_label(replace(two, frame=150))
>>> [label.frame for label in fewer.labels], [label.frame for label in undo(fewer).labels]
([131], [131, 160])
>>> undo(state).labels == state.labels
True

```

### `set_rally_bound`, `add_rally`, `remove_rally`

```python
>>> moved = set_rally_bound(replace(two, frame=120))
>>> moved.rally
RallySpan(rally=1, start_frame=120, end_frame=200)
>>> set_rally_bound(replace(two, frame=180)).rally.end_frame
180
>>> added = add_rally(replace(two, frame=700))
>>> [span for span in added.rallies], added.rally.rally
([RallySpan(rally=1, start_frame=100, end_frame=200), RallySpan(rally=2, start_frame=400, end_frame=500), RallySpan(rally=3, start_frame=700, end_frame=760)], 3)
>>> add_rally(replace(two, frame=150)).rallies == two.rallies
True
>>> removed = remove_rally(go_to_rally(two, 0))
>>> removed.rallies, removed.labels, removed.frame
((RallySpan(rally=1, start_frame=400, end_frame=500),), (), 400)

```

### `handle_key(state, key)`
The dispatcher the window calls with a key name (`","`, `"h"`, `"esc"`, ...): while a
hit is pending only `m`/`o`, a type key and `esc` do anything; otherwise steps, rally
moves and the single-key actions.

```python
>>> flow = state
>>> for key in ("]", ">", ".", "h", "o", "k"):
...     flow = handle_key(flow, key)
>>> flow.frame, flow.labels
(411, (HitLabel(rally=2, frame=411, player='opp', type_group='smash', source='hand'),))
>>> handle_key(begin_hit(flow), ".").frame == flow.frame, handle_key(begin_hit(flow), "esc").pending
(True, None)

```

### `labels_frame`, `rallies_frame`, `labels_from_frame`, `rallies_from_frame`, `proposals_from_hits`, `label_paths`, `write_atomically`, `save_labels`, `initial_state`
The two tables and their round trip; `initial_state` takes the detector's hits and
rallies plus, optionally, saved labels and rallies (the saved rallies win).

```python
>>> labels_frame(two)
   rally  frame_num player type_group    source
0      1        131     me        net      hand
1      1        160    opp       lift  accepted
>>> rallies_frame(two)
   rally  start_frame  end_frame
0      1          100        200
1      2          400        500
>>> import tempfile
>>> from pathlib import Path
>>> folder = Path(tempfile.mkdtemp())
>>> saved = save_labels(two, folder / "clip.csv", folder / "clip_rallies.csv")
>>> saved.dirty, sorted(path.name for path in folder.iterdir())
(False, ['clip.csv', 'clip_rallies.csv'])
>>> again = initial_state(hits, rallies, 999, 30.0, pd.read_csv(folder / "clip.csv"), pd.read_csv(folder / "clip_rallies.csv"))
>>> again.labels == two.labels, again.rallies == two.rallies
(True, True)
>>> [path.as_posix() for path in label_paths(Path("data/recordings/stand_in/set1.mp4"))]
['data/labels/stand_in_set1.csv', 'data/labels/stand_in_set1_rallies.csv']

```

## Drawing

### `timeline_x`, `draw_timeline(state, width, height=48)`, `status_line(state)`, `compose_panels(left, right, timeline, status, help_lines=KEY_HELP)`, `scaled(frame, width)`, `draw_loop_panel(frames, indices, position, cursor_frame, speed)`, `loop_position(indices, elapsed_seconds, fps, speed)`, `read_loop_frames(capture, span, width=480, max_frames=900)`, `key_name(code)`
Pure drawing on arrays, plus the loop's bookkeeping: `loop_position` maps elapsed wall
time at `speed` to an index of the stored slice; `read_loop_frames` reads a rally's
frames once (every k-th when the rally has more than `max_frames`); `key_name` turns
OpenCV's key code into the names `handle_key` understands (arrow codes become `,`/`.`).

```python
>>> import numpy as np
>>> timeline_x(150, RallySpan(1, 100, 200), 101)
50
>>> draw_timeline(two, 300).shape, draw_timeline(replace(two, rallies=()), 300).shape
((48, 300, 3), (48, 300, 3))
>>> status_line(replace(two, frame=131))
'f131  rally 1 of 2 [100-200]  labels in rally: 2  total: 2  (unsaved)'
>>> status_line(set_pending_player(begin_hit(state), "me"))
'f100  rally 1 of 2 [100-200]  labels in rally: 0  total: 0   HIT at f100 by me: press a type key'
>>> canvas = compose_panels(np.zeros((540, 960, 3), np.uint8), np.zeros((270, 480, 3), np.uint8), draw_timeline(two, 960), status_line(two))
>>> canvas.shape
(718, 1450, 3)
>>> indices = list(range(100, 200))
>>> loop_position(indices, 0.0, 30.0, 0.5), loop_position(indices, 1.0, 30.0, 0.5), loop_position(indices, 4.0, 30.0, 0.5)
(0, 15, 60)
>>> draw_loop_panel([np.zeros((270, 480, 3), np.uint8)], [100], 0, 100, 0.5).shape, draw_loop_panel([], [], 0, 100, 0.5).shape
((270, 480, 3), (270, 480, 3))
>>> key_name(27), key_name(32), key_name(2), key_name(ord("h")), key_name(255)
('esc', 'space', ',', 'h', None)

```

### `label_video(video_path, labels_file, rallies_file, players_dir=PLAYERS_DIR, hits_dir=HITS_DIR, window_name="label hits") -> int`, `main(arguments) -> int`
The window loop and the command entry point.
