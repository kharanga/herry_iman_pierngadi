# vision/serve_stance.py

Finds serves from the player tracks alone, by the stance before them. Before a serve,
both players stand nearly still, each in a service court, diagonal from each other. The
server moves first. The serve frame is the server's detected hit closest to the end of
the stance (from 0.3 s before it to 0.5 s after). If there is no such hit, the serve
frame is the stance's end.

Used by `python -m vision.movement --in-play-model M --serve-stance` (`docs/vision/movement.md`):
the in-play model finds the stretches, each stretch's serve comes from this rule, and the
`--model` hits fill the rally after it.

Positions are court metres (`x_m` across, 0 to 6.1; `y_m` along, net at 6.7), speeds m/s,
both from `vision.movement.motion_features` and its smoothing.

## Command

```
.venv/bin/python -m vision.serve_stance --train VIDEO LABELS HIT_MODEL [--train ...]
```

Tunes the three stance constants on fully tracked, labelled training videos (loaded with
`vision.hit_training.load_club`, so doubtful frames are left out and unreviewed labels
refused). HIT_MODEL gives the hits the serve frame snaps to, and must not have been
trained on that video. The command prints, for each of the 18 grid points, the
labelled serves found at ±0.5 s, the guesses (only those inside the video's labelled
span, ±0.5 s) and F1 = 2 × found ÷ (guesses + labelled), best first. Then it prints the
best point. Exit codes: `0`; `1` with `error: ...` for a missing file or column; `2` for
bad arguments.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `STILL_SPEED_M_PER_S` | `0.5` | both players under this speed count as still (tuned) |
| `STANCE_SECONDS` | `0.5` | a stance lasts at least this long (tuned) |
| `MOVE_SPEED_M_PER_S` | `0.8` | the server is the first player over this speed after the stance (tuned) |
| `MOVE_WINDOW_SECONDS` | `1.5` | the first move is looked for this long after the stance; no move, no serve |
| `SERVE_LEAD_SECONDS`, `SERVE_REACH_SECONDS` | `0.3`, `0.5` | the server's hit is looked for from this long before the stance end to this long after |
| `STANCE_GAP_SECONDS` | `0.2` | breaks in the stance this short are closed |
| `COURT_MARGIN_M` | `0.5` | margin around a service court for tracking error |
| `MIN_SERVE_GAP_SECONDS` | `3.0` | of two serve guesses closer than this, the later one is kept |
| `STILL_SPEED_GRID`, `STANCE_SECONDS_GRID`, `MOVE_SPEED_GRID` | `(0.3, 0.5, 0.8)`, `(0.5, 1.0, 1.5)`, `(0.8, 1.2)` | the tuning grid |
| `SERVE_BEFORE_STRETCH_SECONDS`, `SERVE_AFTER_STRETCH_START_SECONDS` | `2.0`, `1.0` | in the pipeline: a stretch takes the serve guess nearest its start within this window |

The tuned values come from Lin, Ding and n7 (the club training games) with the nine-match
hit scorer: 97 of 151 labelled serves found at ±0.5 s, from 164 guesses
(`docs/results/rally_start_end.md`).

## Functions

### `in_service_court(x, y, margin_m=0.5) -> ndarray`, `diagonal(near_x, far_x) -> ndarray`
In a service court: between the short service line (1.98 m from the net) and the
baseline, across the full court width, each with the margin. Diagonal: the players are
on opposite sides of the centre line.

```python
>>> import numpy as np, pandas as pd
>>> from vision.serve_stance import diagonal, in_service_court
>>> in_service_court(np.array([1.0, 1.0, 1.0, 7.0]), np.array([2.0, 5.5, 9.0, 2.0]))
array([ True, False,  True, False])
>>> diagonal(np.array([2.0, 2.0]), np.array([4.0, 1.0]))
array([ True, False])

```

### `runs_of(mask, min_length, max_gap) -> list[(int, int)]`, `first_mover(speeds, after, window, threshold) -> str | None`
Runs of true samples, with gaps of up to `max_gap` closed, keeping runs of at least
`min_length` samples. Then the player who first goes over the speed in the `window`
samples after sample `after`.

```python
>>> from vision.serve_stance import first_mover, runs_of
>>> runs_of(np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1], dtype=bool), min_length=3, max_gap=1)
[(0, 5)]
>>> first_mover({"me": np.array([0.1, 0.1, 0.2, 1.5, 2.0]), "opp": np.array([0.1, 0.1, 1.2, 1.5, 2.0])}, after=1, window=3, threshold=0.8)
'opp'

```

### `ServeGuess`, `serve_hit_near(hits, server, earliest, latest, stance_end) -> int | None`, `later_of_close_guesses(guesses, min_gap) -> list[ServeGuess]`
A guess holds the serve frame, the server, the stance's first and last frame, and
whether the frame came from a hit. The server's hit nearest the stance end within the
window, and the thinning of guesses closer than `min_gap` frames.

```python
>>> from vision.serve_stance import ServeGuess, later_of_close_guesses, serve_hit_near
>>> hits = pd.DataFrame({"frame": [100, 118, 140], "hitter": ["me", "me", "opp"], "score": [0.9, 0.8, 0.7]})
>>> serve_hit_near(hits, "me", 95, 130, 120), serve_hit_near(hits, "opp", 95, 130, 120)
(118, None)
>>> guesses = [ServeGuess(10, "me", 0, 10, False), ServeGuess(50, "me", 40, 50, True), ServeGuess(300, "opp", 280, 300, True)]
>>> [guess.frame for guess in later_of_close_guesses(guesses, 90)]
[50, 300]

```

### `stance_mask(tracks, settings, stance)`, `serve_guesses(tracks, hits, settings, stance=StanceSettings()) -> list[ServeGuess]`, `StanceSettings`
The per-sample stance mask (both still, both in a service court, diagonal) with each
player's speed, and the serve guesses of a whole track. `hits` are detected hits in
video frames; the guesses are in video frames.

### `serve_for_stretch(guesses, stretch_start, settings)`, `rally_after_serve(candidates, serve, stretch_end, settings)`, `stance_rallies(...)`
The pipeline pieces. A stretch takes the guess nearest its start, from 2 s before it to
1 s after. The rally is the serve, then the alternating hit chain after it, up to
0.2 s past the stretch end. Leading hits by the server are dropped, and so is anything
later. `stance_rallies` returns the hits and rallies tables of `vision.movement`.
A stretch without a guess falls back to `vision.in_play.rally_of_span`.
The rally starts at the serve frame.

```python
>>> from vision.movement import HitCandidate, settings_for_video
>>> from vision.serve_stance import rally_after_serve, serve_for_stretch
>>> settings = settings_for_video(30.0, 1280.0)
>>> serve_for_stretch(guesses, 60, settings).frame, serve_for_stretch(guesses, 200, settings)
(50, None)
>>> candidates = [HitCandidate(55, "me", 0.6), HitCandidate(70, "opp", 0.9), HitCandidate(95, "me", 0.8), HitCandidate(200, "opp", 0.7)]
>>> [(hit.frame, hit.hitter) for hit in rally_after_serve(candidates, HitCandidate(50, "me", float("nan")), 100, settings)]
[(50, 'me'), (70, 'opp'), (95, 'me')]

```

### `serve_scores`, `f1_of`, `grid`, `TrainingVideo`, `load_training_video`, `guesses_inside`, `tuning_rows`, `main`
The tuning command's pieces.

```python
>>> from vision.serve_stance import f1_of, grid
>>> round(f1_of(97, 164, 151), 3), len(grid())
(0.616, 18)

```
