# `scout/run.py`: from set videos to the report

One command for a whole match: for each set video it runs court calibration, player tracking, hit detection
and shot features, then predicts shot types, counts every habit and writes (and serves) the phone report.
The pipeline tracks the two players only (no shuttle): hits come from player movement, the hit point is the
hitter's feet and the landing is where the opponent hits the reply. Every step is the vision or analytics
track's own public function or command; this module only orders them, skips work that is already done and
combines the sets.

## Command

```
.venv/bin/python -m scout.run set1.mp4 [set2.mp4 set3.mp4] --match-id <id> [options]
```

- One video per set, in set order: the first is set 1, and so on (1 to 3 videos). The near player is `me` in
  every video (the camera is moved behind me between sets), so there is no end-switch handling.
- `--match-id` (required): the id of this match's own rows in `shots` and of its report, e.g. `club_2026-11-02`.
- `--range START END`: only that frame range of every video (for testing on part of a video). Default: the
  whole video, frames 0 to its frame count minus 1, read from the file.
- `--force STEP`: rerun a step even if its outputs exist; repeatable or comma-separated. Steps: `court`,
  `players`, `hits`, `features`, `all`.
- `--device mps|cpu`: where player detection runs, passed to `vision.players`. Default: not passed, so
  `vision.players` picks `mps` when available, else `cpu`.
- `--batch-frames N`: frames per detection call, passed to `vision.players` (default its
  `DETECTION_BATCH_FRAMES`, 8). Must be at least 1.
- `--calibration-frame N`: the frame to calibrate the court on (default 0).
- `--pick-points`: court calibration with the point-picker window (`vision.court --pick-points`), for a camera that
  can't see all 4 corners (e.g. the near baseline out of frame). Without it, the 4-corner window opens.
- `--court-points SET "NAME=X,Y" ...`: court calibration for one set without a window: the set number, then 4 or more
  named points (names from `python -m vision.court <video> --list-points`; quote each, the names contain spaces).
  Repeatable, one per set. Overrides `--pick-points` for that set.
- `--left-handed me,opp`: passed to shot features (zones).
- `--database` (default `data/scout.duckdb`), `--port` (default 8765), `--no-serve` (write the report only).
- Prints one line per step (`set 1 players: reused ...` or `ran, wrote ... (812.3 s)`), then `Report written: ...`,
  then serves the page like `report.serve` and prints the URL to open on the phone. Exit 1 with `error: ...` if a
  video is missing, a step fails, or the arguments are wrong.

## What runs, in order

Per video (set N):

| Step | Runs | Output (skipped if present) |
|---|---|---|
| `court` | `vision.court <video> --frame N` plus `--points ...` or `--pick-points` when given (else the 4-corner window) | `data/calibration/<video_id>/court.json` |
| `players` | `vision.players <video> --start --end [--device D] --batch-frames N` | `data/players/<video_id>/positions_<start>_<end>.csv`, `boxes_...` |
| `hits` | `vision.hits <video> --start --end` | `data/hits/<video_id>/hits_<start>_<end>.csv`, `rallies_...` |
| `features` | `vision.features.build_own_match` for set N (hit point = the hitter's feet) | always rerun (about a second) |

`<video_id>` is `vision.court.video_id(video)`: the file name without extension for a video directly under `data/`
(`data/match.mp4` -> `match`), else the parent folder and the file name (`data/recordings/stand_in/set1.mp4` ->
`stand_in_set1`). The set videos of one run must have different file names and different ids (checked, with or
without `--range`).

**Outputs from another video are never reused.** Two different videos can still share an id, e.g.
`2026_11_02/club/set1.mp4` and `2026_11_09/club/set1.mp4` (both `club_set1`). So every output gets a source record
next to it, `<output>.source.json`: the video's resolved path, its size, and SHA-256 hashes of its first and last
megabyte. A step is reused only when all its outputs exist and every record matches the video by size and both hashes
(so a moved or renamed copy of the same file still counts as the same video). If a record names another video, the
step is rerun and the log says whose outputs they were; the new outputs and records replace the old ones. Outputs
made before records existed are reused only when all of these hold: this video's own `court.json` names the
same file (court calibrations store their video's path), the frame size it records (if any) matches the video, every
one of the step's outputs is newer than the video file, and the court step was not rerun in this run. A record is then
added. Otherwise they are rerun: a new recording saved over an old file at the same path is newer than the old tracks,
so it never inherits them. `--force` reruns a step whatever its records say.

Then, for the match:

1. **Write shots**: all sets' rows together with `vision.features.write_own_shots` (it replaces this match id's own rows).
   Each set's rows are first built into a throwaway database, because `build_own_match` replaces every own row of the
   match id on each call, so calling it once per video into the real database would keep only the last set.
2. **Classifier**: `shots.predict.predict_shot_types` (fills `type_pred` / `type_conf` for every row whose hitter's
   feet and reply hitter's feet are known; a rally's last shot gets none).
3. **Habit counts**: `analytics.tendencies.tendencies_for` for `me` and `opp`, saved as
   `data/reports/<match_id>/tendencies_me.json` and `tendencies_opp.json`.
4. **Report**: `report.serve.write_report(match_id, "me")` -> `data/reports/<match_id>/index.html`, then served
   unless `--no-serve`.

On own footage there are no rally results, so the winner and error families stay empty.

## Functions

- `run_match(video_paths, match_id, database_path, frame_range=None, forced_steps=frozenset(), left_handed=frozenset(),
  model_path=MODEL_PATH, calibration=CalibrationInputs(), tracking=TrackingOptions())`: everything above; returns
  `(report_path, StepLog)`. Errors: `FileNotFoundError` (a video, or a model file a step needs), `ValueError` (bad
  arguments, or calibration given for a set number with no video), `StepFailedError` (a vision command returned
  non-zero; `step`, `video_path`, `exit_code`), and the report's `UnknownMatchError` when no shot was produced.
- `set_videos(video_paths, frame_range=None)`: `SetVideo(video_path, set_no, start_frame, end_frame)` per video.
- `CalibrationInputs(frame=0, pick_points=False, court_points_by_set={})`, `court_arguments(video, calibration)`,
  `calibration_inputs(parsed)`, `court_points_by_set(groups)`, `set_number(text)`: the calibration options.
- `TrackingOptions(device=None, batch_frames=DETECTION_BATCH_FRAMES)`, `players_arguments(video, tracking)`,
  `tracking_options(parsed)`, `batch_frames(text)` (argparse type, at least 1): the detection options. `DEVICES`
  (`mps`, `cpu`).
- `step_outputs(step, video)`, `step_arguments(step, video, calibration=CalibrationInputs(), tracking=TrackingOptions())`,
  `is_forced(step, forced_steps)`, `parse_forced_steps(force_texts)`: the skip rule and the command lines.
- `run_video_steps(video, forced_steps, log, calibration, tracking)`, `set_rows(video, match_id, left_handed)`,
  `write_match_rows(database_path, match_id, rows_by_set)`, `write_tendencies(match_id, database_path)`, `tendencies_path(match_id, player)`.
- `video_fingerprint(video_path)`, `source_record_path(output_path)`, `write_source_record(output_path, video_path, fingerprint)`,
  `read_source_record(output_path)`, `is_same_source(record, fingerprint)`, `calibration_names_video(video)`,
  `video_frame_size(video_path)`, `outputs_are_newer_than_video(outputs, video_path)`,
  `reuse_decision(outputs, fingerprint, may_adopt_unrecorded, video_path=None)`: the source guard above. `FINGERPRINT_BYTES` (1 MB),
  `SOURCE_RECORD_SUFFIX` (`.source.json`).
- `video_frame_count(video_path)`: frames in the file (OpenCV); `FileNotFoundError` / `ValueError` if unreadable.
- `StepLog`: the printed lines and seconds per step. Constants: `VIDEO_STEPS` (`court`, `players`, `hits`),
  `FORCEABLE_STEPS`, `MAX_SET_VIDEOS`, `STEP_COMMANDS`, `REPORT_PLAYERS`, `DATABASE_PATH`.

Vision functions this module depends on (public entry points only): `vision.court.main`, `calibration_path`,
`video_id`; `vision.players.main`, `positions_path`, `boxes_path`, `GPU_DEVICE`, `CPU_DEVICE`, `DETECTION_BATCH_FRAMES`;
`vision.hits.main`, `output_paths`; `vision.features.build_own_match`, `write_own_shots`.

```python
>>> import tempfile
>>> from pathlib import Path
>>> from scout.run import SetVideo, TrackingOptions, parse_forced_steps, set_videos, step_arguments, step_outputs
>>> video = SetVideo(Path("data/recordings/set2.mp4"), 2, 0, 899)
>>> [path.as_posix() for path in step_outputs("players", video)]
['data/players/recordings_set2/positions_0_899.csv', 'data/players/recordings_set2/boxes_0_899.csv']
>>> [path.as_posix() for path in step_outputs("court", video)], [path.as_posix() for path in step_outputs("hits", video)]
(['data/calibration/recordings_set2/court.json'], ['data/hits/recordings_set2/hits_0_899.csv', 'data/hits/recordings_set2/rallies_0_899.csv'])
>>> step_arguments("players", video)
['data/recordings/set2.mp4', '--start', '0', '--end', '899', '--batch-frames', '8']
>>> step_arguments("players", video, tracking=TrackingOptions("cpu", 1))
['data/recordings/set2.mp4', '--start', '0', '--end', '899', '--device', 'cpu', '--batch-frames', '1']
>>> step_arguments("hits", video)
['data/recordings/set2.mp4', '--start', '0', '--end', '899']
>>> from scout.run import CalibrationInputs
>>> calibration = CalibrationInputs(120, False, {2: ("near_left=1,2", "near_right=3,4", "far_left=5,6", "far_right=7,8")})
>>> step_arguments("court", video, calibration)
['data/recordings/set2.mp4', '--frame', '120', '--points', 'near_left=1,2', 'near_right=3,4', 'far_left=5,6', 'far_right=7,8']
>>> step_arguments("court", video, CalibrationInputs(pick_points=True))
['data/recordings/set2.mp4', '--frame', '0', '--pick-points']
>>> step_arguments("shuttle", video)
Traceback (most recent call last):
    ...
ValueError: unknown step: 'shuttle'
>>> sorted(parse_forced_steps(["players,hits", "court"]))
['court', 'hits', 'players']
>>> parse_forced_steps(["shuttle"])
Traceback (most recent call last):
    ...
argparse.ArgumentTypeError: unknown step(s) to force: ['shuttle']; choose from court, players, hits, features, all
>>> from scout.run import parse_arguments, tracking_options
>>> tracking_options(parse_arguments(["set1.mp4", "--match-id", "club", "--device", "cpu", "--batch-frames", "4"]))
TrackingOptions(device='cpu', batch_frames=4)
>>> tracking_options(parse_arguments(["set1.mp4", "--match-id", "club"]))
TrackingOptions(device=None, batch_frames=8)
>>> folder = Path(tempfile.mkdtemp())
>>> for name in ("set1.mp4", "set2.mp4"):
...     _ = (folder / name).write_bytes(b"")
>>> [(video.set_no, video.video_path.name, video.start_frame, video.end_frame) for video in set_videos([folder / "set1.mp4", folder / "set2.mp4"], (100, 199))]
[(1, 'set1.mp4', 100, 199), (2, 'set2.mp4', 100, 199)]
>>> for day in ("2026_11_02", "2026_11_09"):
...     (folder / day / "club").mkdir(parents=True)
...     _ = (folder / day / "club" / "set1.mp4").write_bytes(day.encode())
>>> try:
...     set_videos([folder / "2026_11_02" / "club" / "set1.mp4", folder / "2026_11_09" / "club" / "set1.mp4"], (0, 9))
... except ValueError as error:
...     print(error)
set videos must have different file names (their outputs are keyed by vision.court.video_id): ['club_set1', 'club_set1']
>>> from scout.run import is_same_source, reuse_decision, video_fingerprint, write_source_record
>>> first_day, next_day = (video_fingerprint(folder / day / "club" / "set1.mp4") for day in ("2026_11_02", "2026_11_09"))
>>> output = folder / "outputs" / "court.json"
>>> _ = output.parent.mkdir()
>>> _ = output.write_text("{}")
>>> _ = write_source_record(output, folder / "2026_11_02" / "club" / "set1.mp4", first_day)
>>> reuse_decision([output], first_day, False)
(True, 'reused')
>>> can_reuse, reason = reuse_decision([output], next_day, False)
>>> can_reuse, reason.startswith("made from another video (") and reason.endswith("2026_11_02/club/set1.mp4)")
(False, True)
>>> reuse_decision([folder / "outputs" / "missing.csv"], first_day, True)
(False, 'missing')
>>> set_videos([folder / "set1.mp4"] * 4, (0, 9))
Traceback (most recent call last):
    ...
ValueError: give 1 to 3 set videos, in set order; got 4
>>> try:
...     set_videos([folder / "missing.mp4"], (0, 9))
... except FileNotFoundError as error:
...     str(error).startswith("video not found:"), str(error).endswith("missing.mp4")
(True, True)

```
