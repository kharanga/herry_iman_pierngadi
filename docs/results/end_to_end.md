# End to end: results

## How to run
```
.venv/bin/python -m scout.run set1.mp4 [set2.mp4 set3.mp4] --match-id <id>
```
How it works and every option: `docs/analytics/run.md`.

## Status
| Check | Status |
|---|---|
| One command runs every step per set video, then shot types, habit counts and the report | done, tested below |
| A rerun skips steps whose outputs exist | done, tested below |
| Report for a full match of mine | **open**: no recording of my own yet; stand-in set videos (a club match from YouTube) are being prepared |
| Accuracy against one hand-labelled game (hits, shot groups) | **open**: needs my own footage and a hand-labelled game |

## Test: two "set videos" from the broadcast match
There is no footage of mine yet, so two links to `data/match.mp4` (Axelsen vs Ginting), named `set1.mp4` and
`set2.mp4`, stood in for two set videos. Each got its own court and camera calibration, made with the
calibration commands' non-interactive options (the 4 ShuttleSet corners and the 2 net-post tops already measured
for `data/match.mp4`). Both ran the same 747-frame stretch (frames 14201-14947, one long rally):

```
.venv/bin/python -m scout.run set1.mp4 set2.mp4 --match-id int_two_sets --range 14201 14947 --shuttle-ensemble nonoverlap --no-serve
```

It ran against a copy of `data/scout.duckdb`, so the real database was not changed.

| Step | Set 1 | Set 2 |
|---|---|---|
| court, camera | reused (0 s) | reused (0 s) |
| players | 391 s | 266 s |
| shuttle (`nonoverlap`) | 315 s | 459 s |
| hits | 10 s (25 hits) | 18 s (25 hits) |
| features | 9 s (25 shots) | 16 s (25 shots) |

Then, for the match: write shots 0.2 s (50 own rows: 25 in set 1, 25 in set 2), classifier 3.1 s, habit counts 0.5 s,
report 0.3 s. The report covers sets 1 and 2 (60 habit entries for me and opp together, all from predicted shot types).

- **Both sets are kept.** `vision.features.build_own_match` replaces all own rows of the match id on every call, so
  calling it once per video into the real database would keep only the last set. `scout.run` builds each set into a
  throwaway database and writes all sets at once: the database holds 50 rows, 25 per set.
- **Reruns skip the slow steps.** Running the same command again reused every court, camera, players, shuttle and hits
  output and finished in 8 s. `--force hits` reran only that step.
- **Repeatable.** With the same inputs, building a set's rows twice gives identical rows.
- **Name collision found.** Every per-video output (calibration, tracks, hits) is stored under the video's file name
  without extension, and the test links were named `set1.mp4` / `set2.mp4`, the same names as the stand-in club
  videos in `data/recordings/stand_in/`. When the stand-in `set1` was calibrated, it replaced the test's `set1`
  calibration, so a later rerun of the test used the wrong court: 0 hits after `--force hits`, and contact points metres
  away. The timings above come from the first run, before that happened.
- **Guard against reusing another video's outputs.** Outputs are now keyed by `vision.court.video_id` (parent folder +
  file name), and every output carries a source record (size and hashes of the video's first and last megabyte). A step
  is reused only for the same video; outputs from another video with the same id (e.g. `2026_11_02/club/set1.mp4`, then
  `2026_11_09/club/set1.mp4`) are recomputed and the log names the other video. Set videos in one run must differ in
  file name and id, with or without `--range`. On the broadcast test range, `scout.run` still reuses every existing
  track (the records were added the first time, vouched for by that video's court calibration).
- **Errors stop the run clearly**: a missing video or an unknown `--force` step exits 1 with `error: ...`; a failing
  vision command stops the run with the step and the video named.

## What a real set will cost
On this Mac, player tracking ran at about 1.9 to 2.8 frames per second and shuttle tracking (`nonoverlap`, the fast
mode) at about 1.6 to 2.4. A 20-minute set at 30 fps is about 36,000 frames, so roughly 4 to 5 hours for players and
another 4 to 6 hours for the shuttle, per set; at 60 fps, twice that. The default `weighted` shuttle mode is about 8x
slower again. That is why a rerun reuses every existing output: the tracking only has to run once per video.
These are estimates from 747-frame runs, to be replaced by measured numbers once a full set is processed.

## Notes
- Frame counts and fps are read from each video file, so 60 fps footage needs no setting.
- A camera that can't see the near baseline is calibrated from named court points: `--pick-points` (a click window
  where you choose the point) or `--court-points SET "NAME=X,Y" ...` (no window), both passed to `vision.court`; and
  `--net-posts SET X,Y X,Y` passed to `vision.camera`.
- Each video has its own calibration; the near player is `me` in every video, and there is no end-switch handling.
- On my footage there are no rally results, so the winner and error families of the report stay empty.
