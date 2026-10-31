# End to end: results

## How to run
```
.venv/bin/python -m scout.run set1.mp4 [set2.mp4 set3.mp4] --match-id <id> [--device mps|cpu] [--batch-frames N]
```
How it works and every option: `docs/analytics/run.md`. The pipeline tracks the two players only: court
calibration, player tracking (GPU), hits from player movement with an inferred serve per rally, shot features
from the feet, the feet classifier, habit counts, the phone report.

## Status
| Check | Status |
|---|---|
| One command runs court -> players -> hits -> features per set video, then shot types, habit counts and the report | done |
| `--shuttle-ensemble` and `--hit-point` gone; `--device` and `--batch-frames` passed to player tracking | done |
| A rerun skips steps whose outputs exist (source records) | done, tested below |
| Full stand-in set on the GPU through classifier, habit counts and report, with timings | done, below |
| Accuracy against a hand-labelled game | **open**: no labels on the stand-in; the movement detector's held-out numbers are in `docs/results/hits_movement.md`, the classifier's in `docs/results/classifier_feet.md` |

## Stand-in set 1, whole video
`data/recordings/stand_in/set1.mp4`: the first stretch of the club singles video that stands in for my own
recordings (`docs/results/stand_in.md`): 1920 x 1080, 60 fps, 26,235 frames (7 min 17 s), near player = `me`.
The court calibration from 7 named points already existed (`data/calibration/stand_in_set1/court.json`).

```
.venv/bin/python -m scout.run data/recordings/stand_in/set1.mp4 --match-id stand_in_set1 --no-serve --database <copy of data/scout.duckdb>
```

| Step | Time | Output |
|---|---|---|
| court | reused (calibration adopted, source record added) | `court.json` |
| players (`--device mps --batch-frames 8`) | **2,023 s** (33.7 min, 13.0 frames/s) | positions and boxes for 26,235 frames |
| hits (movement detector + inferred serves) | 2.2 s | 36 rallies, 251 detected hits, 36 inferred serves |
| features (feet) | 1.0 s | 287 shot rows |
| write shots + classifier + habit counts + report | 0.8 s | 287 own rows, `data/reports/stand_in_set1/` |

Player tracking was run once with exactly the command `scout.run` issues (same arguments, same source record), so
the `scout.run` command itself, reusing it, took 5.3 s. The whole set costs **34 minutes**, all of it player
tracking: 26,235 frames at 13 frames/s on the MacBook Air's GPU, with 26,225 main-view frames, `me` detected
in 26,197 of them and `opp` in 25,933 (`docs/results/players.md` for the speed table). The missing detections
are a few long runs (the longest 26 frames for `me`, 100 for `opp`), not 8-frame batches, so no detection
batch was dropped.

### What came out
| | |
|---|---|
| Rallies | 36, covering 18,353 of 26,235 frames (5 min 06 s of 7 min 17 s); 3 to 22 shots, 2.3 to 24.4 s each |
| Shot rows | 287: 251 detected hits + 36 inferred serves (one per rally, 0.76 s before the first detected hit) |
| Rows with a landing point and a prediction | 251 (the 36 rally-ending shots have no reply, so no landing and no prediction) |
| Serve rows | 36, every one with the server's feet (fixed camera, the serve frame is always in view); predicted `serve` 35, `net` 1 |
| `me` | 149 rows, 124 predicted: net 35, drop 23, serve 22, drive 20, lift 12, clear 7, smash 5 |
| `opp` | 138 rows, 127 predicted: net 45, drive 28, lift 18, serve 13, clear 11, drop 6, smash 6 |
| Habit entries | `me` 106 (21 zone, 81 reply, 4 serve placement), `opp` 114 (22, 89, 3); winner and error families empty (no rally results) |
| Report | `data/reports/stand_in_set1/index.html`, 298 KB; phone-width screenshot `data/checks/report/stand_in_set1_390px.png` |
| Database | the 287 rows are also in `data/scout.duckdb` (`source = 'own'`, `match_id = 'stand_in_set1'`), so `report.serve stand_in_set1 --me me` works from the real command |

**Doubled serves.** When the movement detector catches the serve itself, the serve rule puts a second serve row
in front of it (`docs/results/no_shuttle.md`: 3 of 7 in-sample on the broadcast, 0 of 7 held out). On the stand-in,
0 of the 36 rallies have a second row typed `serve`: every `serve` prediction (35) sits on an inferred serve row at
`ball_round` 1, and no `ball_round` 2 row is predicted `serve`. That check can only see a doubled serve the
classifier recognises as one; a detected serve typed as a return by the feet features would pass unnoticed, and
without labels on this footage that remains possible.

**No labels, so no accuracy.** The stand-in has no hand-labelled hits or shot types; `docs/results/stand_in.md`'s
20-hit spot check was of the shuttle-based detector. The numbers above say the pipeline runs end to end on a full
60 fps set and what it produces, not how right it is. The 36 rallies in 7 minutes and the 2 to 24 s rally
lengths are plausible for club singles; the 25 rallies that end on a `me` row against 11 on an `opp` row are not
rally results, only who hit last before the detector lost the rally.

## The broadcast match, rebuilt
The own rows of `match_tracked` (5 labelled ranges of Axelsen vs Ginting, `docs/results/features.md`) were rebuilt
from the movement hits with `vision.features` and re-predicted into `data/scout.duckdb`: 132 rows over 7 rallies
(7 inferred serves), 125 with a landing point, 119 predicted; 6 of the 7 serve rows have no hitter position
because the broadcast cuts to a close-up at the serve (`docs/results/no_shuttle.md`), so `me` has no serve bullets
on that page. `docs/results/tendencies.md` and `docs/results/report.md` carry the rebuilt counts (65 bullets for
the opponent, 71 for me).

## Reruns and the source guard
- On the broadcast range (`data/match.mp4`, frames 14201-14947), `scout.run` reuses court, players and hits
  from their source records, builds 25 shots and writes the report in 5 s; the log has no camera or shuttle line.
- Every output carries `<output>.source.json` (size and hashes of the video); outputs from another video with the
  same id are recomputed and the log names the other video; `--force <step>|all` reruns regardless.
- Set videos must differ in file name and `video_id`; the two-set behaviour (each set's rows kept, one write for the
  match) is unchanged from the first version of this command.

## What a real set costs
Player tracking is the whole cost: about 13 frames/s on the GPU at 1920 x 1080 / 60 fps, so a 20-minute set
(72,000 frames) takes about 90 minutes, and a 3-set match about 4 to 5 hours of unattended Mac time, once. The
other steps take seconds, and every rerun (a new classifier, a changed habit count) reuses the tracks. Recording at
30 fps would halve it; `docs/results/players.md` has the batch-size and device measurements.

## Notes
- Frame counts and fps are read from each video file, so 60 fps footage needs no setting.
- A camera that can't see the near baseline is calibrated from named court points: `--pick-points` (a click window)
  or `--court-points SET "NAME=X,Y" ...` (no window), both passed to `vision.court`.
- Each video has its own calibration; the near player is `me` in every video, and there is no end-switch handling.
- On own footage there are no rally results, so the winner and error families of the report stay empty.
