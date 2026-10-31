# The pipeline without the shuttle: results

The shuttle left the pipeline: hits come from the players' movement
(`docs/results/hits_movement.md`), the hit point is the hitter's feet, a rally's last
shot has no landing point, and each rally gets an inferred serve row. This page records
what changed in the rows and in the classifier's output on the five labelled ranges of
the broadcast match, and the one number the serve inference rests on.

## How to run
```
M=Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals
for r in "14201 14947 1" "16551 17343 1" "81360 82116 2" "107560 108103 3" "114673 116641 3"; do
  set -- $r   # zsh: set -- ${=r}
  .venv/bin/python -m vision.hits data/match.mp4 --start $1 --end $2 --evaluate $M $3 [--no-serves]
done
.venv/bin/python -m vision.features data/match.mp4 --match-id match_movement \
  --range 14201 14947 1 --range 16551 17343 1 --range 81360 82116 2 \
  --range 107560 108103 3 --range 114673 116641 3 --database <copy of data/scout.duckdb>
.venv/bin/python -m shots.predict <copy>
```
`vision.hits` now runs the movement detector (`docs/vision/hits.md`); it needs only the
player tracks and the movement model. `vision.features` reads the hits and positions
files, no shuttle, camera or contact file. The comparison below was made on a copy of
the database; the shipped `match_tracked` rows are unchanged until the integration
step rebuilds them.

## What changed in the rows
| | Shuttle pipeline (`docs/results/features.md`) | Players only, no serve rows | Players only, serve rows (shipped) |
|---|---|---|---|
| Own rows on the 5 ranges | 126 (8 rallies, one a pickup) | 125 (7 rallies) | 132 (7 rallies, 7 serves) |
| Hit point | contact floor point (fallback feet) | feet | feet |
| Rows with a landing point | 121 | 118 | 125 |
| Rows without (last shot of a rally) | 5 | 7 | 7 |
| `ball_round` 1 | the first detected hit (the return) | the return | the inferred serve |

## The inferred serve
The movement detector misses every serve (7 of 7 on these ranges): the server flicks
from standing and the receiver waits. `vision.hits` therefore adds, for each detected
rally, a row for the other player than the first hitter, `SERVE_TO_RETURN_SECONDS`
before the first detected hit, with `score` NaN in `hits_*.csv` so it is visibly
inferred; the detected hits then start at `ball_round` 2. A rally whose first detected
hit is too close to the start of the tracked range to fit the gap gets no serve row.

**The gap: 0.76 s.** The median time from the serve to the return over the 3,502
ShuttleSet rallies that have both a `ball_round` 1 and 2 with hit frames (query over
the `shots` table, `(frame_num of shot 2 - frame_num of shot 1) / fps`): median 0.76 s,
quartiles 0.64 and 1.00 s, mean 0.89 s. At 30 fps that is 23 frames, at 60 fps 46.

**Where it lands on the labelled serves** (shipped model on the 5 ranges, which it was
trained on; the labelled first shot vs the inferred serve frame):

| Range | Labelled serve | First detected hit | Inferred serve | Verdict |
|---|---|---|---|---|
| 14201-14947 | 14231 | 14232 (the serve itself, found in-sample) | 14209 | doubled: a false serve 22 frames before the real one, by the wrong player |
| 16551-17343 | 16581 | 16601 (the return) | 16578 | right, 3 frames early |
| 81360-82116 | 81390 | 81409 (the return) | 81386 | right, 4 frames early |
| 107560-108103 | 107590 | 107628 (the return, 1.27 s after a long serve) | 107605 | right shot, 15 frames late |
| 114673-116641 rally 20 | 114703 | 114704 (the serve itself, in-sample) | 114681 | doubled |
| rally 21 | 115589 | 115588 (the serve itself, in-sample) | 115565 | doubled |
| rally 22 | 116493 | 116515 (the return) | 116492 | right, 1 frame early |

3 of 7 inferred serves are within ±5 frames of the labelled serve, 1 is the right shot
15 frames late, and 3 are false: on those three rallies the shipped model, which was
trained on these very ranges, found the serve itself, and the rule then put a second,
wrong serve in front of it. Held out (`docs/results/hits_movement.md`), the detector
missed all 7 serves, so the rule adds the right row; on new footage the model is out of
sample and the held-out behaviour is the one to expect, but the doubling is a real
failure mode whenever the detector does catch a serve (a stillness test before the
first hit could tell the two apart; not done). None of the 7 labelled serves has a
hitter position in ShuttleSet, so the server side could not be checked against the
labels; it follows from alternation.

## Effect on the classifier
`shots.predict` with the feet model (`docs/results/classifier_feet.md`, as retrained
for the six-zone layout) on the same rows, before and after the serve rows (the
"before" column is from the model before that retraining; one drive moved from lift to
clear with it):

| | Without serve rows | With serve rows |
|---|---|---|
| `ball_round` 1 predicted as | serve 5, lift 1, drive 1 (the returns, mislabelled) | serve 1, unpredicted 6 (the serve rows) |
| The returns (now `ball_round` 2) predicted as | | drive 4, net 2, clear 1 |
| Rows whose prediction changed | | 9 of 125 detected rows: the 6 returns, 1 at `ball_round` 4, 2 at 24 (features of the previous shot) |
| Overall predictions | net 38, lift 23, smash 17, drive 12, clear 12, drop 11, serve 5, none 7 | net 41, lift 19, smash 17, drive 16, clear 14, drop 11, serve 1, none 13 |

Without the serve row the classifier called 5 of the 7 returns a serve (`ball_round` 1
is almost always a serve in ShuttleSet, and the model learned it). With the serve row
the returns get return-like types. The serve rows themselves are mostly unpredicted on
this broadcast video: 6 of the 7 have no hit point, because the broadcast is on a
close-up of the server at the serve frame and the player tracker has no main-view
position within 3 frames (`docs/results/players.md`, "Skipped frames"). The one serve
row with a position was predicted `serve`. (The three doubled serves above still
remove the `ball_round` 1 label from the real serve, which then sits at `ball_round` 2
and is typed as a return: the cost of the doubling.) On a fixed club camera every serve frame is
in view, so the serve rows get the server's feet and a prediction.

## Limitations
- The serve frame is a population median, not a detection: a slow or fast return puts
  it up to about 15 frames off, and a missed return puts it on the wrong shot. It only
  has to be close enough for the feet lookup (within 3 frames of a tracked position).
- When the detector does find the serve, the rule doubles it (3 of 7 in-sample, 0 of 7
  held out).
- The server's position at that frame is the only thing the row carries from video;
  everything else is inferred from the detected hits.
