# Shot features from tracking: results

## How to run
```
M=Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals
.venv/bin/python -m vision.features data/match.mp4 --match-id match_tracked \
  --range 14201 14947 1 --range 16551 17343 1 --range 81360 82116 2 \
  --range 107560 108103 3 --range 114673 116641 3 --compare $M
.venv/bin/python -m shots.predict
.venv/bin/python -m analytics.tendencies match_tracked me
.venv/bin/python -m report.serve match_tracked --me me --no-serve
.venv/bin/python -m vision.features data/match.mp4 --match-id match_tracked --spot-check --compare $M
```
The last command draws the check images under `data/checks/features/` from the rows in
the database and prints the `not_over_net` breakdown and the sample tally quoted below.
The five ranges are the ones tracked for hit detection (`docs/results/hits.md`); the
command needs their player, shuttle and hit files under `data/`. `--compare` prints the
comparison below; it is meaningful after `shots.predict` has filled `type_pred`.

## What was written
126 rows with `source = 'own'`, `match_id = 'match_tracked'`, in `data/scout.duckdb`:
8 rallies (the 7 real ones plus the 1-hit "rally" of a shuttle pickup from hit
detection), 121 rows with a landing point. The 5 without are the last shots of rallies
whose range ended while the shuttle was still in flight (no rest frame to take the
floor point from). Every row passed the table's constraints: `player` is `me` or `opp`,
the hitter is on the near half after normalisation (`player_y` ≤ 5.4 m on every row,
`opponent_y` ≥ 8.8 m), zones and `is_cross` come from `analytics.court.shot_geometry`,
the same function ShuttleSet rows go through. A rebuild with the same `match_id`
replaces exactly these rows; the 36,483 ShuttleSet rows are untouched (counted before
and after).

Columns that tracking can't produce are NULL on every own row:

| Column | Why NULL |
|---|---|
| `type_en`, `type_group` | no labels; `type_pred`/`type_conf` are the classifier's output instead |
| `backhand` | needs the racket side; not tracked |
| `roundscore_me`, `roundscore_opp` | the score isn't read from video |
| `getpoint_player`, `lose_reason` | who won the rally and why isn't detected |
| `landing_x/y` on 5 last shots | range ended mid-flight (above) |

Downstream ran unchanged on the own match: `shots.predict` filled `type_pred` for
121 rows (the 5 without a landing point stay NULL, as for ShuttleSet),
`analytics.tendencies` returned 6 tendencies for `me` and 4 for `opp` (all from
predictions, n of 6-11), and `report.serve` wrote `data/reports/match_tracked/index.html`.

## Classifier accuracy on tracked positions
114 own rows match a labelled ShuttleSet shot within ±5 frames in the same set (one to
one, closest first). 110 of them have both a prediction and a labelled group other
than `unknown`.

| | Accuracy |
|---|---|
| Classifier on ShuttleSet's labelled positions (held out, `docs/results/classifier.md`) | **86.8%** |
| **Classifier on tracked positions, this match** | **73 of 110 = 66.4%** |

| Labelled | Correct | Mistaken for |
|---|---|---|
| clear | 7 of 7 | |
| smash | 12 of 14 | drop 2 |
| net | 29 of 36 | drive 5, drop 1, serve 1 |
| lift | 12 of 19 | net 4, clear 3 |
| drop | 8 of 15 | smash 3, clear 2, drive 1, net 1 |
| drive | 5 of 18 | lift 6, net 4, serve 2, drop 1 |
| serve | 0 of 1 | net 1 |

Clears and smashes survive; drives, drops and lifts don't. That is the first number for
the whole pipeline's shot typing and it bears about as much as 110 shots from one match
can (±9 points). Two measured reasons it is lower than 86.8%:

**1. The hit point convention differs from ShuttleSet's.** The contract takes the
hitter's feet as the hit point. In ShuttleSet the hit point is **not** the feet: on
this match it is a median 0.88 m from the labelled `player_location` (mean 0.91, 90th
percentile 1.51 m; 0.82 m median over all 44 matches), and the overlay
`data/checks/features/match_tracked_label_vs_feet.jpg` (6 matched rows, the labelled
hit pixel in red and the tracked feet in yellow on the same frame) shows what it is: the
floor point under the racket contact, about 1 m in front of the feet on a lunging or
reaching shot, on the feet for an overhead. By group the gap is 1.07 m for
net shots, 0.98 lifts, 0.88 drives, 0.80 drops, 0.59 smashes, 0.55 clears. The
classifier learned shot length and direction from contact points; a feet-based hit
point makes every reaching shot about 1 m shorter and shifts the hit zone back, which
is where the drop/lift/drive confusions sit. The note in the project's data section
that `hit_x/hit_y` mark the feet is therefore an approximation that holds for
overheads only.

**2. Tracking error.** Against the labelled `player_location`, the tracked feet are a
median 0.61 m off (90th percentile 1.36 m; `me` 0.66, `opp` 0.55). Part of that is the
annotator's click (a body position, not the sole of the shoe), part the 1-frame offset of
detected hits (a lunging player moves 0.1-0.2 m per frame), and part the box-bottom
foot point of a jumping or lunging player.

Against the labelled hit point the tracked hit point is a median 1.14 m off (mean 1.25),
the landing point 1.23 m (mean 2.28, pulled up by the shots whose next hit is wrong).
Zone agreement with the labels: `hit_zone` 59.6%, `landing_zone` 52.6%, `is_cross` 88.2%.
Hitter agreement 93.9% (107 of 114, the hit-detection number; ShuttleSet's `A`/`B` were
mapped to near/far with the ends rule from `docs/results/schema.md`).

**Errors inherited from hit detection** show up as rows, not as wrong values. Counted by
`not_over_net_breakdown` (printed by `--spot-check`), 25 of the 126 rows have
`not_over_net` as their landing zone: 23 of them are rows whose next hit in the rally is by the
same player (an extra or a missed hit in between, so the "landing" is that player's own
feet) and 2 rally-end rest points on the hitter's side (set 3 rallies 2 and 4: the missed
smash at 115082, and the pickup false positive at 115856). ShuttleSet uses that zone for
shots into the net; for own rows it should be read as "hit sequence error here", and the
20% rate is the cost of the hit detector's 90% precision and recall compounded over
pairs of consecutive hits.

## Spot check against the video
`data/checks/features/match_tracked_spot_check_{0..3}.jpg`, drawn by `--spot-check`:
20 own rows drawn at random (numpy seed 1; frames 14347, 14532, 14675, 16681, 16749,
16757, 16853, 17215, 17248, 81472, 81862, 107722, 107782, 108071, 114857, 114901,
114969, 115032, 116547, 116621). For each, the hit frame with the hit point (yellow,
the hitter's feet mapped back to pixels) and the opponent's position (blue), and the
next hit's frame with the landing point (red, the receiver's feet), with the row's
player, `type_pred`, confidence, zones and `is_cross` and the matched label's group and
player.

What they show: in all 20 the hit marker is on the hitter's feet and, where there is a
next hit, the landing marker is on the receiver's feet at that frame (that is the
definition). Hitter, zones and `is_cross` read correctly for the position in every
tile. The 4 odd rows are the inherited ones above: two consecutive hits by the same
player (16757->16771, 81472->81507, 114901->114929, 114969->114978) with a
`not_over_net` landing. Predictions in the sample: 9 right, 7 wrong, 2 without a label
(16757, 114969: detections with no labelled hit within 5 frames) and 2 without a
prediction (108071, a last shot with no landing point, and 116621, which has neither),
as counted by `sample_tally`. The 7 wrong: 16681 net predicted drive, 17215 drive
predicted lift, 107722 lift predicted net, 114901 drive predicted net, 115032 lift
predicted net, 14532 drop predicted clear, 16749 drop predicted smash.

## Own footage
The criterion "20 shots spot-checked against the video" is met on the broadcast
rallies above. Its own-footage version is pending: there is no recording from the club
camera yet. When there is, the same command writes the rows (`--range` per tracked
range with its set number, since sets aren't detected from video), and
`--left-handed me` or `opp` sets the handedness that zones depend on.
