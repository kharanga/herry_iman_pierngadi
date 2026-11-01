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
command needs their player and hit files under `data/` (`python -m vision.players`,
`python -m vision.hits`). `--compare` prints the comparison below; it is meaningful
after `shots.predict` has filled `type_pred`.

## Where the rows come from
The rows now in the database are the **player-only pipeline's**: hits from the two
players' movement (`docs/results/hits_movement.md`), an inferred serve row at the start
of each rally, the hitter's feet as the hit point, the reply hitter's feet as the
landing, and a NULL landing on each rally's last shot (`docs/results/no_shuttle.md`).
No shuttle track, contact estimate or camera is involved any more. The earlier versions
of this page (the shuttle-based hits with the feet, then with the estimated racket
contact, as the hit point) are in the git history; their headline was 73-74 correct
predictions of 110 judged (66-67%) on 126 rows, with 25 rows whose landing fell in
`not_over_net` because of hit-sequence errors.

## What was written
132 rows with `source = 'own'`, `match_id = 'match_tracked'`, in `data/scout.duckdb`:
7 rallies (the 7 labelled ones; 26, 28, 27, 18, 16, 11 and 6 rows), 7 of them inferred
serve rows at `ball_round` 1, 125 rows with a landing point. The 7 without are the last
shots of the rallies (no next hit). 6 rows have no hit point: six of the seven serve
rows, because the broadcast is on a close-up of the server at the inferred serve frame
and the player tracker has no main-view position within 3 frames; the seventh serve
row has one. Every row passed the table's constraints: `player` is `me` or `opp`, the
hitter is on the near half after normalisation (`opponent_y` ≥ 8.9 m on every row),
`hit_x/y` equals `player_x/y` on every row with a hit point, zones and `is_cross` come
from `analytics.court.shot_geometry`, the same function ShuttleSet rows go through
(the six-zone layout: front, mid and rear bands with edges 1.98 m, the short service
line, and 3.50 m from the net, split into forehand and backhand at the centre line;
`data/checks/court/match_zones.jpg` shows it on this video). A
rebuild with the same `match_id` replaces exactly these rows; the 36,483 ShuttleSet
rows are untouched (counted before and after).

Columns that tracking can't produce are NULL on every own row:

| Column | Why NULL |
|---|---|
| `type_en`, `type_group` | no labels; `type_pred`/`type_conf` are the classifier's output instead |
| `backhand` | needs the racket side; not tracked |
| `roundscore_me`, `roundscore_opp` | the score isn't read from video |
| `getpoint_player`, `lose_reason` | who won the rally and why isn't detected |
| `landing_x/y` on the 7 last shots | no next hit (above) |
| `hit_x/y`, `player_x/y` on 6 serve rows | no tracked position at the inferred frame (above) |

Downstream ran unchanged on the own match: `shots.predict` filled `type_pred` for
119 rows (the 7 last shots have no landing and the 6 serve rows without a position have
no hit point, so both stay NULL, as for ShuttleSet), `analytics.tendencies` returned
entries for `me` and `opp` from predictions, and `report.serve` wrote
`data/reports/match_tracked/index.html` (66 entries for `me`, 68 for `opp`).

## Classifier accuracy on tracked positions
123 own rows match a labelled ShuttleSet shot within ±5 frames in the same set (one to
one, closest first). 116 of them have both a prediction and a labelled group other
than `unknown`.

| | Accuracy |
|---|---|
| Feet-only classifier on ShuttleSet's labelled positions (held out, `docs/results/classifier_feet.md`) | **86.5%** |
| **Feet-only classifier on tracked positions, this match** | **92 of 116 = 79.3%** |

| Labelled | Correct | Mistaken for |
|---|---|---|
| net | 36 of 38 | drive 1, drop 1 |
| smash | 14 of 15 | drop 1 |
| clear | 6 of 7 | smash 1 |
| lift | 13 of 19 | clear 5, drive 1 |
| drive | 11 of 19 | lift 5, net 3 |
| drop | 12 of 16 | smash 2, net 1, clear 1 |
| serve | 0 of 2 | clear 1, net 1 |

By set: sets 1-2 60 of 74, set 3 32 of 42. These are the predictions of the classifier
trained on the ten shot labels (serve short / long, net, return net, drive, push, clear,
drop, smash, lift), scored at group level. The total is the same 92 of 116 as with the
earlier group-level model; drops gained 2 (10 to 12), and lifts and nets each lost 1. Net shots, smashes and clears are nearly all
right; drives and lifts confuse each other, drops spread. The two labelled serves that
were judged are the real serves that the movement detector itself found on these
ranges (it was trained on them), which the inferred serve row then pushed to
`ball_round` 2, where the classifier types them as a return (`docs/results/no_shuttle.md`).

This is 13 points above the 66% of the shuttle-based rows, and the reason is the hit
sequence, not the positions: the movement detector's hits alternate players by
construction, so no row has the same player twice in a row, the landing is always the
opponent's reply, and 0 of the 132 rows have `not_over_net` as their landing zone (it
was 25 of 126). The
shuttle-based rows carried every extra or missed hit into a wrong landing and a wrong
reply situation. Hitter agreement with the labels is 123 of 123 (the shuttle's was 107
of 114).

**The positions themselves.** Against the labelled `player_location`, the tracked feet
are a median 0.57 m off (mean 0.68). Against the labelled hit point (the floor point
under the racket contact, about 1 m in front of the feet on a reaching shot) the feet
hit point is a median 1.05 m off (mean 1.10), the landing point 1.06 m (mean 1.14).
Zone agreement with the labels: `hit_zone` 69.9%, `landing_zone` 70.0%, `is_cross`
92.2% (with the earlier nine-zone layout and its 1 m middle column: 61.8%, 64.2% and
90.5%; the wider zones forgive more of the feet-to-contact gap). The gap to the 86.5% held-out figure is what the feet convention and the
tracking error cost; `docs/results/classifier_feet.md` has the timing-jitter table for
the frame offsets of detected hits.

## Spot check against the video
`data/checks/features/match_tracked_spot_check_{0..3}.jpg`, drawn by `--spot-check`:
20 own rows drawn at random (numpy seed 1; frames 14285, 14327, 14492, 14676, 16683,
16750, 16811, 16894, 17314, 17335, 81488, 81806, 107647, 107724, 108042, 114811,
114871, 114979, 115082, 116515). For each, the hit frame with the hit point (yellow,
the feet, mapped back to pixels) and the opponent's position (blue), and the next hit's
frame with the landing point (red, the receiver's feet), with the row's player,
`type_pred`, confidence, zones and `is_cross` and the matched label's group and player.

In all 20 the hit marker is on the hitter's feet and, where there is a next hit, the
landing marker is on the receiver's feet at that frame (that is the definition).
Hitter, zones and `is_cross` read correctly for the position in every tile.
Predictions in the sample: 15 right, 4 wrong, 0 without a label and 1 without a
prediction (17335, the last shot of its rally, no landing point), as counted by
`sample_tally`. The 4
wrong: 14327 smash predicted drop, 81488 net predicted drive, 81806 lift predicted
clear, 114811 lift predicted clear.

`data/checks/features/match_tracked_label_vs_feet.jpg` (6 matched rows, the labelled
hit pixel in red and the tracked feet in yellow on the same frame) shows the convention
gap: ShuttleSet's hit point is the floor point under the racket contact, about 1 m in
front of the feet on a lunging or reaching shot, on the feet for an overhead.

## Own footage
The criterion "20 shots spot-checked against the video" is met on the broadcast
rallies above. Its own-footage version is pending: there is no recording from the club
camera yet. When there is, the same command writes the rows (`--range` per tracked
range with its set number, since sets aren't detected from video), and
`--left-handed me` or `opp` sets the handedness that zones depend on.
