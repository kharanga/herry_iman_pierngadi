# Shots inside rallies the user marked

How good is shot analysis once the rally bounds are right? The user marked each rally's
start (serve contact) and end (landing) on two new club games. The model then labelled
the shots inside those rallies only, and the user reviewed every shot in
`vision.label_ui --review`.

- **Videos:** `Wkux0r8gg0A`, first set to 8:00, and `NW8Vf_ElDUo`, first set to 11:00.
  Both were downloaded at 720p, with 30 fps copies cut 5 s after the set
  (`docs/vision/halve_fps.md`). Each copy was calibrated by the user and tracked in full.
- **Rallies:** the user's own: 27 on Wkux and 35 on NW8V, all with checked bounds. They
  were frozen in `data/vis043/user_rallies/` before the model ran.
- **Model** (`vision.movement --rallies-from-labels`, `docs/vision/movement.md`):
  - a serve row at each rally's start frame, with the server taken as the player
    opposite the first detected hit;
  - the nine + Lin + Ding tuned hit chain (0.55 / 0.6) inside the rally, from more than
    0.25 s after the serve to the end mark;
  - shot types from AVG with club = Lin + Ding + n7 (`docs/results/retrain_with_review.md`);
  - written with `vision.prelabel --rallies-checked`. The untouched model files are in
    `data/vis043/<video>/model_prelabels/`.
- **Files read:** the reviewed files, copied to `data/vis043/reviewed/`.
- **Fully reviewed:** both files have no `source = model` row left. Wkux's last one
  (frame 5226, rally 12) was reviewed later and rejected; the frozen copy includes it.
- **Status:** these are test games. Neither was used for training before this report.

**What this measures.** The user started from the model's proposals, so the timing leans
in the model's favour. No hit was moved, and frames were kept as the model had them. The
types, hitters, false hits and missed hits were judged by the user.

## Review

| | Wkux0r8gg0A | NW8Vf_ElDUo |
|---|---|---|
| Model rows reviewed | 161 (154 kept, 7 rejected) | 269 (225 kept, 44 rejected) |
| Kept exactly as the model had them (frame, type and hitter) | 122/161 = 75.8% | 174/269 = 64.7% |
| Kept with the type corrected | 26 | 35 |
| Kept with the hitter corrected | 7 | 19 |
| Kept and moved | 0 | 0 |
| Rejected | 7 | 44 |
| Added by the user (missed by the model) | 3 (2 lifts, 1 unknown) | 3 (lift, clear, drop) |
| Re-timed pairs (a rejection with an added hit within 1 s) | 0 | 0 (1 within 2 s) |

### Shot type on kept model rows (model type × user type)

Wkux0r8gg0A:

| model \ user | serve | net | lift | drive | clear | drop | smash | unknown | right |
|---|---|---|---|---|---|---|---|---|---|
| serve | 26 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 26/27 = 96.3% |
| net | 0 | 25 | 1 | 0 | 0 | 0 | 0 | 1 | 25/27 = 92.6% |
| lift | 0 | 0 | 33 | 1 | 0 | 0 | 0 | 0 | 33/34 = 97.1% |
| drive | 0 | 1 | 1 | 3 | 0 | 0 | 0 | 0 | 3/5 = 60.0% |
| clear | 0 | 0 | 0 | 0 | 5 | 1 | 2 | 0 | 5/8 = 62.5% |
| drop | 0 | 1 | 0 | 0 | 0 | 11 | 1 | 1 | 11/14 = 78.6% |
| smash | 0 | 1 | 0 | 0 | 1 | 1 | 8 | 0 | 8/11 = 72.7% |
| unknown | 0 | 2 | 4 | 0 | 0 | 2 | 3 | 17 | - |

NW8Vf_ElDUo:

| model \ user | serve | net | lift | drive | clear | drop | smash | unknown | right |
|---|---|---|---|---|---|---|---|---|---|
| serve | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 32/32 = 100.0% |
| net | 2 | 46 | 1 | 1 | 0 | 0 | 0 | 0 | 46/50 = 92.0% |
| lift | 0 | 2 | 56 | 0 | 0 | 0 | 0 | 0 | 56/58 = 96.6% |
| drive | 0 | 1 | 0 | 4 | 0 | 0 | 1 | 0 | 4/6 = 66.7% |
| clear | 0 | 0 | 0 | 2 | 12 | 2 | 3 | 0 | 12/19 = 63.2% |
| drop | 0 | 0 | 0 | 0 | 1 | 21 | 4 | 1 | 21/27 = 77.8% |
| smash | 0 | 0 | 1 | 0 | 1 | 0 | 14 | 2 | 14/18 = 77.8% |
| unknown | 1 | 0 | 1 | 0 | 0 | 1 | 4 | 8 | - |

"unknown" from the model is a row with no landing point, mostly a rally's last shot.

### Hitter

| | Wkux0r8gg0A | NW8Vf_ElDUo |
|---|---|---|
| Hitter right on kept rows | 147/154 = 95.5% | 206/225 = 91.6% |
| Hitter corrections on serve rows | 7 of 7 | 19 of 19 |
| Serve rows with the wrong server | 7 of 27 | 19 of 32 kept |

Every hitter correction is on a serve row. The hit chain's own hitters were all right.

### Why the server is wrong

The serve row's server is the player opposite the first detected hit. In 25 of the 26
rallies with the wrong server, that first hit was false. It was the server's own
follow-through after the serve, rejected by the user (6 of 7 on Wkux, 19 of 19 on NW8V).
It came 0.27-1.37 s after the serve (median 0.4 s on Wkux, 0.63 s on NW8V), past the
0.25 s that the pipeline treats as the serve itself. The other Wkux rally's first hit was
a kept drop with the right hitter. In rallies with the right server, the first chain hit
came 0.8-1.5 s after the serve (10th to 90th percentile) and was the receiver's return.

### What the 44 rejected NW8V rows were

| Rejected rows | Count | What they were |
|---|---|---|
| First row after the serve | 19 | the server's follow-through (above): 15 typed net, 3 lift, 1 drive |
| Last row of the rally | 20 | all untyped, 0.6-1.6 s after the last real shot (median 0.8 s), by the other player in 17 of 20 |
| Serve row | 3 | the user's start mark came before the serve. The real serve was the next model row, 0.47, 1.0 and 1.13 s later, kept and retyped as a serve |
| Mid-rally | 2 | a smash (rally 11) and a lift (rally 34) |

The 20 rally-end rows come from where the end was marked. On NW8V the user's end mark
sits a median 1.4 s after the last real hit, against 1.0 s on Wkux. In that time, the
other player's movement after the shuttle lands scores as a hit. 1 of Wkux's 27 last
rows was rejected: frame 5226, untyped, by the other player, 0.5 s after the last real
shot.

Wkux's 7 rejected rows: 6 first rows after the serve (net, 0.37-0.6 s after it) and that
last row.

## Correct shots

**Definition, used everywhere on this page:** correct shots = labelled shots that the
model matched (±0.50 s, one to one) and typed the same as the user ÷ (labelled shots +
the model's rejected rows). Serve rows count. A shot the user left unknown is never
correct, and nor is a model row with no type. Every rejected row counts as a false hit.
That includes rows where the user added a hit nearby instead (`vision.label_scores
--fps 30`, `correct_shots`). The hitter isn't part of it, so a serve row typed serve
counts even with the wrong server (see Hitter above). The second column below fixes that. No hit was moved, so ±0.25 s
gives the same numbers.

| Game | Rallies | Labelled shots | Rejected rows | Correct shots | Correct shots, right hitter too |
|---|---|---|---|---|---|
| Wkux0r8gg0A | the user's, 27 | 157 | 7 | **111/164 = 67.7%** | **105/164 = 64.0%** |
| NW8Vf_ElDUo | the user's, 35 | 228 | 44 | **185/272 = 68.0%** | **166/272 = 61.0%** |
| Both | the user's, 62 | 385 | 51 | **296/436 = 67.9%** | **271/436 = 62.2%** |
| n7MqoAVqEW4 (frozen copy `data/vis041/labels/`) | the combined detector's | 254 | 80 | 165/334 = 49.4% | 164/334 = 49.1% |

"Right hitter too" also needs the matched row's player to be the labelled one, with the
same denominator (`correct_shots_right_hitter`). Habits are counted per player, so a
shot credited to the wrong player is wrong in the report. The gap is the wrong servers:
6 serve rows typed right but with the wrong server on Wkux, and 19 on NW8V.

On the two new games every rejected row lies inside a rally. So `vision.label_scores`'
"false hits inside labelled rallies" is exactly the rejected rows. n7's row uses one
frozen copy, `data/vis041/labels/`: 254 kept labels and 80 rejected rows. Most of its
rejected rows lie outside the rallies the user kept, so `vision.label_scores` doesn't
count them; the denominator adds all 80 by hand. The numerators (165 and 164) come from
`vision.label_scores` on that copy.

Detected hits alone (no serve rows), ±0.50 s:

| Game | Found | Precision in the rallies | Hitter right | Type agrees |
|---|---|---|---|---|
| Wkux0r8gg0A | 131/157 = 83.4% | 131/134 = 97.8% | 100.0% | 84.2% |
| NW8Vf_ElDUo | 199/228 = 87.3% | 199/234 = 85.0% | 100.0% | 84.5% |
| n7MqoAVqEW4 | 195/254 = 76.8% | 195/211 = 92.4% | 97.9% | 84.8% |

## What it says

- **With the rallies given, about two thirds of all shots come out right.** Correct
  shots are 68% on both games, against 49.4% on n7, where the model found the rallies
  itself. With the hitter required too, it is 62.2% (271/436), against 49.1%. The
  difference is the wrong servers. That is the gain from right bounds: fewer missed hits (3 added per game,
  against 40 on n7) and far fewer false rows (7 and 44, against 80 on a game of 254
  shots).
- **Shot type agrees on 89% of matched rows typed by both** (Wkux 111/124, NW8V
  185/207).
  Serves, nets and lifts are 92-100%. Clears (62-63%), drives (60-67%), drops (78%) and
  smashes (73-78%) are where it misses, mostly among the rear-court shots.
- **The two mistakes that cost the most are both near the rally's edges:**
  - **The server.** The server's follow-through after the serve is detected as a hit,
    mostly 0.3-0.8 s later. That flips the server (26 serve rows) and adds a false net,
    lift or drive (25 rows). Taking the server from the stance rule's first mover
    (`docs/vision/serve_stance.md`), or not counting hits until 0.7 s after the serve,
    would address both. Not tried here.
  - **The rally end.** With a late end mark, a hit by the other player after the
    landing is added (20 rows on NW8V, 1 on Wkux). Dropping a rally's last hit when it comes within
    about 1.5 s of the end mark and has no landing point would remove most of them. Not
    tried here.
- **Hit timing:** the user moved no hit. Given the bias above, this says the model's
  frames were close enough to keep, not that they are exact.
- Two games, 385 labelled shots.

## How to run

zsh, from the project root.

The copies, calibration and tracks (the tracking used the GPU, one video at a time):

```
for pair in Wkux0r8gg0A:485 NW8Vf_ElDUo:665; do
  IFS=: read id t <<< "$pair"
  .venv/bin/yt-dlp --no-playlist -f "bv*[height<=720][ext=mp4]/bv*[height<=720]" --remux-video mp4 -o "data/recordings/club/%(id)s.%(ext)s" "https://www.youtube.com/watch?v=$id"
  ffmpeg -t $t -i data/recordings/club/$id.mp4 -an -vf "select=not(mod(n\,2)),setpts=N/(30000/1001)/TB" -fps_mode passthrough -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p data/recordings/club/${id}_30.mp4
done
.venv/bin/python -m vision.court data/recordings/club/Wkux0r8gg0A_30.mp4 --frame 6000 --pick-points
.venv/bin/python -m vision.court data/recordings/club/NW8Vf_ElDUo_30.mp4 --frame 3000 --pick-points
for id in Wkux0r8gg0A_30 NW8Vf_ElDUo_30; do
  .venv/bin/python -m vision.players data/recordings/club/$id.mp4 --plan-stretches $id.stretches.csv --stretch-seconds 20
  .venv/bin/python -m vision.players data/recordings/club/$id.mp4 --stretches $id.stretches.csv --merge --batch-frames 4
done
```

The rallies, marked by the user (b at serve contact, e at landing):

```
.venv/bin/python -m vision.label_ui data/recordings/club/Wkux0r8gg0A_30.mp4 --match-id club_Wkux0r8gg0A_30 --blank --port 8773
.venv/bin/python -m vision.label_ui data/recordings/club/NW8Vf_ElDUo_30.mp4 --match-id club_NW8Vf_ElDUo_30 --blank --port 8774
```

The model's labels, then the review (end frames 14535 and 19930):

```
N7V=data/recordings/club/n7MqoAVqEW4_30.mp4
TRAIN=(--train data/recordings/club/q3pfhnvKPig.mp4 data/labels/club_q3pfhnvKPig.csv data/flight/club_q3pfhnvKPig/scout.duckdb 29710 --train data/recordings/club/E6SRRmvJ6dI.mp4 data/labels/club_E6SRRmvJ6dI.csv data/flight/club_E6SRRmvJ6dI/scout.duckdb 31799 --train $N7V data/vis041/labels/club_n7MqoAVqEW4_30.csv data/flight/club_n7MqoAVqEW4_30/scout.duckdb 21727)
for target in Wkux0r8gg0A_30:14535:8775 NW8Vf_ElDUo_30:19930:8776; do
  IFS=: read id E port <<< "$target"
  V=data/recordings/club/$id.mp4; M=club_$id; O=data/vis043/$M
  mkdir -p data/vis043/user_rallies $O/model_prelabels
  cp data/labels/$M.csv data/labels/${M}_rallies.csv data/vis043/user_rallies/
  .venv/bin/python -m vision.movement $V --start 0 --end $E --model data/models/movement_hits_nine_plus_lin_ding_tuned.joblib --hit-score-min 0.55 --rally-end-score-min 0.6 --rallies-from-labels data/vis043/user_rallies/${M}_rallies.csv --hits-dir $O/hits
  cp data/scout.duckdb $O/scout.duckdb
  .venv/bin/python -m vision.features $V --match-id $M --range 0 $E 1 --database $O/scout.duckdb --hits-dir $O/hits
  .venv/bin/python -m vision.club_classifier $TRAIN --predict $V $O/scout.duckdb $E $O/classifier
  .venv/bin/python -m vision.prelabel $V --hits $O/hits/$M/hits_0_$E.csv --database $O/classifier/average.duckdb --match-id $M --rallies-checked --labels $O/model_prelabels/$M.csv
  .venv/bin/python -m vision.prelabel $V --hits $O/hits/$M/hits_0_$E.csv --database $O/classifier/average.duckdb --match-id $M --rallies-checked --force
  .venv/bin/python -m vision.label_ui $V --match-id $M --review --port $port
done
```

The report, from frozen copies of the reviewed files:

```
mkdir -p data/vis043/reviewed
for M in club_Wkux0r8gg0A_30 club_NW8Vf_ElDUo_30; do
  cp data/labels/$M.csv data/labels/${M}_rallies.csv data/labels/${M}_rejected.csv data/vis043/reviewed/
  .venv/bin/python -m vision.review_report data/recordings/club/${M#club_}.mp4 --labels data/vis043/reviewed/$M.csv
done
```

Correct shots, from the frozen copies:

```
.venv/bin/python -m vision.label_scores --labels data/vis043/reviewed/club_Wkux0r8gg0A_30.csv --match-id club_Wkux0r8gg0A_30 --fps 30 --run model data/vis043/club_Wkux0r8gg0A_30/hits/club_Wkux0r8gg0A_30/hits_0_14535.csv data/vis043/club_Wkux0r8gg0A_30/classifier/average.duckdb
.venv/bin/python -m vision.label_scores --labels data/vis043/reviewed/club_NW8Vf_ElDUo_30.csv --match-id club_NW8Vf_ElDUo_30 --fps 30 --run model data/vis043/club_NW8Vf_ElDUo_30/hits/club_NW8Vf_ElDUo_30/hits_0_19930.csv data/vis043/club_NW8Vf_ElDUo_30/classifier/average.duckdb
.venv/bin/python -m vision.label_scores --labels data/vis041/labels/club_n7MqoAVqEW4_30.csv --match-id club_n7MqoAVqEW4_30 --fps 30 --run model data/club_review/hits/club_n7MqoAVqEW4_30/hits_0_21727.csv data/club_review/typed/average.duckdb
```

The breakdowns of the rejected rows and serve rows compare `model_prelabels/` with
`reviewed/` (row position in its rally, and gap to the serve row and to the user's end
mark).
