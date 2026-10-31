# Hits from player movement only: results

The hit detector in `vision/movement.py` finds hit frames, hitters and rallies from the
two players' tracks alone (`docs/vision/movement.md`): no shuttle track. This page
measures it on the same five labelled ranges of the broadcast match as the shuttle-based
detector (`docs/results/hits.md`) and on the first minute of the club stand-in video.

## How to run
```
M=Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals
R="--range 14201 14947 1 --range 16551 17343 1 --range 81360 82116 2 --range 107560 108103 3 --range 114673 116641 3 --match $M"
.venv/bin/python -m vision.movement data/match.mp4 --cross-validate $R        # each range held out, 1-2 min
.venv/bin/python -m vision.movement data/match.mp4 --train $R                 # the shipped model, data/models/movement_hits.joblib
.venv/bin/python -m vision.movement data/match.mp4 --start 14201 --end 14947 --evaluate $M 1 --hits-dir /tmp/movement_hits
.venv/bin/python -m vision.movement data/recordings/stand_in/set1.mp4 --start 0 --end 3599 --spot-check 20 --hits-dir /tmp/movement_hits
```
(In zsh write `${=R}`.) Each range needs its player tracking files
(`python -m vision.players <video> --start N --end N`); nothing else. Detection on a
range takes seconds; the stand-in minute at 60 fps takes 4 s. `--hits-dir` keeps the
runs on this page away from the shuttle-based files in `data/hits/`, which the shot
features still read until the shuttle leaves the pipeline.

## Setup
- The same five ranges, 7 rallies and 128 labelled hits as `docs/results/hits.md`, the
  same ±5 frame match (one to one, closest first) and the same hitter rule; ±10 frames
  is reported as well. Hitter accuracy is over matched hits whose label has the
  hitter's position.
- The per-frame hit score is a gradient-boosted classifier on 20 movement features
  (speed, acceleration, direction change, box height and width changes, court `y`, of
  the player and of the opponent), trained on the labelled hits (±2 frames) of the
  ranges it is given. The rhythm step (alternating hitters, gaps of 10 to 70 frames, a
  weak start or end of a rally dropped) has fixed thresholds, chosen on the same
  ranges with each held out in turn: score 0.5, rally-end score 0.7, minimum gap 10.
- Two honest numbers: **each range held out** (the model trained on the other four,
  so every range is scored by a model that never saw it) and **sets 1-2 to set 3**
  (trained on the 80 hits of sets 1 and 2, scored on the 48 of set 3; the protocol of
  the shuttle-based page). The shipped model is trained on all five ranges, so its own
  numbers on them are in-sample and listed last.

## Results

| Range | Set | Labelled | Movement, held out: detected | matched | precision | recall | hitter | Shuttle-based (`hits.md`): precision | recall | hitter |
|---|---|---|---|---|---|---|---|---|---|---|
| 14201-14947 (rally 4) | 1 | 26 | 21 | 18 | 85.7% | 69.2% | 18/18 | 96.0% | 92.3% | 24/24 |
| 16551-17343 (rally 6) | 1 | 27 | 26 | 25 | 96.2% | 92.6% | 25/25 | 92.9% | 96.3% | 26/26 |
| 81360-82116 (rally 22) | 2 | 27 | 25 | 21 | 84.0% | 77.8% | 19/21 | 100% | 81.5% | 19/22 |
| 107560-108103 (rally 14) | 3 | 18 | 18 | 15 | 83.3% | 83.3% | 14/15 | 88.9% | 88.9% | 15/16 |
| 114673-116641 (rallies 20-22) | 3 | 30 | 30 | 24 | 80.0% | 80.0% | 24/24 | 78.8% | 86.7% | 23/26 |
| **All, each range held out, ±5** | | 128 | 120 | 103 | **85.8%** | **80.5%** | 100/103 = 97.1% | **90.5%** | **89.1%** | 107/114 = 93.9% |
| **All, each range held out, ±10** | | 128 | 120 | 109 | **90.8%** | **85.2%** | 105/109 = 96.3% | | | |

The shuttle-based "all" is on the shipped thresholds, which were tuned on sets 1-2, so
its sets 1-2 part is in-sample; its held-out part is the next table.

| Sets 1-2 to set 3 (48 held-out hits) | Detected | Matched | Precision | Recall | Hitter |
|---|---|---|---|---|---|
| Movement, ±5 | 45 | 37 | 82.2% | 77.1% | 36/37 |
| Movement, ±10 | 45 | 40 | 88.9% | 83.3% | 39/40 |
| Shuttle-based, ±5 (`hits.md`) | 51 | 42 | 82.4% | 87.5% | 38/42 |

Shipped model (trained on all five ranges) on the same ranges, in-sample, ±5:
100% / 96.2%, 96.3% / 96.3%, 96.2% / 92.6%, 100% / 94.4%, 90.0% / 90.0% (precision /
recall per range), hitter 120/120. Those are what the pipeline produces on these
ranges, not an estimate of what it does elsewhere.

**Compared with the shuttle.** Movement alone is 5 points below on precision and
9 on recall at ±5 frames when every range is held out, equal on precision and
10 points below on recall on held-out set 3, and better at naming the hitter (97% vs
94%: the hitter is the player whose movement scored, so it can't be the far player on a
near-side hit). At ±10 frames it reaches the shuttle's ±5 numbers. Frame offset of the
matched detections: median 0, mean +0.2, 77% within ±2 frames (shuttle: 96% within ±2).

**What the sample bears.** 128 hits from 7 rallies of one match: about ±6 points at
95% confidence on the held-out totals, ±10 on set 3 alone; the rallies were chosen for
length, not at random.

## Error analysis (each range held out, ±5)

Misses, 25 of 128, by the labelled type:

| Labelled type | Missed | Why |
|---|---|---|
| first shot of a rally (serve, labelled `unknown`, short or long service) | **7 of 7** | the server barely moves: a serve is a flick from standing, and the receiver stands still until it. Nothing in either player's track marks the frame. The rally is found from its second hit |
| return net (擋小球) | 5 of 17 | a soft block at the net from a lunge already in progress: the hitter's speed and direction don't change at the contact |
| net shot (放小球) | 3 of 13 | same: the lunge ends after the shot |
| drop (切球) | 3 of 11 | hit from the base with little movement, followed by a slow recovery |
| smash (殺球) | 2 of 12 | jump smashes: the box bottom rises and falls, which the position reads as moving backward then forward |
| push (推球) | 2 of 16 | |
| rush, cross net, driven flight | 1 each of 2, 9, 2 | |
| lift, clear, passive drop, wrist smash, drive, defensive lob | 0 of 19, 7, 6, 4, 2, 1 | the big movements (a lift from a lunge, a clear from the back) are the ones movement sees |

Of the 25 misses, 7 are serves and 11 are net-area shots. The shuttle detector missed
serves too (the broadcast cuts to the server), so on these ranges both detectors
start a rally at its second hit.

False positives, 17 of 120: 12 inside a labelled rally, 5 outside. The ones inside are
an extra hit squeezed into a long gap between two real hits (a slow clear exchange:
the chain prefers two 30-frame gaps over one 60-frame gap when a player's recovery
movement scores) or a hit placed 6-10 frames off a real one, which then counts as a
miss and a false positive. The ones outside are the first steps after the last shot of
a rally: a player walking to pick the shuttle up scores like a slow approach. The
rally-end rule (both players below 0.8 m/s for 15 frames) ends the rally there, but the
walk itself is inside it.

Rally segmentation, each range held out (labelled rallies as in `hits.md`):

| Range | Labelled rallies (first - last hit) | Movement rallies (first hit - end) |
|---|---|---|
| 14201-14947 | 14231-14917 | 14254-14746 and 14775-14947: one labelled rally split where two consecutive hits were missed (a gap over 70 frames) |
| 16551-17343 | 16581-17313 | 16601-17343 |
| 81360-82116 | 81390-82086 | 81410-82116 |
| 107560-108103 | 107590-108073 | 107629-108103 |
| 114673-116641 | 114703-115082, 115589-115795, 116493-116611 | 114719-115167, 115606-115902, 116515-116641 |

With the shipped model every range gives the labelled rallies, none split and none
merged, and no extra one (the shuttle detector added a 1-hit "rally" at a pickup).

## Club stand-in video, first minute (60 fps, 1920x1080)
`data/recordings/stand_in/set1.mp4` frames 0-3599, the player tracks of
`docs/results/stand_in.md`, settings scaled to 60 fps (gaps 20-140 frames, score
smoothing 6, box sizes at 1280 px). No labels, so the measure is the 20-hit spot check:
`data/checks/hits/stand_in_set1_movement_spot_check_0_3599.jpg`, 20 random detections
(seed 1), each as the frame 12 before, the hit frame and 12 after, cropped around the
hitter's box. Detected: 5 rallies, 39 hits (the shuttle detector: 9 rallies, 49 hits,
of which the spot check found 6 of 20 to be between-rally floor bounces, pick-ups and
false tracks).

| Frame | Detected | Hit? | Hitter | Note |
|---|---|---|---|---|
| 156 | me | yes | right | near-side lunge, shuttle at the racket |
| 239 | opp | yes | right | far-side lunge at the net |
| 361 | opp | yes | right | far-side net shot |
| 624 | me | yes | right | near player reaching behind |
| 774 | me | unsure | | near player mid-court, racket hidden by the body |
| 816 | opp | yes | right | far-side overhead, shuttle above the player 12 frames later |
| 893 | me | yes | right | near-side lunge at the net, shuttle at the racket |
| 1125 | opp | yes | right | far-side overhead |
| 1183 | me | yes | right | near-side low hit, shuttle at the racket |
| 1238 | me | yes | right | near-side low lunge |
| 1482 | me | **no** | | near player walking upright, racket at the side: after the rally |
| 1787 | me | **no** | | near player standing at the back, racket in hand: before the serve |
| 1876 | me | unsure | | near player looking up with the racket back; the shuttle is still above him 12 frames later, so the hit is later than the detection |
| 1949 | me | yes | right | near-side overhead, arm up 12 frames before |
| 2010 | opp | yes | right | far-side lunge at the net, shuttle above |
| 2055 | me | yes | right | near-side lunge at the net, shuttle at the racket 12 frames later |
| 2108 | opp | yes | right | far-side net exchange |
| 2161 | me | yes | right | near-side low dig |
| 2440 | me | unsure | | the near player fills the crop (next to the camera); nothing to judge |
| 3186 | opp | **no** | | far player standing at the net, racket down: between rallies (rally 5, 3 hits, is the pick-up and the walk before a serve) |

**14 real hits, 3 not hits, 3 unsure.** The hitter is right on all 14 (the shuttle
detector: 13 real hits of 20, hitter right on 10, 6 not hits at floor bounces, pick-ups
and false shuttle tracks). The movement detector's non-hits are players standing or
walking between points, not shuttle artefacts, and they sit at the ends of rallies.
One detection (1876) is early: the player had set himself under a high clear and the
movement stopped before the contact.

Rally 5 (frames 3123-3355, 3 hits) is not a rally: the near player scoops the shuttle
off the floor, the far player waits, and the near player bends for the serve; the
serve itself (frame 3247 on the shuttle sheet) is not among the three. That is the
rally-start weakness from the error analysis: nothing in the tracks separates "walk to
the shuttle and bend down" from an approach to a net shot.

## What this changes downstream
Hits start a rally at its second shot (the serve is not detected), the hit point is the
hitter's feet at the detected frame, and the landing of the last shot of a rally is
unknown. For the shot features that means rallies without a serve row, and 77% of
hit frames within ±2 frames of the contact against 96% from the shuttle: the feet at a
frame 3-5 frames off are 0.1-0.5 m off on a lunge. The timing-jitter table in
`docs/results/classifier_feet.md` is the cost of that on shot typing.

## Limitations
- One broadcast match, 128 hits, one club minute without labels; the model is trained
  on 5 ranges of one match and the stand-in numbers come from it unchanged.
- Serves are not found; a rule for the first hit (both players still, then the
  receiver's first step) would need the serve's timing from the receiver's reaction,
  which is 10-20 frames after the contact.
- The between-rally walk after a point is sometimes read as 1-3 extra hits; the
  rally-end rule is stillness, which players don't show while picking the shuttle up.
- Everything is at the players' feet in the court plane; a jump smash reads as a
  backward-forward step.
