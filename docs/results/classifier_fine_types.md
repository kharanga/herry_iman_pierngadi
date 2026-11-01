# Finer shot types from the feet: results

The shipped classifier names 7 shot groups. ShuttleSet labels 18 types (plus `unknown`). This page measures
how many of those 18 the feet-only classifier can tell apart, before any club games are labelled. The shipped
model is unchanged: it is a measurement only.

## How to run
```
.venv/bin/python -m shots.fine_types
```
1 min 52 s on the dev MacBook Air, CPU only (2 threads): 11 label sets × 5 folds. Every number below comes from
`data/models/fine_type_comparison.json`. Method: `docs/analytics/classifier.md`, section `shots/fine_types.py`.

## Setup
- The same rows as the shipped model: 31,544 labelled ShuttleSet shots from 44 matches (`unknown` left out,
  hitter and landing point present).
- The same 16 `feet` features, the same `HistGradientBoostingClassifier(random_state=0)`, and the same
  `GroupKFold(5)` by match, so no match is in both train and test.
- "Mapped back to the 7 groups" scores a finer model's predictions after renaming each type to its group. It
  shows whether training on finer labels costs group accuracy.
- A candidate split **holds up** when both sides reach F1 0.5 and the model tells the two sides apart at least
  5 points better than always picking the commonest side. That comparison counts only shots whose true and
  predicted labels are both in the group. Both thresholds were fixed before the run.

## Results
### Accuracy
| Labels | Count | Accuracy (mean ± sd over 5 folds) | Mapped back to the 7 groups |
|---|---|---|---|
| 7 groups (shipped) | 7 | 86.6% ± 0.7% | 86.6% |
| 18 types | 18 | 73.6% ± 1.0% | 85.8% |
| middle: 7 groups + the splits that hold up | 10 | 82.8% ± 1.2% | 86.4% |

### Per type (18-type model, out of fold)
| Group | Type | Shots | Precision | Recall | F1 |
|---|---|---|---|---|---|
| serve | short service | 2,033 | 0.99 | 1.00 | 1.00 |
| serve | long service | 360 | 0.97 | 0.97 | 0.97 |
| clear | clear | 2,734 | 0.94 | 0.94 | 0.94 |
| drop | drop | 1,924 | 0.57 | 0.61 | 0.59 |
| drop | passive drop | 1,280 | 0.50 | 0.42 | 0.46 |
| smash | smash | 1,921 | 0.59 | 0.67 | 0.62 |
| smash | wrist smash | 1,407 | 0.38 | 0.32 | 0.34 |
| net | net shot | 5,834 | 0.89 | 0.90 | 0.90 |
| net | return net | 3,205 | 0.76 | 0.80 | 0.78 |
| net | cross-court net shot | 1,201 | 0.71 | 0.64 | 0.67 |
| net | rush | 218 | 0.54 | 0.44 | 0.48 |
| lift | lob | 4,985 | 0.77 | 0.85 | 0.81 |
| lift | defensive return lob | 274 | 0.49 | 0.43 | 0.46 |
| drive | drive | 628 | 0.46 | 0.48 | 0.47 |
| drive | driven flight | 63 | 0.00 | 0.00 | 0.00 |
| drive | back-court drive | 442 | 0.40 | 0.33 | 0.36 |
| drive | push | 2,682 | 0.56 | 0.48 | 0.52 |
| drive | defensive return drive | 353 | 0.30 | 0.24 | 0.26 |

### Confusion (18-type model, out of fold; rows true, columns predicted)
Codes: SS short service, LS long service, CL clear, DR drop, PD passive drop, SM smash, WS wrist smash, NS net shot, RN return net, XN cross-court net shot, RU rush, LO lob, DL defensive return lob, DV drive, DF driven flight, BD back-court drive, PU push, DD defensive return drive.

| true \ predicted | SS | LS | CL | DR | PD | SM | WS | NS | RN | XN | RU | LO | DL | DV | DF | BD | PU | DD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SS | 2029 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| LS | 8 | 349 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CL | 0 | 0 | 2563 | 24 | 18 | 15 | 20 | 1 | 5 | 0 | 0 | 35 | 11 | 3 | 1 | 10 | 24 | 4 |
| DR | 0 | 0 | 25 | 1167 | 412 | 59 | 123 | 22 | 76 | 10 | 0 | 0 | 0 | 2 | 3 | 21 | 3 | 1 |
| PD | 0 | 0 | 13 | 562 | 544 | 26 | 48 | 7 | 24 | 3 | 0 | 2 | 0 | 2 | 1 | 47 | 1 | 0 |
| SM | 0 | 0 | 23 | 62 | 28 | 1279 | 417 | 0 | 0 | 0 | 3 | 2 | 0 | 37 | 1 | 61 | 8 | 0 |
| WS | 0 | 1 | 17 | 104 | 38 | 664 | 447 | 3 | 4 | 3 | 3 | 0 | 0 | 42 | 2 | 70 | 7 | 2 |
| NS | 2 | 0 | 0 | 21 | 1 | 0 | 0 | 5264 | 272 | 206 | 0 | 3 | 0 | 5 | 30 | 0 | 29 | 1 |
| RN | 1 | 0 | 1 | 50 | 6 | 2 | 1 | 344 | 2573 | 90 | 0 | 3 | 11 | 51 | 5 | 1 | 10 | 56 |
| XN | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 199 | 207 | 765 | 0 | 0 | 0 | 4 | 4 | 0 | 6 | 10 |
| RU | 0 | 0 | 0 | 0 | 0 | 3 | 5 | 7 | 0 | 1 | 96 | 2 | 0 | 43 | 2 | 0 | 57 | 2 |
| LO | 0 | 0 | 21 | 0 | 2 | 2 | 0 | 8 | 3 | 1 | 3 | 4217 | 22 | 1 | 1 | 0 | 699 | 5 |
| DL | 0 | 0 | 29 | 1 | 0 | 1 | 3 | 0 | 24 | 0 | 0 | 29 | 119 | 6 | 0 | 1 | 17 | 44 |
| DV | 0 | 0 | 5 | 1 | 0 | 23 | 22 | 6 | 69 | 3 | 31 | 6 | 3 | 304 | 1 | 5 | 107 | 42 |
| DF | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 38 | 11 | 1 | 1 | 0 | 0 | 1 | 0 | 2 | 7 | 1 |
| BD | 0 | 0 | 12 | 26 | 49 | 96 | 99 | 0 | 2 | 0 | 0 | 1 | 0 | 5 | 1 | 145 | 2 | 4 |
| PU | 0 | 5 | 20 | 3 | 0 | 5 | 2 | 22 | 12 | 1 | 41 | 1151 | 22 | 80 | 8 | 2 | 1285 | 23 |
| DD | 0 | 0 | 8 | 2 | 0 | 4 | 4 | 2 | 92 | 1 | 0 | 5 | 55 | 69 | 0 | 1 | 27 | 83 |

### Each candidate split on its own (7 groups + that split)
| Split | Sides: F1 (shots) | Telling the sides apart | Commonest side | Gain | Overall, as groups | Holds up |
|---|---|---|---|---|---|---|
| push vs drive | drive 0.51 (1,486); push 0.53 (2,682) | 89.5% of 2,215 | 63.7% | +25.8 pts | 86.5% | yes |
| rush vs net | net 0.97 (10,240); rush 0.42 (218) | 99.9% of 10,068 | 99.2% | +0.8 pts | 86.5% | no |
| return net vs net shot | net 0.91 (7,253); return net 0.78 (3,205) | 90.6% of 10,000 | 70.1% | +20.6 pts | 86.3% | yes |
| cross-court net shot | cross-court net shot 0.67 (1,201); net 0.92 (9,257) | 93.0% of 10,045 | 88.2% | +4.8 pts | 86.6% | no |
| wrist smash vs smash | smash 0.62 (1,921); wrist smash 0.34 (1,407) | 61.0% of 2,776 | 60.7% | +0.2 pts | 86.6% | no |
| passive drop vs drop | drop 0.60 (1,924); passive drop 0.45 (1,280) | 63.5% of 2,657 | 58.8% | +4.8 pts | 86.5% | no |
| defensive return lob vs lob | defensive return lob 0.46 (274); lift 0.81 (4,985) | 99.1% of 4,351 | 97.0% | +2.2 pts | 86.4% | no |
| short vs long service | long service 0.99 (360); short service 1.00 (2,033) | 99.7% of 2,392 | 85.0% | +14.8 pts | 86.4% | yes |

### Middle set (10 labels; split off: push, return net, short service, long service)
| Label | Shots | Precision | Recall | F1 |
|---|---|---|---|---|
| net | 7,253 | 0.91 | 0.91 | 0.91 |
| lift | 5,259 | 0.77 | 0.84 | 0.80 |
| smash | 3,328 | 0.83 | 0.87 | 0.85 |
| return net | 3,205 | 0.77 | 0.80 | 0.79 |
| drop | 3,204 | 0.87 | 0.85 | 0.86 |
| clear | 2,734 | 0.95 | 0.93 | 0.94 |
| push | 2,682 | 0.58 | 0.48 | 0.52 |
| short service | 2,033 | 0.99 | 1.00 | 1.00 |
| drive | 1,486 | 0.56 | 0.48 | 0.52 |
| long service | 360 | 0.97 | 0.99 | 0.98 |

In the split table, the side that keeps the group name (e.g. `net`, `lift`) is the rest of the group without the
split-off type.

## What the numbers say
- **All 18 types: 73.6%,** against 86.6% for the 7 groups. The 18-type model is about as good at groups
  (85.8% mapped back), so the loss is all inside the groups. Eight types are below F1 0.5. `driven flight`
  (63 shots) is never predicted.
- **The serve split is reliable.** Short against long service is 99.7% right against 85.0% for always
  guessing short, with F1 0.99 and 1.00.
- **Return net against net shot is reliable.** It is 90.6% right against 70.1%, with F1 0.78 for return
  net. Feature importances were not measured here, so which features carry it is open.
- **Push against drive separates within the group** (89.5% against 63.7%), but both sides stay weak, at
  F1 0.52 each. The main error is not drive for push: 1,151 of 2,682 pushes are called `lob`. That error is
  also there in the shipped `drive` group (F1 0.59). Splitting push off costs nothing at group level
  (86.5%), but each push count would be wrong about half the time.
- **Not reliable from the feet: wrist smash vs smash** (61.0% against 60.7%, no better than guessing) and
  **passive drop vs drop** (+4.8 points, passive drop F1 0.45). What separates each pair is the swing and
  the shuttle speed, which feet positions don't show.
- **Cross-court net shot is borderline** (F1 0.67, +4.8 points, just under the bar). The report already
  counts straight against cross-court for every net shot from `is_cross`, so a separate type adds little.
- **Rush and defensive return lob are too rare to learn** (218 and 274 shots). Their high "telling apart"
  numbers only reflect a 99% and 97% majority, and their F1 is 0.42 and 0.46.
- **The middle set (10 labels) scores 82.8%:** the 7 groups with push, return net and both services split
  off. It still scores 86.4% as groups, so it gives the four extra labels at no group-level cost.

## Recommendation
- **Split:** short vs long service, and return net vs net shot. Both are reliable from feet positions alone.
- **Optional:** push vs drive. It passes the bar but only just (F1 0.52 and 0.51), and only as well as the
  current `drive` group. Report it only if a half-right push count is useful.
- **Keep merged:** smash/wrist smash, drop/passive drop, lift/defensive return lob, net/rush, and the
  drive family (`drive`, `driven flight`, `back-court drive`, `defensive return drive`; F1 0.00 to 0.47).
  Cross-court net shots stay as the straight/cross count, not a type.
- For club labelling, this means labelling the 7 groups plus *serve short/long* and *net: return or not*.
  Labelling the rest by type gives the classifier nothing it can learn from the feet.

## Caveats
- All of this is ShuttleSet: true feet positions at exact hit frames. On club footage the feet come from
  tracking, and the hit frames come from the movement detector. `docs/results/classifier_feet.md` measured
  ±5 frames of hit-timing jitter at 3.4 points for the groups; finer splits are likely hit harder.
- On club footage the serve row is inferred, 0.76 s before the first detected hit (`docs/results/no_shuttle.md`).
  Its flight time is therefore a constant, and short vs long has to come from where the receiver stands.
  The 99.7% above is for real serve frames and is not measured for inferred ones.
- The middle set's 82.8% is scored on the same match folds that chose its splits, so it is slightly optimistic.
  Choosing the splits inside each fold would remove that.
- Pro play: what a club player's push or return net looks like from the feet may differ. Labelled club
  games are the check.
