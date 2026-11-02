# Retraining with the reviewed club game

The user reviewed a model-labelled club game (`n7MqoAVqEW4`, 30 fps, first 12:05;
`docs/results/club_review.md`). This page tests whether adding it to training helps. It
went into both models, with nothing else changed:
- **Hit scorer:** the nine pro matches plus Lin, Ding and n7. The clubs together carry
  1/3 of the sample weight, 1/9 each. The thresholds were tuned on its 12 training
  sources (`--save --tune`) and came out 0.55 / 0.6, the same as before. Saved as
  `data/models/movement_hits_nine_plus_lin_ding_n7_tuned.joblib`.
- **Club classifier:** Lin + Ding + n7, 912 labelled hits with feet points (704 before),
  averaged 50/50 with the shipped pro classifier, the AVG of
  `docs/results/shot_classifier_club.md`.

n7's labels were frozen for training in `data/vis041/labels/`: 254 kept hits, all
reviewed, and 80 rejected.

**A bias to keep in mind.** n7's hits were reviewed from the model's proposals: the user
kept the model's frame unless it was clearly off. So its hit frames sit close to where
the model already puts them, and teach the scorer its own timing. Its shot types,
hitters and missed and false hits were judged by the user. The two test videos were
labelled from a blank start, so they don't carry this bias.

**Test:** test game 1 (`UHKzdJqgFwk`, labelled range, 110 hits) and stand-in set 1
(172 hits), both at 30 fps:
- the old (N+L+D) and new (N+L+D+n7) scorer, each alone and inside the combined
  detector;
- each typed by the old (club = L+D) and new (club = L+D+n7) AVG classifier.

The old scorer with the old AVG reproduces `docs/results/shot_classifier_club.md`
exactly.

## Correct shots, ±0.50 s (±0.25 s in brackets)

| Scorer | Hits | AVG classifier | Test game 1 | Set 1 | Both |
|---|---|---|---|---|---|
| Old (N+L+D) | alone | old (L+D) | 62/125 = 49.6% (47.7%) | 100/214 = 46.7% (41.9%) | 162/339 = 47.8% |
| Old (N+L+D) | alone | **new (L+D+n7)** | **64/125 = 51.2%** (**49.2%**) | **102/214 = 47.7%** (**42.8%**) | **166/339 = 49.0%** |
| Old (N+L+D) | combined | old (L+D) | 62/123 = 50.4% (46.5%) | 98/201 = 48.8% (42.5%) | 160/324 = 49.4% |
| Old (N+L+D) | combined | **new (L+D+n7)** | **63/123 = 51.2%** (**47.3%**) | **101/201 = 50.2%** (**43.9%**) | **164/324 = 50.6%** |
| New (N+L+D+n7) | alone | old (L+D) | 57/126 = 45.2% (43.4%) | 89/225 = 39.6% (35.8%) | 146/351 = 41.6% |
| New (N+L+D+n7) | alone | new (L+D+n7) | 58/126 = 46.0% (44.2%) | 91/225 = 40.4% (37.5%) | 149/351 = 42.5% |
| New (N+L+D+n7) | combined | old (L+D) | 57/124 = 46.0% (42.6%) | 94/204 = 46.1% (39.6%) | 151/328 = 46.0% |
| New (N+L+D+n7) | combined | new (L+D+n7) | 59/124 = 47.6% (44.2%) | 99/204 = 48.5% (41.9%) | 158/328 = 48.2% |

## Hits, and shot type

The classifier doesn't move the hits. At ±0.50 s, on detected hits:

| Scorer | Hits | Test game 1: found / precision / hitter | Set 1: found / precision / hitter |
|---|---|---|---|
| Old | alone | 89.1% / 90.7% / 96.9% | 80.8% / 82.2% / 94.2% |
| Old | combined | 78.2% / 89.6% / 96.5% | 69.8% / 83.9% / 95.8% |
| New | alone | 85.5% / 87.9% / 95.7% | 82.6% / 80.2% / 95.1% |
| New | combined | 75.5% / 87.4% / 95.2% | 68.6% / 81.9% / 97.5% |

Shot type with serves at ±0.50 s, old AVG → new AVG:
- **Test game 1:** old scorer alone 66.0 → 68.1%, combined 71.3 → 72.4%; new scorer alone
  62.6 → 63.7%, combined 67.1 → 69.4%.
- **Set 1:** old scorer alone 72.5 → 73.9%, combined 76.6 → 78.9%; new scorer alone
  66.4 → 67.9%, combined 73.4 → 77.3%.

## What it says

- **n7 helps the shot classifier.** The new AVG beats the old one in all 8 pairings, by
  1-5 correct shots (+1 to +4 percentage points of shot type). Summed over both videos,
  the old scorer with the new AVG is the best of everything so far:
  - alone 166/339 (49.0%), against 162/339;
  - combined 164/324 (50.6%), against 160/324.
- **n7 doesn't help the hit scorer.** With n7 in, the scorer loses 2-11 correct shots,
  from fewer hits found and lower precision on test game 1, and more false hits on set 1.
  One possible reason is the bias above: n7 teaches the scorer the model's own hit
  timing. Another is that n7's share of the weight dilutes the other two clubs. This run
  doesn't tell them apart.
- **Recommendation:** keep the current hit scorer (nine + Lin + Ding) and switch the club
  classifier inside AVG to Lin + Ding + n7. This is two videos, 282 labelled hits; a
  1-3 shot difference is noise, but the classifier's gain is consistent in every pairing.

## How to run

zsh, from the project root. `data/vis041/labels/` holds the frozen copy of n7's reviewed labels.

The new hit scorer, tuned on its 12 training sources (it prints `--hit-score-min 0.55 --rally-end-score-min 0.6`):

```
O=data/vis041
LIN=(--club data/recordings/club/q3pfhnvKPig.mp4 data/labels/club_q3pfhnvKPig.csv)
DING=(--club data/recordings/club/E6SRRmvJ6dI.mp4 data/labels/club_E6SRRmvJ6dI.csv)
N7=(--club data/recordings/club/n7MqoAVqEW4_30.mp4 $O/labels/club_n7MqoAVqEW4_30.csv)
OLD_SCORER=data/models/movement_hits_nine_plus_lin_ding_tuned.joblib
NEW_SCORER=data/models/movement_hits_nine_plus_lin_ding_n7_tuned.joblib
.venv/bin/python -m vision.hit_training data/hit_training/matches.txt --tune $LIN $DING $N7 --club-weight 0.3333333333 --save $NEW_SCORER
```

n7's club classifier training rows, built from its reviewed hits (end frame 21727):

```
N7V=data/recordings/club/n7MqoAVqEW4_30.mp4
N7DB=data/flight/club_n7MqoAVqEW4_30/scout.duckdb
mkdir -p data/flight/club_n7MqoAVqEW4_30
.venv/bin/python -m vision.flight_types --train $N7V $O/labels/club_n7MqoAVqEW4_30.csv $N7DB 21727 --hits-out data/flight/hits
cp data/scout.duckdb $N7DB
.venv/bin/python -m vision.features $N7V --match-id club_n7MqoAVqEW4_30 --range 0 21727 1 --database $N7DB --hits-dir data/flight/hits
TRAIN_OLD=(--train data/recordings/club/q3pfhnvKPig.mp4 data/labels/club_q3pfhnvKPig.csv data/flight/club_q3pfhnvKPig/scout.duckdb 29710 --train data/recordings/club/E6SRRmvJ6dI.mp4 data/labels/club_E6SRRmvJ6dI.csv data/flight/club_E6SRRmvJ6dI/scout.duckdb 31799)
TRAIN_NEW=($TRAIN_OLD --train $N7V $O/labels/club_n7MqoAVqEW4_30.csv $N7DB 21727)
```

The 16 test runs: per test video (video : match id : end frame : run folder), per scorer
(name : model : thresholds) and per mode, hits, rows, and both AVG typings. Then one
report per video, with the 8 runs in this order:

```
for target in data/recordings/club/UHKzdJqgFwk_30.mp4:club_UHKzdJqgFwk_30:63443:UHKzdJqgFwk_30 data/recordings/stand_in/set1_30.mp4:stand_in_set1_30:13117:stand_in_set1_30; do
  IFS=: read V M E dir <<< "$target"
  for scorer in old:$OLD_SCORER:0.55:0.6 new:$NEW_SCORER:0.55:0.6; do
    IFS=: read sname model hh rr <<< "$scorer"
    for mode in alone combined; do
      extra=(); [ $mode = combined ] && extra=(--rallies-from data/models/movement_hits.joblib)
      T=$O/$dir/${sname}_scorer_$mode
      .venv/bin/python -m vision.movement $V --start 0 --end $E --model $model --hit-score-min $hh --rally-end-score-min $rr $extra --hits-dir $T/hits
      cp data/scout.duckdb $T/scout.duckdb
      .venv/bin/python -m vision.features $V --match-id $M --range 0 $E 1 --database $T/scout.duckdb --hits-dir $T/hits
      .venv/bin/python -m vision.club_classifier $TRAIN_OLD --predict $V $T/scout.duckdb $E $T/old_classifier
      .venv/bin/python -m vision.club_classifier $TRAIN_NEW --predict $V $T/scout.duckdb $E $T/new_classifier
    done
  done
  args=()
  for sname in old new; do for mode in alone combined; do for cname in old new; do
    T=$O/$dir/${sname}_scorer_$mode
    args+=(--run ${sname}_scorer_${mode}_${cname}_avg $T/hits/$M/hits_0_$E.csv $T/${cname}_classifier/average.duckdb)
  done; done; done
  .venv/bin/python -m vision.label_scores --labels data/labels/$M.csv --match-id $M --fps 30 $args > $O/$dir/report.md
done
```

The run names in the reports are `<scorer>_scorer_<mode>_<classifier>_avg`, for example
`old_scorer_combined_new_avg` = the old scorer inside the combined detector, typed by the new AVG.
