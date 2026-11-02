# Shot classifier from pro and club shots: two candidates

The shipped pro-trained classifier is best on set 1's view, and the club-trained one on
test game 1's (`docs/results/shuttle_features.md`). Two candidates were fixed before
testing, both on feet features only:
- **P+C:** one classifier trained on ShuttleSet's 31,544 labelled shots plus Lin + Ding's
  704 club labels, the club rows carrying 1/3 of the sample weight.
- **AVG:** the shipped pro classifier and the Lin + Ding club classifier, their
  probabilities over the 8 shot details averaged 50/50.

They are compared with the two they come from:
- **Pro:** the shipped `data/models/shot_type_classifier.joblib`.
- **Club:** the Lin + Ding classifier. It reproduces the shuttle experiment's club
  feet-only numbers exactly.

All are gradient-boosted trees, seed 0, over the same 16 feet features. Nothing was tuned
on the test videos.

The hits come from the nine + Lin + Ding tuned scorer at 30 fps, alone and inside the
combined detector (`docs/results/club_test.md`, "At 30 fps"). Only the shot types differ
between candidates. Scores are against the user's labels at ±0.50 s
(`vision.label_scores`).

## Correct shots

Correct shots are matched and typed right, divided by labelled hits plus false hits
inside the labelled rallies.

| Hits | Classifier | Test game 1 (110 hits) | Set 1 (172 hits) | Both |
|---|---|---|---|---|
| Alone | Pro | 49/125 = 39.2% | **102/214 = 47.7%** | 151/339 = 44.5% |
| Alone | Club | **62/125 = 49.6%** | 94/214 = 43.9% | 156/339 = 46.0% |
| Alone | P+C | 57/125 = 45.6% | 99/214 = 46.3% | 156/339 = 46.0% |
| Alone | **AVG** | **62/125 = 49.6%** | 100/214 = 46.7% | **162/339 = 47.8%** |
| Combined | Pro | 48/123 = 39.0% | 98/201 = 48.8% | 146/324 = 45.1% |
| Combined | Club | 60/123 = 48.8% | 92/201 = 45.8% | 152/324 = 46.9% |
| Combined | P+C | 54/123 = 43.9% | **100/201 = 49.8%** | 154/324 = 47.5% |
| Combined | **AVG** | **62/123 = 50.4%** | 98/201 = 48.8% | **160/324 = 49.4%** |

## Shot type on matched hits

At ±0.50 s, hits alone (combined):

| Classifier | Test game 1, with serves | Test game 1, serves left out | Set 1, with serves | Set 1, serves left out |
|---|---|---|---|---|
| Pro | 52.1% (55.2%) | 53.7% (51.2%) | **73.9%** (76.6%) | **78.4%** (73.6%) |
| Club | **66.0%** (69.0%) | **69.5%** (66.2%) | 68.1% (71.9%) | 71.6% (68.2%) |
| P+C | 60.6% (62.1%) | 63.4% (58.8%) | 71.7% (**78.1%**) | 75.9% (**75.5%**) |
| AVG | **66.0%** (**71.3%**) | **69.5%** (**68.8%**) | 72.5% (76.6%) | 76.7% (73.6%) |

Per type, hits alone, typed right of matched:

| Type | Test game 1: Pro | Club | P+C | AVG | Set 1: Pro | Club | P+C | AVG |
|---|---|---|---|---|---|---|---|---|
| lift | 10/24 | 21/24 | 12/24 | 20/24 | 20/24 | 23/24 | 21/24 | 23/24 |
| drop | 6/11 | 7/11 | 7/11 | **8/11** | **17/22** | 12/22 | 15/22 | 15/22 |
| smash | 3/12 | 5/12 | 5/12 | 5/12 | 1/2 | 1/2 | 1/2 | 1/2 |
| drive | 1/5 | 0/5 | 2/5 | 0/5 | 3/15 | 1/15 | 1/15 | 1/15 |
| net | 15/20 | 15/20 | 16/20 | 15/20 | 38/39 | 35/39 | 38/39 | 37/39 |
| clear | 9/10 | 9/10 | 10/10 | 9/10 | 12/14 | 11/14 | 12/14 | 12/14 |
| serve | 5/12 | 5/12 | 5/12 | 5/12 | 11/22 | 11/22 | 11/22 | 11/22 |

**Drops and smashes,** labelled row against predicted, hits alone:
- **Test game 1, smash (12 matched):**
  - pro: 3 smash, 4 drop, 3 clear, 2 drive;
  - club and AVG: 5 smash, 3 clear, 2 drive, 1 drop, 1 net;
  - P+C: 5 smash, 3 drop, 2 clear, 2 drive.
- **Test game 1, drop (11):**
  - pro: 6 drop, 3 clear, 2 smash;
  - club: 7 drop, 2 clear, 2 smash;
  - AVG: 8 drop, 2 smash, 1 clear;
  - P+C: 7 drop, 3 smash, 1 clear.
- **Set 1, drop (22):**
  - pro: 17 drop, 2 net, 1 each clear, serve and smash;
  - club: 12 drop, 6 smash, 2 drive, 1 lift, 1 serve;
  - P+C: 15 drop, 3 smash, 2 net, 1 clear, 1 serve;
  - AVG: 15 drop, 4 smash, 1 each drive, net and serve.
- **Test game 1, lift (24):** pro and P+C type 10 as drives, club 1, AVG 3.
- Set 1 has only 2 labelled smashes; each classifier gets 1 right.

## At ±0.25 s

The same prediction databases scored with the narrower window, the per-type table
included (`vision.label_scores --breakdown-tolerance 0.25`). Nothing else changed.

The classifier doesn't move the hits, so hits found and precision are the same for all
four:

| Hits | Video | Hits found (detected) | Precision in labelled rallies (detected) |
|---|---|---|---|
| Alone | Test game 1 | 91/110 = 82.7% | 91/107 = 85.0% |
| Alone | Set 1 | 129/172 = 75.0% | 129/167 = 77.2% |
| Combined | Test game 1 | 80/110 = 72.7% | 80/95 = 84.2% |
| Combined | Set 1 | 110/172 = 64.0% | 110/141 = 78.0% |

Correct shots:

| Hits | Classifier | Test game 1 | Set 1 | Both |
|---|---|---|---|---|
| Alone | Pro | 47/128 = 36.7% | **95/222 = 42.8%** | 142/350 = 40.6% |
| Alone | Club | **61/128 = 47.7%** | 88/222 = 39.6% | 149/350 = 42.6% |
| Alone | P+C | 55/128 = 43.0% | 92/222 = 41.4% | 147/350 = 42.0% |
| Alone | **AVG** | **61/128 = 47.7%** | 93/222 = 41.9% | **154/350 = 44.0%** |
| Combined | Pro | 45/129 = 34.9% | 90/214 = 42.1% | 135/343 = 39.4% |
| Combined | Club | 58/129 = 45.0% | 86/214 = 40.2% | 144/343 = 42.0% |
| Combined | P+C | 51/129 = 39.5% | **91/214 = 42.5%** | 142/343 = 41.4% |
| Combined | **AVG** | **60/129 = 46.5%** | **91/214 = 42.5%** | **151/343 = 44.0%** |

Shot type on matched hits, hits alone (combined):

| Classifier | Test game 1, with serves | Test game 1, serves left out | Set 1, with serves | Set 1, serves left out |
|---|---|---|---|---|
| Pro | 53.4% (55.6%) | 54.5% (52.0%) | **74.2%** (78.3%) | **79.4%** (76.0%) |
| Club | **69.3%** (71.6%) | **72.7%** (69.3%) | 68.8% (74.8%) | 72.9% (72.0%) |
| P+C | 62.5% (63.0%) | 64.9% (60.0%) | 71.9% (**79.1%**) | 76.6% (**77.0%**) |
| AVG | **69.3%** (**74.1%**) | **72.7%** (**72.0%**) | 72.7% (**79.1%**) | 77.6% (**77.0%**) |

Per type, hits alone, typed right of matched:

| Type | Test game 1: Pro | Club | P+C | AVG | Set 1: Pro | Club | P+C | AVG |
|---|---|---|---|---|---|---|---|---|
| lift | 10/24 | 21/24 | 12/24 | 20/24 | 19/23 | 22/23 | 20/23 | 22/23 |
| drop | 5/8 | 7/8 | 6/8 | **8/8** | **15/19** | 11/19 | 13/19 | 13/19 |
| smash | 3/12 | 5/12 | 5/12 | 5/12 | 1/1 | 1/1 | 1/1 | 1/1 |
| drive | 1/4 | 0/4 | 2/4 | 0/4 | 3/14 | 1/14 | 1/14 | 1/14 |
| net | 14/19 | 14/19 | 15/19 | 14/19 | 36/37 | 33/37 | 36/37 | 35/37 |
| clear | 9/10 | 9/10 | 10/10 | 9/10 | 11/13 | 10/13 | 11/13 | 11/13 |
| serve | 5/11 | 5/11 | 5/11 | 5/11 | 10/21 | 10/21 | 10/21 | 10/21 |

- **The ranking holds at ±0.25 s.** AVG has the most correct shots over both videos
  (154 of 350 alone, 151 of 343 combined). It equals the club classifier on test game 1
  alone, and on set 1 combined it ties P+C (91 of 214), one shot ahead of pro.
- Fewer hits match in the narrower window, so every count is a little lower than at
  ±0.50 s. The differences between classifiers barely change: AVG over pro on test game
  1 is +14 shots alone and +15 combined.

## What it says

- **AVG is the steadiest candidate.**
  - On test game 1 it matches or beats the club classifier (49.6% alone, 50.4% combined).
  - On set 1 it comes within 1 point of the pro one alone (46.7% against 47.7%) and
    equals it combined (48.8%).
  - Summed over both videos it has the most correct shots: 162 of 339 alone, 160 of 324
    combined.
- **P+C sits between the two.** The 31,544 pro shots outweigh the club ones even at 1/3
  of the weight, so it keeps typing club lifts as drives (10 of 24 on test game 1). Its
  best result is set 1 combined, 100 of 201, two shots ahead of pro and AVG.
- **Drops:** averaging recovers most of the drops the club classifier loses to smash on
  set 1 (15 of 22 against 12, pro 17), and it is the best on test game 1 (8 of 11).
- **Smashes and drives stay weak** for every candidate: 5 of 12 smashes at best on test
  game 1, and drives 0-3 of 15 on set 1.
- **Small numbers:** two videos, 282 labelled hits in all. A 1-3 shot difference is noise,
  but AVG's lead over pro on test game 1 (13-14 shots) is not.

## Adding the reviewed club game n7 (test game 1 and set 1, 30 fps)

The club classifier inside AVG was retrained on Lin + Ding + n7 (912 labelled hits with
feet points, 704 before). The new AVG beats the old one in all 8 pairings of scorer,
detector and classifier, by 1-5 correct shots. With the nine + Lin + Ding scorer, summed
over both videos:
- alone: 166/339 = 49.0%, against 162/339;
- combined: 164/324 = 50.6%, against 160/324.

The recommendation is AVG with club = Lin + Ding + n7. Full tables and commands:
`docs/results/retrain_with_review.md`.

## How to run

```
T=data/club_test; O=data/shot_classifier_club; U=data/recordings/club/UHKzdJqgFwk_30.mp4; S1=data/recordings/stand_in/set1_30.mp4
.venv/bin/python -m vision.club_classifier \
  --train data/recordings/club/q3pfhnvKPig.mp4 data/labels/club_q3pfhnvKPig.csv data/flight/club_q3pfhnvKPig/scout.duckdb 29710 \
  --train data/recordings/club/E6SRRmvJ6dI.mp4 data/labels/club_E6SRRmvJ6dI.csv data/flight/club_E6SRRmvJ6dI/scout.duckdb 31799 \
  --predict $U $T/UHKzdJqgFwk_30/lin_ding_alone/scout.duckdb 18149 $O/UHKzdJqgFwk_30/alone \
  --predict $U $T/UHKzdJqgFwk_30/lin_ding_combined/scout.duckdb 18149 $O/UHKzdJqgFwk_30/combined \
  --predict $S1 $T/stand_in_set1_30/lin_ding_alone/scout.duckdb 13117 $O/stand_in_set1_30/alone \
  --predict $S1 $T/stand_in_set1_30/lin_ding_combined/scout.duckdb 13117 $O/stand_in_set1_30/combined   # about 10 s CPU
```

The club training databases are the ones from `docs/results/shuttle_features.md`. Each
video is scored with `vision.label_scores --fps 30` against its 30 fps labels. Each mode
gets four `--run`s on the same hits file: the original database (pro), then
`club.duckdb`, `pro_plus_club.duckdb` and `average.duckdb`. The reports are
`data/shot_classifier_club/{UHKzdJqgFwk_30,stand_in_set1_30}/report.md`. The ±0.25 s section adds
`--breakdown-tolerance 0.25` to the same runs (`report_025.md`).
