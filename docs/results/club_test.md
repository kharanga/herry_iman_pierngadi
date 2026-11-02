# The club test: one held-out game

The final test of the hit detector and the shot classifier, on a club game that played no
part in training, tuning or any earlier choice (`docs/results/club_games.md`, "Two test
games"). The rule for picking the winner was fixed before any test label was read.

**This is 110 labelled hits from one game.** One correct shot more or less moves the
decision metric by about 0.8 points, so differences of a point or two are noise. The
second test game (`xBClOgrP1Hs`) is not labelled yet and will be a second run.

## Test data

- **Video:** `UHKzdJqgFwk`, 60 fps. The low camera is behind the near court at a corner,
  and the view is wide-angle.
- **Labels:** the user's, in blank mode: 110 hits in 13 rallies, all with checked bounds.
  They cover frames 161-15,603, the first 4.3 minutes. The hits are 27 lifts, 21 net
  shots, 16 smashes, 13 serves, 11 drops, 10 clears, 6 drives and 6 unknown. Labels
  follow the user's rule that a drive directly after a clear is a smash.
- **Tracks:** frames 0-36,299 (the first 10 minutes), the cleaned player choice.
  Every candidate's hits fall inside that range.

## Candidates and rule

Every scorer is as already saved, with no refit and no tuning on this game:

| Candidate | Hit scorer (thresholds) | Rallies and serves |
|---|---|---|
| Default | five-range model (0.5 / 0.7) | its own gap rule |
| Nine matches, alone | nine-match model (0.5 / 0.7) | its own gap rule |
| Nine matches, combined | nine-match model (0.5 / 0.7) | the default's (`--rallies-from`) |
| Nine + Lin, alone | nine + Lin 1/3 tuned (0.55 / 0.5) | its own gap rule |
| Nine + Lin, combined | nine + Lin 1/3 tuned (0.55 / 0.5) | the default's |
| Nine + Lin + Ding, alone | nine + Lin + Ding tuned (0.55 / 0.6) | its own gap rule |
| Nine + Lin + Ding, combined | nine + Lin + Ding tuned (0.55 / 0.6) | the default's |

**Decision rule, fixed before the labels were read.** The winner has the most correct
shots at ±0.50 s. That means the labelled hits matched and given the right shot type
(group level, with the serve rows), divided by all labelled hits plus the candidate's
unmatched detections inside the labelled rallies. A tie, within 1 point of the best,
goes to higher precision.

## Results

At ±0.50 s, with ±0.25 s in brackets. "Detected" rows leave out the inferred serve rows;
shot types and correct shots count them.

| Candidate | **Correct shots** | Hits found (detected) | Precision in labelled rallies (detected) | Precision, with serve rows | Hitter right | Shot type, with serves | 8-label detail | Shot type, labelled serves left out | Serves found / typed `serve` (of 13) | Rallies matched of 13; merged / split / missed | Start / end error, median abs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Default | 31.2% (25.5%) | 75.5% (62.7%) | 85.6% (73.4%) | 86.0% | 91.6% | 45.3% (48.6%) | 45.3% | 41.8% (43.9%) | 7 / 6 | 10; 0 / 0 / 3 | 0.56 s / 5.78 s |
| Nine matches, alone | 32.6% (29.7%) | **95.5% (85.5%)** | 81.4% (77.0%) | 80.9% | 95.2% | 44.9% (46.6%) | 44.9% | **50.0% (50.6%)** | 12 / 1 | 9; 4 / 0 / 0 | 3.32 s / 11.15 s |
| **Nine matches, combined** | **38.1%** (33.1%) | 80.9% (73.6%) | 86.4% (80.2%) | 85.8% | 94.4% | **53.3% (53.7%)** | **53.3%** | 49.4% (50.0%) | 7 / 7 | 10; 0 / 0 / 3 | 0.56 s / 5.78 s |
| **Nine + Lin, alone** | **37.7%** (**36.6%**) | 89.1% (84.5%) | **92.5% (88.6%)** | 89.8% | **95.9%** | 48.9% (50.6%) | 48.9% | 48.1% (49.4%) | 13 / 7 | **13; 0 / 1 / 0** | **0.37 s / 1.00 s** |
| Nine + Lin, combined | 37.0% (34.1%) | 79.1% (75.5%) | 92.6% (89.2%) | 91.3% | 95.4% | 49.4% (50.0%) | 49.4% | 45.1% (46.2%) | 7 / 7 | 10; 0 / 0 / 3 | 0.56 s / 5.78 s |
| Nine + Lin + Ding, alone | 36.6% (33.1%) | 88.2% (81.8%) | 91.5% (85.7%) | 88.8% | 94.8% | 46.9% (47.2%) | 46.9% | 45.9% (45.6%) | 11 / 6 | 12; 0 / 0 / 1 | 0.38 s / 2.11 s |
| **Nine + Lin + Ding, combined** | **37.8%** (32.5%) | 79.1% (73.6%) | 91.6% (86.2%) | **91.4%** | 94.3% | 50.6% (50.6%) | 50.6% | 46.3% (46.7%) | 7 / 7 | 10; 0 / 0 / 3 | 0.56 s / 5.78 s |

Correct shots as counts at ±0.50 s:

| Default | Nine matches, alone | Nine matches, combined | Nine + Lin, alone | Nine + Lin, combined | Nine + Lin + Ding, alone | Nine + Lin + Ding, combined |
|---|---|---|---|---|---|---|
| 39 / 125 | 44 / 135 | 48 / 126 | 46 / 122 | 44 / 119 | 45 / 123 | 45 / 119 |

The rally counts are against the 13 labelled rallies. The candidates also find 22 to 31
rallies after frame 15,603, where nothing is labelled, so the "extra" rallies in the
report are not errors and are left out here.

## Applying the rule

- **The top three are a tie.** They are nine matches combined (38.1%), nine + Lin +
  Ding combined (37.8%) and nine + Lin alone (37.7%), all within 1 point of the best.
  The gap is less than one correct shot.
- **The tie goes to precision, but the rule doesn't say which:**
  - Counted on detected hits, as in every hit table so far, nine + Lin alone is the
    most precise (92.5%, against 91.6% and 86.4%).
  - Counted with the serve rows, as the correct-shots metric is, nine + Lin + Ding
    combined is (91.4%, against 89.8% and 85.8%).

  Either reading leaves out nine matches combined. **The choice between the two is the
  user's.** It is not made here.

## What it says

- **Shot types are much worse on this game than on set 1.** About half are right
  (45-53% with serves at ±0.50 s), against about 75% for the same candidates on set 1.
  Finding hits is not the problem: the best candidates find 89-96%, with the hitter
  right on 95%. The classifier is, on this low, wide-angle view.
- **Nine + Lin alone has the best rally bounds by far:**
  - all 13 rallies matched, with one split;
  - starts and ends within 0.37 s and 1.00 s (median);
  - all 13 serves found, 7 typed `serve`.

  The combined candidates take the default's rallies, which miss 3 of the 13 and end
  5.8 s late (median).
- **Serves:** the combined detector types every serve it finds (7 of 7) but finds only
  7. Nine matches alone finds 12 and types 1.
- **Set 1 ranks them differently** (the reference below), so neither game alone settles
  the winner. The second test game will help.

## Set 1 reference

The same seven candidates on stand-in set 1, against the user's live labels (with the
drive-after-clear rule). Set 1 helped choose the earlier models' settings, so it is a
reference, not a test.

| Candidate | Correct shots ±0.50 s | Hits found (with serve rows) | Precision, with serve rows | Shot type, with serves |
|---|---|---|---|---|
| Default | 104 / 234 = 44.4% | 86.6% | 70.6% | 74.8% |
| Nine matches, alone | 104 / 262 = 39.7% | 96.5% | 64.8% | 65.8% |
| Nine matches, combined | 109 / 232 = 47.0% | 91.3% | 72.4% | 74.1% |
| Nine + Lin, alone | 98 / 237 = 41.4% | 91.3% | 70.7% | 68.1% |
| Nine + Lin, combined | **106 / 210 = 50.5%** | 89.0% | **80.1%** | **76.3%** |
| Nine + Lin + Ding, alone | 98 / 230 = 42.6% | 89.0% | 72.5% | 69.5% |
| Nine + Lin + Ding, combined | 103 / 213 = 48.4% | 87.2% | 78.5% | 75.7% |

## How to run

```
V=data/recordings/club/UHKzdJqgFwk.mp4; T=data/club_test/UHKzdJqgFwk; E=126886
for run in A_default:movement_hits:0.5:0.7:alone nine_alone:movement_hits_ten_matches:0.5:0.7:alone nine_combined:movement_hits_ten_matches:0.5:0.7:combined \
    lin_alone:movement_hits_nine_plus_lin_w0.33_tuned:0.55:0.5:alone lin_combined:movement_hits_nine_plus_lin_w0.33_tuned:0.55:0.5:combined \
    lin_ding_alone:movement_hits_nine_plus_lin_ding_tuned:0.55:0.6:alone lin_ding_combined:movement_hits_nine_plus_lin_ding_tuned:0.55:0.6:combined; do
  IFS=: read name model h r mode <<< "$run"
  extra=(); [ $mode = combined ] && extra=(--rallies-from data/models/movement_hits.joblib)
  .venv/bin/python -m vision.movement $V --start 0 --end $E --model data/models/$model.joblib --hit-score-min $h --rally-end-score-min $r $extra --hits-dir $T/$name/hits
  cp data/scout.duckdb $T/$name/scout.duckdb
  .venv/bin/python -m vision.features $V --match-id club_UHKzdJqgFwk --range 0 $E 1 --database $T/$name/scout.duckdb --hits-dir $T/$name/hits
  .venv/bin/python -m shots.predict $T/$name/scout.duckdb
done
.venv/bin/python -m vision.label_scores --labels data/labels/club_UHKzdJqgFwk.csv --match-id club_UHKzdJqgFwk --fps 60 \
  --run A_default $T/A_default/hits/club_UHKzdJqgFwk/hits_0_$E.csv $T/A_default/scout.duckdb   # and one --run per candidate, in the order above
```

The tracks file keeps the whole-video name (`positions_0_126886.csv`) but holds frames
0-36,299 only, so `--end 126886`. The report is `data/club_test/UHKzdJqgFwk/report.md`.

The set-1 reference is `vision.label_scores` on `data/labels/stand_in_set1.csv` with the
set-1 runs of `docs/results/hits_club.md`: `five_ranges`, `nine_matches`,
`combined_nine_matches`, `nine_plus_lin_tuned`, `combined_nine_plus_lin_tuned`,
`lin_ding_tuned` and `combined_lin_ding_tuned`, under the names above. Its report is
`data/club_test/set1_reference.md`.
