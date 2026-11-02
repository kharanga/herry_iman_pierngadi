# Rally start and end: in-play stretches with a stance serve

Can a refit in-play model end rallies better, and a stance rule start them better, than
the combined detector? The combined detector takes its rallies and serves from the
default hit model and its hits from nine + Lin + Ding.

- **Rally end:** the in-play model (`docs/vision/in_play.md`), refit on the nine pro
  matches plus the checked rallies of Lin, Ding and n7. The clubs together carry 1/3 of
  the weight. Each club's hit-score features come from the nine-match scorer, which was
  not trained on it. Frames: Lin 13,157 (6,588 in play), Ding 15,669 (5,317), n7 6,903
  (503; only 9 of its 48 rallies have checked bounds). Saved as
  `data/models/in_play_nine_plus_clubs.joblib`.
- **Rally start:** the stance rule (`docs/vision/serve_stance.md`). Before a serve, both
  players stand nearly still in diagonal service courts, and the server moves first. The
  serve is the server's hit nearest the stance end, or the stance end if no hit is found.
- **Pipeline** (`vision.movement --in-play-model ... --stretch-model ... --serve-stance`):
  1. the in-play stretches, fed by nine-match hit scores;
  2. the stance serve nearest each stretch's start;
  3. the nine + Lin + Ding hit chain after it, up to the stretch end (later hits dropped);
  4. shot types by AVG with club = Lin + Ding + n7 (`docs/results/retrain_with_review.md`).

  The default detector is unchanged.

The test is set 1 (28 rallies with checked bounds, 172 hits) and test game 1
(`UHKzdJqgFwk`, 13 checked rallies, 110 hits), both at 30 fps. It was fixed before any
run. Nothing was tuned on either video. Three detectors were run, each with nine + Lin +
Ding hits at 0.55 / 0.6 and the same AVG types:
- **combined:** `--rallies-from data/models/movement_hits.joblib`;
- **in-play alone:** the refit in-play stretches, with the inferred serve at each
  stretch start (within 0.3 s of the first hit, that hit is the serve);
- **stance:** the same stretches, with the stance serve.

## Tuning the stance rule (training games only)

Lin, Ding and n7 (151 labelled serves), with the nine-match scorer's hits. A serve is
found when it falls within ±0.5 s; guesses count inside each video's labelled span.

| Version | Serves found | Guesses |
|---|---|---|
| First try: the server's first hit from the stance start | 40 of 151 | 128 |
| Hit nearest the stance end (−0.3 to +0.5 s), guesses under 3 s apart thinned | **97 of 151** | 164 |

The first try usually grabbed an earlier hit, so its median guess sat 0.4-0.75 s before
the labelled serve. The second version's best of the 18 grid points is still under
0.5 m/s, a stance of 0.5 s and a move over 0.8 m/s. The rest of the grid, at 0.5 m/s
and 0.8 m/s for the move:
- a stance of 1 s finds 80 serves from 117 guesses;
- a stance of 1.5 s finds 53 from 81.

At 0.3 m/s it finds 68 or fewer. At 0.8 m/s still, the most it finds is 85, from 202
guesses.

Of the 54 serves the rule misses:
- 37 have a guess 0.5-2 s away, on either side (timing);
- 17 have no guess within 2 s.

On its own, with its frozen constants and the nine-match hits, the rule finds:
- test game 1: 10 of 13 serves at ±0.5 s (9 at ±0.25 s), from 13 guesses;
- set 1: 13 of 31 at ±0.5 s (13 at ±0.25 s), from 24 guesses.

## Rally bounds (checked rallies only)

| Video | Detector | Frames right | Found | Matched | Merged | Split | Missed | Extra | Start error, median abs (s) | End error, median abs (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Set 1 | combined | 67.6% | 31 | **24 of 28** | 0 | 1 | 4 | 1 | 1.33 | 2.92 |
| Set 1 | in-play alone | 79.5% | 26 | 21 of 28 | 0 | 0 | 7 | 0 | 1.30 | **0.77** |
| Set 1 | stance | **81.1%** | 26 | 21 of 28 | 0 | 0 | 7 | 0 | **0.93** | **0.77** |
| Test game 1 | combined | 70.3% | 13 | 11 of 13 | 0 | 0 | 2 | 2 | 0.70 | 3.63 |
| Test game 1 | in-play alone | **92.4%** | 11 | 11 of 13 | 0 | 0 | 2 | 0 | **0.20** | **0.73** |
| Test game 1 | stance | 91.8% | 11 | 11 of 13 | 0 | 0 | 2 | 0 | 0.47 | **0.73** |

Frames right is judged up to the last labelled frame (7802 on test game 1). Frames inside
set 1's 6 unchecked rallies are left out.

## Serves, and hits after rally ends

| Video | Detector | Serve rows | Serves found ±0.25 s | Fake ±0.25 s | Serves found ±0.50 s | Fake ±0.50 s | Detected hits out of play |
|---|---|---|---|---|---|---|---|
| Set 1 | combined | 31 | **15 of 31** | 16 | **19 of 31** | 12 | 14 |
| Set 1 | in-play alone | 26 | 6 of 31 | 20 | 10 of 31 | 16 | **6** |
| Set 1 | stance | 26 | 9 of 31 | 17 | 11 of 31 | 15 | 9 |
| Test game 1 | combined | 13 | 6 of 13 | 7 | 7 of 13 | 6 | 8 |
| Test game 1 | in-play alone | 11 | **8 of 13** | 3 | **9 of 13** | 2 | **4** |
| Test game 1 | stance | 11 | **8 of 13** | 3 | **9 of 13** | 2 | **4** |

Each detected rally's first row is its serve row. Hits out of play are detected hits
outside every hand-marked rally, up to the last labelled frame.

## Correct shots (the decision metric)

Matched and typed right ÷ (labelled hits + false hits inside labelled rallies), detected
and inferred serves:

| Detector | Test game 1, ±0.50 s | Set 1, ±0.50 s | Both, ±0.50 s | Both, ±0.25 s |
|---|---|---|---|---|
| combined | 63/123 = 51.2% | **101/201 = 50.2%** | **164/324 = 50.6%** | **155/343 = 45.2%** |
| in-play alone | **67/124 = 54.0%** | 80/196 = 40.8% | 147/320 = 45.9% | 136/341 = 39.9% |
| stance | 66/123 = 53.7% | 84/199 = 42.2% | 150/322 = 46.6% | 141/341 = 41.3% |

At ±0.25 s: test game 1 61/129, 65/130, 64/129; set 1 94/214, 71/211, 77/212. The
combined row equals `docs/results/retrain_with_review.md` (old scorer, combined, new AVG).

## What it says

- **The in-play stretches end rallies far better.** End error is 0.7-0.8 s against
  2.9-3.6 s, there are no extra rallies, and fewer hits fall after rally ends.
- **They miss rallies on set 1:** 7 of 28, against 4 for the combined detector. That
  costs 17-21 correct shots on set 1 and outweighs the gain on test game 1 (+3 or +4).
  Summed, the combined detector stays best (164/324 against 150/322 and 147/320).
- **The stance serve is a small gain over the inferred serve at the stretch start.** On
  set 1 it finds 3 more serves at ±0.25 s (1 more at ±0.50 s), cuts the start error from
  1.30 to 0.93 s, and adds 4 correct shots. On test game 1 its serves are the same and
  it has one correct shot fewer. By itself the rule finds 10 of 13 and
  13 of 31 serves at ±0.5 s on the test videos. Inside the pipeline it finds fewer,
  because a serve only counts for a stretch that was found, within 2 s before to 1 s
  after its start.
- **Recommendation:** keep the default and the combined detector. The pipeline is worth
  another look only if the in-play model stops missing rallies, for example with more
  checked club rallies (n7 has 9 of 48). Its end bounds would then help: the combined
  detector's ends run 3 s long.
- This is two videos, 41 checked rallies. A difference of a few shots is noise; the
  7 against 4 missed rallies and the end errors are the clear parts.

## How to run

zsh, from the project root, CPU only.

The in-play model:

```
NINE=data/models/movement_hits_ten_matches.joblib
.venv/bin/python -m vision.in_play data/hit_training/matches.txt --save data/models/in_play_nine_plus_clubs.joblib \
  --club data/recordings/club/q3pfhnvKPig.mp4 data/labels/club_q3pfhnvKPig.csv $NINE \
  --club data/recordings/club/E6SRRmvJ6dI.mp4 data/labels/club_E6SRRmvJ6dI.csv $NINE \
  --club data/recordings/club/n7MqoAVqEW4_30.mp4 data/labels/club_n7MqoAVqEW4_30.csv $NINE
```

Tuning the stance rule (prints the grid; the best point is in the module's constants):

```
.venv/bin/python -m vision.serve_stance \
  --train data/recordings/club/q3pfhnvKPig.mp4 data/labels/club_q3pfhnvKPig.csv $NINE \
  --train data/recordings/club/E6SRRmvJ6dI.mp4 data/labels/club_E6SRRmvJ6dI.csv $NINE \
  --train data/recordings/club/n7MqoAVqEW4_30.mp4 data/labels/club_n7MqoAVqEW4_30.csv $NINE
```

The test (target = video : match id : tracked end : run folder : last labelled frame):

```
O=data/vis042
HITS=data/models/movement_hits_nine_plus_lin_ding_tuned.joblib
IN_PLAY=data/models/in_play_nine_plus_clubs.joblib
N7V=data/recordings/club/n7MqoAVqEW4_30.mp4
TRAIN=(--train data/recordings/club/q3pfhnvKPig.mp4 data/labels/club_q3pfhnvKPig.csv data/flight/club_q3pfhnvKPig/scout.duckdb 29710 --train data/recordings/club/E6SRRmvJ6dI.mp4 data/labels/club_E6SRRmvJ6dI.csv data/flight/club_E6SRRmvJ6dI/scout.duckdb 31799 --train $N7V data/vis041/labels/club_n7MqoAVqEW4_30.csv data/flight/club_n7MqoAVqEW4_30/scout.duckdb 21727)
for target in data/recordings/club/UHKzdJqgFwk_30.mp4:club_UHKzdJqgFwk_30:63443:UHKzdJqgFwk_30:7802 data/recordings/stand_in/set1_30.mp4:stand_in_set1_30:13117:stand_in_set1_30:13117; do
  IFS=: read V M E dir L <<< "$target"
  runs=(); found=()
  for method in combined in_play stance; do
    case $method in
      combined) extra=(--rallies-from data/models/movement_hits.joblib) ;;
      in_play) extra=(--in-play-model $IN_PLAY --stretch-model $NINE) ;;
      stance) extra=(--in-play-model $IN_PLAY --stretch-model $NINE --serve-stance) ;;
    esac
    T=$O/$dir/$method
    .venv/bin/python -m vision.movement $V --start 0 --end $E --model $HITS --hit-score-min 0.55 --rally-end-score-min 0.6 $extra --hits-dir $T/hits
    cp data/scout.duckdb $T/scout.duckdb
    .venv/bin/python -m vision.features $V --match-id $M --range 0 $E 1 --database $T/scout.duckdb --hits-dir $T/hits
    .venv/bin/python -m vision.club_classifier $TRAIN --predict $V $T/scout.duckdb $E $T/classifier
    runs+=(--run $method $T/hits/$M/hits_0_$E.csv $T/classifier/average.duckdb)
    found+=(--found $method $T/hits/$M/hits_0_$E.csv)
  done
  .venv/bin/python -m vision.label_scores --labels data/labels/$M.csv --match-id $M --fps 30 $runs > $O/$dir/report.md
  .venv/bin/python -m vision.rally_bounds $V --rallies data/labels/${M}_rallies.csv --labels data/labels/$M.csv --end $L $found > $O/$dir/bounds.md
done
```
