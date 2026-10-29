# Habit counts: results

## How to run
```
.venv/bin/python -m analytics.tendencies <match_id> <player> [sets]
```
How habits are counted: `docs/analytics/tendencies.md`. Each player's habits are counted only from their own
shots; there is no comparison with other players, no score, no ranking and no minimum sample size.

## Runs on every match
All 88 ShuttleSet players (44 matches x 2) and the 2 players of my tracked match run without error.
Loading the shots and building the observations takes 0.2 s; counting all 90 players then takes 0.7 s.

| | Entries per player |
|---|---|
| ShuttleSet, whole match | 51 to 89 (median 78) |
| `match_tracked`, `me` | 40 |
| `match_tracked`, `opp` | 39 |

- One entry per (situation, choice), 6,906 in total over 1,780 situations.
- In every situation the choices' `k` add up to its `n`: checked for all 1,780 situations (1,780 hold).
- Nothing is filtered: 1,255 entries are a single shot ("1 of n").
- Entries per family: zone_mix 2,924, response 2,342, serve_placement 538, winning_shot 351, error_shot 482, error_reason 269.
- Label source: ShuttleSet 6,467 `labels` and 360 `mixed`
  (a shot typed `unknown` got the classifier's group); my tracked match 79 `predictions`
  and 0 `mixed`, since nobody labels my shots.

Example, Ginting (B) in Axelsen vs Ginting (Thailand Open 2021 SF), after a net shot by Axelsen:

- After a net shot: net shots 80 of 148 times.
- After a net shot: drives 38 of 148 times.
- After a net shot: lifts 30 of 148 times.

The choices add up: 80 + 38 + 30 = 148.

## Limits
- Counts describe what happened in this match; small `n` means little, and the page shows `n` so the reader can judge.
- Player labels on one-shot `unknown` rallies are unreliable in ShuttleSet, so those rallies (and the 2 known A/B-swapped
  rallies) are left out of per-player counts.
- On my footage every shot type is a classifier prediction (held-out accuracy on ShuttleSet in
  `docs/results/classifier.md`); positions and timing come from tracking.
- A set 1 -> set 2 check of how well habits carry over was removed with the comparison to pros; it can come back as its own task.
