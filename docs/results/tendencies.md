# Habit counts: results

## How to run
```
.venv/bin/python -m analytics.tendencies <match_id> <player> [sets]
```
How habits are counted: `docs/analytics/tendencies.md`. Each player's habits are counted only from their own
shots; there is no comparison with other players, no score, no ranking and no minimum sample size.

## Runs on every match
All 88 ShuttleSet players (44 matches x 2) and the 4 players of my two own matches (the tracked broadcast rallies,
`match_tracked`, and the stand-in club set, `stand_in_set1`) run without error.
Loading the shots and building the observations takes 0.2 s; counting all 92 players then takes 1.1 s.

| | Entries per player |
|---|---|
| ShuttleSet, whole match | 127 to 326 (median 228.5) |
| `match_tracked`, `me` | 66 |
| `match_tracked`, `opp` | 67 |
| `stand_in_set1`, `me` | 107 |
| `stand_in_set1`, `opp` | 117 |

- One entry per (situation, choice), 20,716 in total over 7,470 situations.
- In every situation the choices' `k` add up to its `n`: checked for all 7,470 situations (7,470 hold).
- Nothing is filtered: 10,702 entries are a single shot ("1 of n").
- Entries per family: zone_mix 2,224, response 16,929, serve_placement 461, winning_shot 351, error_shot 482, error_reason 269.
- Label source: ShuttleSet 19,975 `labels`, 306 `mixed` and
  78 `predictions` (20,359 in all). A situation is `mixed` when some of its shots were typed `unknown`
  in ShuttleSet and got the classifier's group, and `predictions` when all of them were: the 78 `predictions`
  entries (78 `response`) come from situations so small that every shot in them was `unknown`, e.g.
  `response:serve|rear_forehand|mid_backhand:drive cross, 1 of 1`; my own matches 357 `predictions`
  and 0 `mixed` (133 for `match_tracked`, 224 for `stand_in_set1`), since nobody labels my shots.

Replies count the other player's previous shot by its type, where it was hit from and where it landed, and the
reply by its type and direction: 31,945 counted replies, 19,872 straight,
11,032 cross and 1041 with no direction (the reply landed on the replier's own half, `not_over_net`, so it has no direction; `is_cross` is NULL there).

Example, Ginting (B) in Axelsen vs Ginting (Thailand Open 2021 SF), in his most frequent reply situation:

- After a lift from the other player's front backhand to the rear backhand: smashes cross 6 of 23 times.
- After a lift from the other player's front backhand to the rear backhand: smashes straight 6 of 23 times.
- After a lift from the other player's front backhand to the rear backhand: drops straight 4 of 23 times.
- After a lift from the other player's front backhand to the rear backhand: clears straight 3 of 23 times.
- After a lift from the other player's front backhand to the rear backhand: drops cross 3 of 23 times.
- After a lift from the other player's front backhand to the rear backhand: clears cross 1 of 23 times.

The choices add up: 6 + 6 + 4 + 3 + 3 + 1 = 23.

## Limits
- Counts describe what happened in this match; small `n` means little, and the page shows `n` so the reader can judge.
- Player labels on one-shot `unknown` rallies are unreliable in ShuttleSet, so those rallies (and the 2 known A/B-swapped
  rallies) are left out of per-player counts.
- On my footage every shot type is a classifier prediction (held-out accuracy on ShuttleSet in
  `docs/results/classifier_feet.md`); positions and timing come from tracking.
- A set 1 -> set 2 check of how well habits carry over was removed with the comparison to pros; it can come back as its own task.
