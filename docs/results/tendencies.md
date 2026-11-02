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
Loading the shots and building the observations takes 0.1 s; counting all 92 players then takes 0.9 s.

| | Entries per player |
|---|---|
| ShuttleSet, whole match | 117 to 325 (median 207.5) |
| `match_tracked`, `me` | 66 |
| `match_tracked`, `opp` | 63 |
| `stand_in_set1`, `me` | 110 |
| `stand_in_set1`, `opp` | 123 |

- One entry per (situation, choice), 19,048 in total over 6,958 situations.
- In every situation the choices' `k` add up to its `n`: checked for all 6,958 situations (6,958 hold).
- Nothing is filtered: 9,484 entries are a single shot ("1 of n").
- Entries per family: zone_mix 2,220, response 15,362, serve_placement 364, winning_shot 351, error_shot 482, error_reason 269.
- Label source: ShuttleSet 18,360 `labels`, 244 `mixed` and
  82 `predictions` (18,686 in all). A situation is `mixed` when some of its shots were typed `unknown`
  in ShuttleSet and got the classifier's group, and `predictions` when all of them were: the 82 `predictions`
  entries (81 `response`, 1 `serve_placement`) come from situations so small that every shot in them was `unknown`, e.g.
  `response:serve|rear_forehand|mid_backhand:drive cross, 1 of 1`; my own matches 362 `predictions`
  and 0 `mixed` (129 for `match_tracked`, 233 for `stand_in_set1`), since nobody labels my shots.

Replies count the other player's previous shot by its type, where it was hit from and where it landed, and the
reply by its type and direction: 31,945 counted replies, 19,872 straight,
11,032 cross and 1041 with no direction (the reply landed on the replier's own half, `not_over_net`, so it has no direction; `is_cross` is NULL there).

Example, Ginting (B) in Axelsen vs Ginting (Thailand Open 2021 SF), in his most frequent reply situation:

- After a lift from the other player's front backhand to the rear backhand: smashes straight 7 of 28 times.
- After a lift from the other player's front backhand to the rear backhand: smashes cross 6 of 28 times.
- After a lift from the other player's front backhand to the rear backhand: clears straight 5 of 28 times.
- After a lift from the other player's front backhand to the rear backhand: drops straight 4 of 28 times.
- After a lift from the other player's front backhand to the rear backhand: drops cross 3 of 28 times.
- After a lift from the other player's front backhand to the rear backhand: clears cross 2 of 28 times.
- After a lift from the other player's front backhand to the rear backhand: smashes 1 of 28 times.

The choices add up: 7 + 6 + 5 + 4 + 3 + 2 + 1 = 28. The last smash has no direction: the reply came back on
Ginting's own half (`not_over_net`), so it has no straight or cross. The situation grew from 23 to 28 with the new
band edges (`docs/results/schema.md`).

## Limits
- Counts describe what happened in this match; small `n` means little, and the page shows `n` so the reader can judge.
- Player labels on one-shot `unknown` rallies are unreliable in ShuttleSet, so those rallies (and the 2 known A/B-swapped
  rallies) are left out of per-player counts.
- On my footage every shot type is a classifier prediction (held-out accuracy on ShuttleSet in
  `docs/results/classifier_feet.md`); positions and timing come from tracking.
- A set 1 -> set 2 check of how well habits carry over was removed with the comparison to pros; it can come back as its own task.
