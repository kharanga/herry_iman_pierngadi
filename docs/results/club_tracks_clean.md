# Cleaning the club tracks: who counts as a player

The club videos are filmed from beside the court, with the umpire, line judges, other
players and passers-by in view. The first player choice (fixed side per track,
continuity first) let some of them stand in for a player. This page counts the errors
before and after the new player choice in `vision/players.py` (`docs/vision/players.md`,
"Choosing the players").

The counts come from `python -m vision.track_check` (`docs/vision/track_check.md`).
The sample sheets are in `data/checks/track_check/<video id>_<error>_<before|after>.jpg`
(gitignored).

## The new player choice

- **No swaps:** every frame each track's half is checked again, with a 0.5 m hysteresis
  at the net, so `me` / `opp` can't stay swapped after an end change.
- **Still stand-ins:** a person moving inside the court lines beats one standing still in
  the margin. A held person who has stood still outside the lines for 2 s is dropped
  when someone moving is inside.
- **Passers-by:** a box at the image's side edge more than 1.6 times the player's usual
  height is someone walking in front of the camera. It is never taken as the player.
- **`doubtful` flag:** set on a still stand-in, on a hold outside the lines while
  someone is inside, and on frames where a passer-by was set aside. Training drops
  these rows.
- **Raw detections are saved** (`detections_<start>_<end>.csv`), so later rule changes
  re-run with `python -m vision.players <video> --start 0 --end N --from-detections`,
  without YOLO.

## Lin v Yeung (`q3pfhnvKPig`, 24 fps, 29,711 frames; 67 labelled rallies, 421 labelled hits)

Re-tracked in full: 36 min on the GPU (MPS, batch 4), alone, swap guard on. No
"NMS time limit" warning. The raw detections were saved, so after the review fix to the
player choice (track memory across gaps, the still limit in video frames) the players
were chosen again from them with `--from-detections`, without YOLO. "After" is that
rebuild, which is Lin's track file now.

Kept for comparison:

- the original tracks in `data/players_before_clean/club_q3pfhnvKPig/`;
- the first re-track's choice, from before the fix, in
  `data/players_before_fix/club_q3pfhnvKPig/`.

The fix changed 934 of 59,391 player rows (1.57%), 378 of 27,317 inside the labelled
rallies (1.38%). The hit-scorer results in `docs/results/hits_club.md` use the rebuilt
tracks.

| Check | Before | After |
|---|---|---|
| Still stand-ins (over 0.3 m outside the lines, slower than 0.3 m/s, for 2 s or more) | 40 runs, 5,209 player-frames (3.6 min), 377 inside rallies | 11 runs, 2,234 player-frames (1.6 min), **30 inside rallies** (15 runs / 42 before the fix) |
| Passers-by | 0 | 0 |
| me/opp swapped | 0 frames | 3 frames (14457-14461, at the net) |
| Gaps of 0.5 s or more inside rallies | 0 | 0 |
| Labelled hits: hitter on its half | 419 of 421 | **421 of 421** |
| Labelled hits: hitter moving within 1 s | 406 of 421 | 414 of 421 |
| Labelled hits: both | 404 of 421 = 96.0% | **414 of 421 = 98.3%** |

**What the sheets show:**

- **Before:** `opp` often sat on the seated line judge at the far right corner while the
  real far player was on court (frame 15562). `me` sat on people beside the near court
  between points.
- **After:** the line judge is no longer taken while the far player is in view. The
  remaining stand-ins are game breaks and the time between points, when the players
  really are off court by the umpire's chair. Then the tracker holds the nearest still
  person, sometimes the player herself at the sideline.

**Doubtful rows:**

- 4,323 of 59,391 player rows (7.3%) are flagged doubtful.
- That covers 2,025 of the 2,234 remaining stand-in player-frames.
- Inside the labelled rallies only 488 of 27,317 player rows (1.8%) are doubtful.

## Stand-in set 1 (`stand_in_set1`, 60 fps)

Not re-tracked. The diagnosis of its current tracks, against the user's relabel (172 hits,
34 rallies; diagnosed on 2026-11-01 from a read-only copy in
`data/club_labels/set1_labels_relabel/`):

- 0 still stand-ins, 0 passers-by and 0 swaps;
- two gaps of 0.5 s or more inside rallies (62 player-frames);
- 171 of 172 labelled hits with the hitter on its half and moving.

Against the 153 labels before the relabel, the same tracks gave one 31-frame gap and 152
of 153.

The new rules only change a frame when one of those errors is present: a still person
outside the lines, an oversized box at the side edge, or a track past the net. So we
expect re-tracking to give the same rows, apart from the new `doubtful` column. That is
reasoning from the rules, not a measurement: set 1 has no saved raw detections to re-run
the choice on. Set 1 keeps
its tracks, and the hit-scorer evaluation on set 1 (`docs/results/hits_club.md`) uses
them.

## Chong v Ding (`E6SRRmvJ6dI`, 31,800 frames)

Re-tracked in full with the fixed player choice: 33 min on the GPU (MPS, batch 4), alone,
swap guard on, no "NMS time limit" warning. The original tracks are kept in
`data/players_before_clean/club_E6SRRmvJ6dI/`. Ding has no labels yet, so both columns
are scored on the same 87 rallies the detector found on the original tracks
(`data/hits/club_E6SRRmvJ6dI/rallies_0_31799.csv`).

| Check | Before | After |
|---|---|---|
| Still stand-ins | 0 | 0 |
| Passers-by (club members walking right in front of the camera) | 60 runs, 449 player-frames, 342 inside rallies | 46 runs, 298 player-frames, 219 inside rallies |
| me/opp swapped | 1,223 frames (3.8%), mostly one stretch around frames 18,600-19,800 after an end change | **0 frames** |
| Gaps of 0.5 s or more inside rallies | 47 runs, 1,055 player-frames (of 38,062), mostly a player hidden by someone passing in front of the camera | 39 runs, 785 player-frames (of 38,062) |

- The halves are re-checked every frame, which ends the swapped stretch.
- Passer-by frames drop by a third. The ones left are flagged doubtful, not recovered.
- Gap frames drop by a quarter.
- The hit scorer and the rallies have not been rerun on the new tracks; that waits for
  the user's labels.

## How to reproduce

```
.venv/bin/python -m vision.track_check data/recordings/club/q3pfhnvKPig.mp4 --end 29710 --labels data/labels/club_q3pfhnvKPig.csv --players-dir data/players_before_clean --sheets --tag before
.venv/bin/python -m vision.players data/recordings/club/q3pfhnvKPig.mp4 --stretches data/club_games/q3pfhnvKPig_stretches.csv --merge --batch-frames 4
.venv/bin/python -m vision.players data/recordings/club/q3pfhnvKPig.mp4 --start 0 --end 29710 --from-detections   # the choice again, after the fix
.venv/bin/python -m vision.track_check data/recordings/club/q3pfhnvKPig.mp4 --end 29710 --labels data/labels/club_q3pfhnvKPig.csv --sheets --tag after
.venv/bin/python -m vision.track_check data/recordings/stand_in/set1.mp4 --end 26234 --labels data/club_labels/set1_labels_relabel/stand_in_set1.csv
.venv/bin/python -m vision.track_check data/recordings/club/E6SRRmvJ6dI.mp4 --end 31799 --rallies data/hits/club_E6SRRmvJ6dI/rallies_0_31799.csv --players-dir data/players_before_clean --sheets --tag before
.venv/bin/python -m vision.players data/recordings/club/E6SRRmvJ6dI.mp4 --stretches data/club_games/E6SRRmvJ6dI_stretches.csv --merge --batch-frames 4
.venv/bin/python -m vision.track_check data/recordings/club/E6SRRmvJ6dI.mp4 --end 31799 --rallies data/hits/club_E6SRRmvJ6dI/rallies_0_31799.csv --sheets --tag after
```
