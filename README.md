# badminton-scout

Singles badminton analysis from a single fixed camera. A Raspberry Pi records a match from behind one
baseline; a Mac tracks both players, finds every hit from their movement, classifies each shot from the
players' feet positions and timing, and builds a phone-friendly report of each player's habits ("short
serves go to the receiver's front backhand, 10 of 19 times").

No shuttle tracking is needed: hits come from the players' movement, the hit point is the hitter's feet,
and a shot's "landing" is where the opponent plays the reply.

## Pipeline

1. **Court calibration**: click known court points on one frame (`vision.court --pick-points`).
2. **Player tracking**: YOLO11s person detection (Ultralytics, pretrained) + ByteTrack, mapped to court
   metres; picks the two players over courtside people and passers-by (`vision.players`).
3. **Hits and rallies**: a gradient-boosting hit scorer over both players' movement (`vision.movement`).
4. **Shot rows**: hitter and receiver positions, zones, straight or cross (`vision.features`).
5. **Shot types**: eight labels (short/long serve, net, drive, clear, drop, smash, lift) from a classifier
   trained on ShuttleSet, optionally averaged with one trained on hand-labelled club games
   (`shots.train`, `vision.club_classifier`).
6. **Habits and report**: plain counts per player, served as a web page for a phone (`analytics.tendencies`,
   `report.serve`).

`scout.run` runs the whole chain for one match, one video per set.

## Setup

Python 3.13 on macOS (Apple Silicon; player detection uses the GPU through PyTorch MPS, CPU otherwise).

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./ingest/fetch_shuttleset.sh                  # ShuttleSet into data/CoachAI-Projects/
.venv/bin/python -m ingest.load_shuttleset     # data/scout.duckdb
.venv/bin/python -m analytics.build_shots      # the shots table
.venv/bin/python -m shots.train                # shot-type classifier -> data/models/
```

Videos, databases and trained models live under `data/` and are never committed. `scout.run` uses the hit
model `data/models/movement_hits.joblib`, trained from ShuttleSet's broadcast videos (downloaded, calibrated
and tracked first) with `vision.movement <video> --train --match <folder> --range START END SET ...`
(`docs/vision/movement.md`). Larger hit models from several matches and club games are trained with
`vision.hit_training` (`docs/vision/hit_training.md`).

## Running a match

```
.venv/bin/python -m scout.run data/recordings/<match>/set1.mp4 [set2.mp4 ...] --match-id <match> --pick-points
```

Each step and its options are documented in `docs/analytics/` and `docs/vision/` (one page per module,
with runnable examples).

## Labelling club footage

`vision.label_ui` is a browser tool for hand-labelling hits, shot types and rally starts/ends
(`--blank`, in 60 s segments), or for reviewing the model's own labels (`--review`). See
`docs/vision/label_ui.md`.

## Tests

```
.venv/bin/python -m pytest tests docs/analytics docs/vision --doctest-glob="*.md"
```

Tests that need ShuttleSet or local videos skip when those files are absent.

## Credits

- **ShuttleSet** (Wang et al., KDD 2023; MIT licence): the shot-labelled professional singles dataset used
  for training and evaluation. https://github.com/wywyWang/CoachAI-Projects
- **Ultralytics YOLO11** (AGPL-3.0) for person detection, with ByteTrack.
- **TrackNetV3** for the optional shuttle-tracking experiments.
