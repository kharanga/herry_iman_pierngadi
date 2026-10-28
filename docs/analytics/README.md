# docs/analytics

Module documentation for the analytics track, written by **analytics-builder**.
Covers code in `ingest/` (ShuttleSet), `analytics/`, `shots/` and `report/`.

One markdown file per module, for example:
- `load_shuttleset.md`: `ingest/load_shuttleset.py`
- `fetch_shuttleset.md`: `ingest/fetch_shuttleset.sh`
- `label_translations.md`: `ingest/label_translations.py`
- `build_shots.md`: `analytics/build_shots.py` and `analytics/schema.sql`, the `shots` table
- `court.md`: `analytics/court.py`, court meters, normalization, zones, cross-court
- `shot_types.md`: `analytics/shot_types.py`, shot type -> group
- `spot_check.md`: `analytics/spot_check.py`, the court diagram of random shots
- `classifier.md`: `shots/features.py`, `shots/train.py`, `shots/predict.py`
- `tendencies.md`: `analytics/tendencies.py`
- `report.md`: the `report/` app and its command

For each public function, class or command: what it does, inputs, output, errors,
and a doctest `>>>` example. Examples run with:

    .venv/bin/python -m pytest docs/analytics --doctest-glob="*.md"

Measured results (accuracy, counts) go in `docs/results/`, not here.
