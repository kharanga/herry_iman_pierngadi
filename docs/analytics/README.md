# docs/analytics

Module documentation for the analytics track, written by **analytics-builder**.
Covers code in `ingest/` (ShuttleSet), `analytics/`, `shots/` and `report/`.

One markdown file per module, for example:
- `load_shuttleset.md`: `ingest/load_shuttleset.py` (ANA-002)
- `fetch_shuttleset.md`: `ingest/fetch_shuttleset.sh` (ANA-002)
- `schema.md`: the `shots` table, label mapping, coordinate conversion (ANA-003)
- `classifier.md`: `shots/train.py`, `shots/predict.py` (ANA-004)
- `tendencies.md`: `analytics/tendencies.py` (ANA-005)
- `report.md`: the `report/` app and its command (ANA-006)

For each public function, class or command: what it does, inputs, output, errors,
and a doctest `>>>` example. Examples run with:

    .venv/bin/python -m pytest docs/analytics --doctest-glob="*.md"

Measured results (accuracy, counts) go in `docs/results/`, not here.
