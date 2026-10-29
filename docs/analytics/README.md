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
- `report.md`: `report/build.py`, `report/page.py`, `report/court_svg.py`, `report/serve.py`, `report/preview.py`

For each public function, class or command: what it does, inputs, output, errors,
and a doctest `>>>` example. Examples run with:

    .venv/bin/python -m pytest docs/analytics --doctest-glob="*.md"

Measured results (accuracy, counts) go in `docs/results/`, not here.

## Examples that need downloaded data

Some doc examples need files that are not in the repo (the ShuttleSet download, a video, model weights).
A markdown file declares them with one line, anywhere in the file:

    Examples need: data/CoachAI-Projects/ShuttleSet/set (run ./ingest/fetch_shuttleset.sh)

Paths are relative to the project root, comma-separated; the part in parentheses is optional and says how
to get the input. The root `conftest.py` reads that line when pytest collects the file: if any listed path is
missing, the whole file's doctests are skipped with the reason `needs <path> (<how to get it>)`. With the
inputs present nothing is skipped. A fresh clone with no `data/` therefore runs the suite with 0 failures,
and `pytest -rs` lists what was skipped and why. Examples that build everything they need in a temp
folder (synthetic rows, in-memory databases) declare nothing.
