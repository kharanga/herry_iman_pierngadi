# docs/vision

Module documentation for the video track, written by **vision-builder**.
Covers code in `vision/` and `ingest/record.*`.

One markdown file per module, for example:
- `alignment.md`: label/video alignment check
- `record.md`: Pi recording script
- `court.md`: court calibration and homography
- `players.md`: player detection and tracking
- `shuttle.md`: TrackNetV3 shuttle tracking
- `hits.md`: hit detection and rally segmentation
- `features.md`: hits to `shots` rows

For each public function, class or command: what it does, inputs, output, errors,
and a doctest `>>>` example using small synthetic arrays, not video files.
Examples run with:

    .venv/bin/python -m pytest docs/vision --doctest-glob="*.md"

Measured results (detection rates, frame errors) go in `docs/results/`, not here.

## Examples that need downloaded data

Some doc examples need files that are not in the repo (a video under `data/`, the ShuttleSet download, model weights).
A markdown file declares them with one line, anywhere in the file:

    Examples need: data/match.mp4 (the Axelsen vs Ginting video, see docs/results/court.md)

Paths are relative to the project root, comma-separated; the part in parentheses is optional and says how
to get the input. The root `conftest.py` reads that line when pytest collects the file: if any listed path is
missing, the whole file's doctests are skipped with the reason `needs <path> (<how to get it>)`. With the
inputs present nothing is skipped. A fresh clone with no `data/` therefore runs the suite with 0 failures,
and `pytest -rs` lists what was skipped and why. Examples that build everything they need in a temp
folder (synthetic rows, in-memory databases) declare nothing.
