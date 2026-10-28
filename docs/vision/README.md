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
