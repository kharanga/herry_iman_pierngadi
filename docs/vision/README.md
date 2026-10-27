# docs/vision

Module documentation for the video track, written by **vision-builder**.
Covers code in `vision/` and `ingest/record.*`.

One markdown file per module, for example:
- `alignment.md`: label/video alignment check (VIS-001)
- `record.md`: Pi recording script (VIS-002)
- `court.md`: court calibration and homography (VIS-003)
- `players.md`: player detection and tracking (VIS-004)
- `shuttle.md`: TrackNetV3 shuttle tracking (VIS-005)
- `hits.md`: hit detection and rally segmentation (VIS-006)
- `features.md`: hits to `shots` rows (VIS-007)

For each public function, class or command: what it does, inputs, output, errors,
and a doctest `>>>` example using small synthetic arrays, not video files.
Examples run with:

    .venv/bin/python -m pytest docs/vision --doctest-glob="*.md"

Measured results (detection rates, frame errors) go in `docs/results/`, not here.
