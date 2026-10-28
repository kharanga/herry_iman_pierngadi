# Third-party notices

This project includes code derived from, and uses data and models from, the
projects below. Their licenses apply to those parts.

## TrackNetV3 (MIT)

`vision/tracknet.py` re-implements the TrackNet and InpaintNet network
definitions from https://github.com/qaz812345/TrackNetV3 (`model.py`) so the
published checkpoints can be loaded. Layer (attribute) names follow the original
because they must match the checkpoint keys; class names were kept for clarity. The checkpoints themselves are
downloaded by `vision/fetch_tracknetv3.sh` and are not part of this repository.

```
MIT License

Copyright (c) 2024 qaz812345

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software, pretrained model checkpoints, and associated documentation
files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ShuttleSet (MIT)

Shot labels and court calibrations come from ShuttleSet (Wang et al., KDD 2023),
part of https://github.com/wywyWang/CoachAI-Projects, Copyright (c) 2022
Wei-Yao Wang, MIT License. The dataset is downloaded by
`ingest/fetch_shuttleset.sh` and is not redistributed here.

## Ultralytics YOLO (AGPL-3.0)

Player detection and tracking use the `ultralytics` package and its pretrained
YOLO weights (https://github.com/ultralytics/ultralytics), licensed under
AGPL-3.0. They are installed as a dependency and downloaded at runtime; no
Ultralytics code or weights are included in this repository.
