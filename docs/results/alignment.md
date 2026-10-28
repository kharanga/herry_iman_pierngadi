# VIS-001: Alignment check — results

## How to run
```
M=Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals
for set_no in 1 2 3; do .venv/bin/python -m vision.alignment $M $set_no data/match.mp4; done
```

## Run on Axelsen vs Ginting (Thailand Open 2021 SF), offset 0
Video: `data/match.mp4`, 30.00 fps, 1280 wide (scale 1.000). 4 shots per set, 12 images, 0 frames unreadable.

| Set | Shots checked (rally/shot, frame) | Output folder |
|---|---|---|
| 1 | 1/2 @ 12359, 17/8 @ 25747, 29/8 @ 40576, 39/22 @ 52638 | `data/checks/<match>/set1/` |
| 2 | 1/2 @ 59004, 13/10 @ 69751, 21/2 @ 80514, 34/2 @ 91863 | `data/checks/<match>/set2/` |
| 3 | 1/2 @ 95959, 12/10 @ 105057, 23/10 @ 117456, 34/19 @ 126430 | `data/checks/<match>/set3/` |

Spot check of set 2 rally 13 shot 10 (return net, frame 69751): the far player is
lunging at the net with the racket at the shuttle, and the red marker sits at
their feet. This agrees with the earlier finding: offset 0, and `hit_x`/`hit_y`
mark the hitter's feet, not the shuttle.

The first shot with a hit point in each set is shot 2: serves (shot 1) have no `hit_x`/`hit_y`.

## Error handling (checked by running the command)
| Input | Result |
|---|---|
| Unknown match folder | `error: set CSV not found: .../NoSuchMatch/set1.csv (check the match folder and set number)`, exit 1 |
| Missing video | `error: video not found: data/nope.mp4`, exit 1 |
| `set_no` 4 | argparse error, exit 2 |
| Set CSV with no `hit_x`/`hit_y` (synthetic, temp dir) | `error: the set has no shots with hit_x and hit_y to check`, exit 1, no folder created |
| `--offset 999999` | each unreadable frame reported, then `error: no frames could be read; check the video and offset`, exit 1 |

No images are written to the project root.
