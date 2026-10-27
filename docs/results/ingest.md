# M1: ShuttleSet ingest — results

## How to run
```
./ingest/fetch_shuttleset.sh
.venv/bin/python ingest/load_shuttleset.py
.venv/bin/python -m pytest tests
```

## Output: `data/scout.duckdb`
| Table | Rows | Contents |
|---|---|---|
| `raw_shots` | 36,484 | every row of every `set/*/set*.csv`, plus `match_id`, `set_no`, `time_seconds`; `frame_num` cast to INTEGER |
| `matches` | 44 | `match.csv`, plus `match_id`, `fps_estimate`, `fps_fit_r2`, `fps` |
| `homography` | 44 | `homography.csv`, with `homography_matrix` parsed to `DOUBLE[3][3]` |

## Acceptance
| Criterion | Result |
|---|---|
| Row count matches source CSVs | 36,484 = 36,484 (also checked per set) |
| 44 matches, 104 sets | 44 matches, 104 sets; set count per match agrees with `match.csv` |
| Every row has `match_id` and `set_no` | 0 missing |
| fps for all 44 matches | 19 at 25 fps, 25 at 30 fps |
| Idempotent | full rebuild in one transaction with `CREATE OR REPLACE`; two rebuilds give identical tables |

## fps estimation
`fps_estimate` is the least-squares slope of `frame_num` against `time_seconds` over all shots in a match.
`time` is only accurate to the second, but each match has hundreds of shots spread over about an hour, so the slope is still precise:
the lowest R² is 0.999999. Estimates were 25.00 for the 25 fps matches and between 29.97 and 30.00 for the rest.
`fps` is the estimate rounded. For Axelsen vs Ginting (Thailand Open 2021 SF) the estimate is 30.0004,
which agrees with the downloaded video (30.00 fps).

## Data notes
- The ShuttleSet README says 36,492 strokes; the CSVs in the repo contain 36,484. We use the CSVs.
- `time` comes in two formats (`00:03:54` and `0:11:15`, the second in 4,423 rows), so it is read as text and parsed manually.
- One folder name contains a space (`Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals`); `match_id` keeps it as-is so it still joins to `match.csv`.
- `hit_x`/`hit_y` are pixel positions at the hitter's feet, not where the shuttle is (seen in the alignment check: `data/checks/`). Labels and video were aligned with offset 0 for all three sets of Axelsen vs Ginting.
