# ShuttleSet ingest — results

## How to run
```
./ingest/fetch_shuttleset.sh
.venv/bin/python -m ingest.load_shuttleset
.venv/bin/python -m pytest tests
```

## Output: `data/scout.duckdb`
| Table | Rows | Contents |
|---|---|---|
| `raw_shots` | 36,484 | every row of every `set/*/set*.csv`, plus `match_id`, `set_no`, `time_seconds`; `frame_num` cast to INTEGER; `type`, `lose_reason`, `win_reason` translated to English |
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

## English labels
The database holds no Chinese. Ingest translates `raw_shots.type`, `raw_shots.lose_reason`
and `raw_shots.win_reason` with the maps in `ingest/label_translations.py`
(documented in `docs/analytics/label_translations.md`). A label missing from a map
stops the build with `UnmappedLabelError` naming the column and the label; nothing becomes NULL silently.

| Criterion | Result |
|---|---|
| `type` only English | 19 English labels; both passive-drop spellings become `passive drop` (the data only uses `過度切球`), `未知球種` becomes `unknown` |
| `lose_reason`, `win_reason` only English | 7 English labels each; NULLs stay NULL (32,975 in each column) |
| No Chinese anywhere | 0 VARCHAR values with a CJK character in `raw_shots`, `matches`, `homography` |
| Every label mapped | 0 unmapped (a synthetic CSV with an unknown label fails the build and keeps the old tables) |
| Nothing else changes | 36,484 rows; every other column of `raw_shots`, and all of `matches` and `homography`, identical row by row to a build without translation |
| Counts preserved | every English value's count equals its Chinese source's count, and each row maps to the English of its own Chinese label (checked against a build without translation) |
| Idempotent | two rebuilds give identical tables |

Counts after translation:

| `type` | Shots |
|---|---|
| net shot | 6,290 |
| lob | 5,331 |
| return net | 3,620 |
| push | 2,925 |
| clear | 2,922 |
| smash | 2,586 |
| drop | 2,144 |
| short service | 2,051 |
| wrist smash | 1,648 |
| unknown | 1,407 |
| cross-court net shot | 1,371 |
| passive drop | 1,356 |
| drive | 700 |
| rush | 512 |
| back-court drive | 473 |
| defensive return drive | 406 |
| long service | 373 |
| defensive return lob | 301 |
| driven flight | 68 |

| `lose_reason` | `win_reason` | Shots | Rallies |
|---|---|---|---|
| opponent winner | winner | 1,189 | 1,189 |
| out | opponent out | 1,048 | 1,047 |
| net | opponent net | 841 | 841 |
| not over net | opponent not over net | 376 | 376 |
| misjudged landing | opponent misjudged landing | 44 | 44 |
| misjudged opponent landing | landed, opponent misjudged | 7 | 7 |
| foul | opponent foul | 4 | 4 |
| total | | 3,509 | 3,508 |

The two reason columns always pair up as in the table above (the same event seen from each side),
and both are NULL on every shot that does not end a rally. Not every rally has a reason:
3,508 of 3,683 rallies do. Shots and rallies differ by one because
`Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2018_Finals` set 3 rally 10 has its last shot
(ball_round 29, `out` / `opponent out`) twice in the source CSV; ingest keeps both rows.

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
