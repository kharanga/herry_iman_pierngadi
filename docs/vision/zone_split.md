# vision/zone_split.py

Zones and habits of a labelled video under the old depth bands (1.98 m and 3.50 m from
the net) and the current ones (`analytics.court.DEPTH_EDGES_M`). Results:
`docs/results/club_zones.md`.

## Command

```
.venv/bin/python -m vision.zone_split --labels LABELS --database DATABASE --match-id ID --out DIR
```

DATABASE holds the video's own rows, built from its labelled hits with `vision.features`
(the hits files come from `vision.flight_types --hits-out`); it isn't changed. The command
writes two copies of it into DIR, `old_bands.duckdb` and `new_bands.duckdb`. In each one,
the match's own rows carry:
- their `hit_zone` and `landing_zone` recomputed under that band setting, from the rows'
  `hit_x/y` and `landing_x/y`;
- the labelled `type_group` and `type_detail`, matched by frame.

It prints:
- per player, the hit-zone and landing-zone counts under both settings;
- how many shots change zone;
- per player, the habits whose count changes (`analytics.tendencies` on each copy). The
  zone-mix and serve-placement habits are listed one by one, the other families as a
  count.

Exit codes: `0`; `1` with `error: ...` for a missing file or column; `2` for bad
arguments.

## Functions

`depth_edges(edges_m)` sets `analytics.court.DEPTH_EDGES_M` for the duration of a
`with` block and puts it back afterwards. `zones_of(rows, edges_m)` gives the zones of
rows under a band setting.

```python
>>> import pandas as pd
>>> import analytics.court as court
>>> from vision.zone_split import OLD_DEPTH_EDGES_M, changed_counts, depth_edges, zones_of
>>> with depth_edges(OLD_DEPTH_EDGES_M):
...     court.DEPTH_EDGES_M
(1.98, 3.5)
>>> court.DEPTH_EDGES_M
(2.75, 4.0)
>>> rows = pd.DataFrame({"player": ["me", "opp"], "hit_x": [4.0, 2.0], "hit_y": [4.3, 1.0], "landing_x": [2.0, 4.0], "landing_y": [9.0, 12.5]})
>>> old = zones_of(rows, OLD_DEPTH_EDGES_M)
>>> new = zones_of(rows, (2.75, 4.00))
>>> old["hit_zone"].tolist(), new["hit_zone"].tolist()
(['mid_forehand', 'rear_backhand'], ['front_forehand', 'rear_backhand'])
>>> old["landing_zone"].tolist(), new["landing_zone"].tolist()
(['mid_forehand', 'rear_backhand'], ['front_forehand', 'rear_backhand'])
>>> changed_counts(old, new)
{'hit_zone': (1, 2), 'landing_zone': (1, 2)}

```
