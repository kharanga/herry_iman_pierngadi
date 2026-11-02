# vision/club_classifier.py

Feet-only shot classifiers that use the user's club labels:
- **club:** trained on the club labels alone;
- **pro_plus_club:** trained on ShuttleSet's labelled shots plus the club labels, the club
  rows carrying 1/3 of the sample weight (`CLUB_WEIGHT_SHARE`);
- **average:** the shipped pro classifier and the club one, their probabilities over the
  8 shot details averaged 50/50 (`AVERAGE_PRO_SHARE`).

Results: `docs/results/shot_classifier_club.md`. Nothing here is used by the pipeline.

## Command

```
.venv/bin/python -m vision.club_classifier --train VIDEO LABELS DATABASE END [--train ...] \
    [--shuttleset-database data/scout.duckdb] [--pro-model data/models/shot_type_classifier.joblib] \
    --predict VIDEO DATABASE END OUT_DIR [--predict ...]
```

- **Training rows:** the training databases hold the club videos' own rows built from the
  labelled hits, as in `vision.flight_types`. The ShuttleSet rows are the shipped
  classifier's training rows (`shots.train`).
- **Typing:** each `--predict` database's own rows with a hitter and a landing point are
  typed by every candidate. Each candidate writes `OUT_DIR/<candidate>.duckdb`, a copy
  with only this match's own predictions replaced (`vision.flight_types.write_predictions`).
- **Exit codes:** `0`; `1` with `error: ...` for a missing file or column; `2` for bad
  arguments.

## Functions

```python
>>> import numpy as np, pandas as pd
>>> from vision.club_classifier import averaged_details, probabilities_over, shares_as_weights
>>> weights = shares_as_weights(pro_count=6, club_count=2)
>>> weights.round(3).tolist(), round(float(weights[6:].sum() / weights.sum()), 3)
([0.889, 0.889, 0.889, 0.889, 0.889, 0.889, 1.333, 1.333], 0.333)
>>> class Fixed:
...     def __init__(self, classes, row):
...         self.classes_, self.row = np.array(classes), np.array(row)
...     def predict_proba(self, features):
...         return np.tile(self.row, (len(features), 1))
>>> pro = Fixed(["drive", "lift"], [0.6, 0.4])
>>> club = Fixed(["drive", "lift", "smash"], [0.1, 0.8, 0.1])
>>> probabilities_over(pro, pd.DataFrame({"x": [0]}))[0].round(2).tolist()
[0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.4]
>>> averaged_details(pro, club, pd.DataFrame({"x": [0, 1]})).tolist()
['lift', 'lift']

```

`probabilities_over` puts a classifier's probabilities in the order of
`vision.flight_types.CLASSIFIED_DETAILS`, with 0 for a detail it never saw.
`averaged_details` takes the detail with the highest averaged probability.
