# `ingest/fetch_shuttleset.sh`

Examples need: data/CoachAI-Projects/ShuttleSet/set (run ./ingest/fetch_shuttleset.sh)

Downloads the ShuttleSet dataset (MIT license, KDD 2023) from
https://github.com/wywyWang/CoachAI-Projects. Only the `ShuttleSet/` folder is
checked out; the rest of the repo is skipped.

## Command

```
./ingest/fetch_shuttleset.sh
```

No arguments. Run from the project root (paths are relative). Needs `git` and network access.

## What it fetches, and where

- Clones `CoachAI-Projects` into `data/CoachAI-Projects/` (gitignored) as a shallow
  (`--depth 1`), blobless (`--filter=blob:none`), sparse clone.
- Sparse checkout is set to `ShuttleSet`, so the files land in
  `data/CoachAI-Projects/ShuttleSet/`:
  - `set/match.csv`: one row per match
  - `set/homography.csv`: per-video homography and court corner pixels
  - `set/<match_folder>/set{1,2,3}.csv`: one row per shot (44 matches, 104 sets)
- This is the `SHUTTLESET_SET_DIR` that `ingest/load_shuttleset.py` reads (see `load_shuttleset.md`).

## Re-run behavior

- If `data/CoachAI-Projects/.git` already exists, the clone is skipped.
- Every run re-applies the sparse checkout (`ShuttleSet` only) and runs
  `git pull --ff-only`, so a re-run updates to the latest upstream commit and
  otherwise changes nothing.
- Safe to run repeatedly.

## Errors

The script uses `set -euo pipefail`, so it stops at the first failing command
with git's non-zero exit code:
- No network, or GitHub unreachable: the clone or pull fails.
- `data/CoachAI-Projects/` exists but is not a git clone: `git clone` refuses a
  non-empty directory. Delete that folder and re-run.
- Local changes or diverged history in the clone: `git pull --ff-only` fails.
  Delete `data/CoachAI-Projects/` and re-run.

A doctest checks that the files the loader needs are in place after a fetch:

```python
>>> from ingest.load_shuttleset import SHUTTLESET_SET_DIR
>>> (SHUTTLESET_SET_DIR / "match.csv").exists(), (SHUTTLESET_SET_DIR / "homography.csv").exists()
(True, True)
>>> len(list(SHUTTLESET_SET_DIR.glob("*/set*.csv")))
104

```
