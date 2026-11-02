import numpy as np
import pandas as pd

from vision.review_report import confusion_lines, lag_lines, retimed_rejections, split_rows


def kept_table(rows):
    return pd.DataFrame(rows, columns=["rally", "frame_num", "player", "type_group", "type_detail", "source", "score", "model_frame", "model_type_detail", "model_player"])


def test_split_rows_separates_reviewed_added_and_unreviewed():
    kept = kept_table([
        (1, 10, "me", "serve", "serve_short", "confirmed", 0.9, 10, "serve_short", "me"),
        (1, 20, "opp", "net", "net", "model", 0.5, 20, "net", "opp"),
        (1, 30, "me", "lift", "lift", "hand", np.nan, np.nan, "", ""),
    ])
    reviewed, added, unreviewed = split_rows(kept)
    assert reviewed["frame_num"].tolist() == [10] and added["frame_num"].tolist() == [30] and unreviewed["frame_num"].tolist() == [20]


def test_retimed_pairs_respect_the_reach_take_the_nearest_and_use_each_added_hit_once():
    rejected = pd.DataFrame({"frame_num": [100, 105, 400], "player": ["me", "opp", "me"], "model_type_detail": ["serve_short", "net", "lift"]})
    added = pd.DataFrame({"frame_num": [102, 130, 431], "player": ["me", "opp", "me"], "type_group": ["serve", "net", "lift"]}, index=[7, 8, 9])
    pairs = retimed_rejections(rejected, added, 30)
    assert pairs[["model_frame", "added_frame"]].values.tolist() == [[100, 102], [105, 130]]
    assert pairs["model_type"].tolist() == ["serve", "net"] and pairs["same_player"].tolist() == [True, True]
    assert retimed_rejections(rejected, added, 31)["added_frame"].tolist() == [102, 130, 431]
    assert len(retimed_rejections(rejected, added.iloc[:0], 30)) == 0


def test_a_late_model_hit_has_a_positive_lag():
    kept = kept_table([(1, 100, "me", "smash", "smash", "moved", 0.9, 106, "smash", "me"), (1, 200, "opp", "net", "net", "moved", 0.9, 197, "net", "opp")])
    lines = lag_lines(kept, 30.0)
    all_row = next(line for line in lines if line.startswith("| all |"))
    assert "-3, +6" in all_row and "+1.5 f" in all_row


def test_the_confusion_counts_reviewed_rows_only_and_scores_the_diagonal():
    kept = kept_table([
        (1, 10, "me", "lift", "lift", "confirmed", 0.9, 10, "lift", "me"),
        (1, 20, "opp", "drive", "drive", "corrected_type", 0.9, 20, "lift", "opp"),
        (1, 30, "me", "lift", "lift", "model", 0.9, 30, "lift", "me"),
    ])
    reviewed, _, _ = split_rows(kept)
    lines = confusion_lines(reviewed)
    assert lines[0] == "| model \\\\ user | lift | drive | right |".replace("\\\\", "\\")
    assert lines[2] == "| lift | 1 | 1 | 1/2 = 50.0% |"
