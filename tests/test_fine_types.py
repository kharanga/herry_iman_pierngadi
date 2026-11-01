import numpy as np
import pandas as pd
import pytest

from analytics.shot_types import TYPE_GROUP_BY_TYPE_EN
from shots.fine_types import (
    CANDIDATE_SPLITS,
    FINE_TYPES,
    SPLIT_MIN_F1,
    SPLIT_MIN_PAIR_GAIN,
    as_groups,
    labels_with_splits,
    main,
    out_of_fold_predictions,
    pair_scores,
    per_label_scores,
    split_holds_up,
)


def test_fine_types_are_the_eighteen_known_types_and_every_split_stays_inside_one_group():
    assert len(FINE_TYPES) == 18 and "unknown" not in FINE_TYPES
    for split_off_types in CANDIDATE_SPLITS.values():
        assert set(split_off_types) <= set(FINE_TYPES)
        assert len({TYPE_GROUP_BY_TYPE_EN[type_en] for type_en in split_off_types}) == 1
    assert (SPLIT_MIN_F1, SPLIT_MIN_PAIR_GAIN) == (0.5, 0.05)


def test_labels_with_splits_keeps_only_the_split_off_types_and_groups_the_rest():
    type_en = pd.Series(["push", "drive", "smash", "short service", "long service"])
    assert labels_with_splits(type_en, ("push",)).tolist() == ["push", "drive", "smash", "serve", "serve"]
    assert labels_with_splits(type_en, ("short service", "long service")).tolist()[3:] == ["short service", "long service"]
    assert labels_with_splits(type_en, ()).tolist() == [TYPE_GROUP_BY_TYPE_EN[name] for name in type_en]
    assert as_groups(["push", "drive", "serve", "short service"]).tolist() == ["drive", "drive", "serve", "serve"]


def test_pair_scores_count_only_shots_true_and_predicted_inside_the_group():
    type_en = pd.Series(["push", "push", "push", "drive", "drive", "back-court drive", "clear", "push"])
    labels = labels_with_splits(type_en, ("push",))
    predictions = np.array(["push", "push", "drive", "drive", "push", "drive", "drive", "clear"], dtype=object)
    scores = pair_scores(type_en, labels, predictions, ("push",))
    assert scores["pair_labels"] == ["drive", "push"] and scores["pair_shots"] == 6
    assert scores["pair_accuracy"] == pytest.approx(4 / 6) and scores["pair_majority"] == pytest.approx(3 / 6)
    assert scores["pair_gain"] == pytest.approx(1 / 6)


def holding(f1_values, gain):
    return {"pair": {"pair_labels": ["a", "b"], "pair_gain": gain}, "per_label": {"a": {"f1": f1_values[0]}, "b": {"f1": f1_values[1]}}}


def test_a_split_holds_up_only_with_both_sides_at_the_f1_bar_and_the_gain():
    assert split_holds_up(holding((0.5, 0.9), 0.05), ("a",))
    assert not split_holds_up(holding((0.49, 0.9), 0.3), ("a",))
    assert not split_holds_up(holding((0.9, 0.9), 0.049), ("a",))


def test_out_of_fold_predictions_never_train_on_the_test_match():
    rng = np.random.default_rng(0)
    match_ids = pd.Series(np.repeat([f"m{index}" for index in range(5)], 40))
    labels = pd.Series(np.where(np.arange(200) % 2, "net", "clear"))
    features = pd.DataFrame({"y": np.where(labels == "net", 1.0, 10.0) + rng.normal(0, 0.1, 200), "x": rng.normal(0, 1, 200)})
    predictions, fold_accuracies = out_of_fold_predictions(features, labels, match_ids)
    assert len(fold_accuracies) == 5 and (predictions == labels.to_numpy()).mean() > 0.95
    scores = per_label_scores(labels, np.array(["net"] * 200, dtype=object), ["clear", "net"])
    assert scores["clear"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 100}


def test_main_without_a_database_exits_without_creating_one(tmp_path):
    import duckdb

    with pytest.raises(duckdb.IOException):
        main([str(tmp_path / "none.duckdb"), str(tmp_path / "out.json")])
    assert list(tmp_path.iterdir()) == []
