import numpy as np
import pandas as pd
import pytest

import vision.club_classifier as club_classifier
from vision.club_classifier import CANDIDATES, averaged_details, candidate_details, probabilities_over, shares_as_weights
from vision.flight_types import CLASSIFIED_DETAILS


class FixedModel:
    def __init__(self, classes, probabilities, predicted=None):
        self.classes_ = np.array(classes)
        self.probabilities = np.array(probabilities, dtype=float)
        self.predicted = predicted

    def predict_proba(self, features):
        return np.repeat(self.probabilities[None, :], len(features), axis=0)

    def predict(self, features):
        return np.array([self.predicted] * len(features))


def test_the_club_rows_carry_a_third_of_the_weight_and_the_mean_weight_is_one():
    weights = shares_as_weights(300, 30)
    assert len(weights) == 330 and weights.mean() == pytest.approx(1.0)
    assert weights[300:].sum() / weights.sum() == pytest.approx(1 / 3)
    assert np.allclose(weights[:300], weights[0]) and np.allclose(weights[300:], weights[300])


def test_probabilities_follow_the_label_order_with_zero_for_an_unseen_label():
    model = FixedModel(["lift", "clear", "net"], [0.2, 0.5, 0.3])
    probabilities = probabilities_over(model, pd.DataFrame({"a": [1, 2]}))
    assert probabilities.shape == (2, len(CLASSIFIED_DETAILS))
    row = dict(zip(CLASSIFIED_DETAILS, probabilities[0]))
    assert row["clear"] == 0.5 and row["net"] == 0.3 and row["lift"] == 0.2 and row["smash"] == 0.0


def test_the_average_is_the_argmax_of_the_blend_and_a_full_pro_share_is_the_pro_choice():
    features = pd.DataFrame({"a": [1]})
    pro = FixedModel(["clear", "lift"], [0.6, 0.4])
    club = FixedModel(["clear", "lift"], [0.1, 0.9])
    assert averaged_details(pro, club, features).tolist() == ["lift"]
    assert averaged_details(pro, club, features, pro_share=1.0).tolist() == ["clear"]
    assert averaged_details(pro, club, features, pro_share=0.0).tolist() == ["lift"]


def test_the_three_candidates_come_from_their_models():
    features = pd.DataFrame({"a": [1, 2]})
    models = {"pro": FixedModel(["clear", "drop"], [0.8, 0.2]), "club": FixedModel(["clear", "drop"], [0.0, 1.0], "drop"), "pro_plus_club": FixedModel(["smash"], [1.0], "smash")}
    details = candidate_details(models, features)
    assert tuple(details) == CANDIDATES
    assert details["club"].tolist() == ["drop", "drop"] and details["pro_plus_club"].tolist() == ["smash", "smash"] and details["average"].tolist() == ["drop", "drop"]


def test_club_rows_keep_labelled_details_with_feet_points(monkeypatch, tmp_path):
    rows = pd.DataFrame({"row_id": [1, 2, 3, 4], "frame_num": [10, 20, 30, 40], "has_points": [True, False, True, True]})
    monkeypatch.setattr(club_classifier, "shot_rows", lambda source: rows)
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"frame_num": [10, 20, 30, 40], "type_detail": ["lift", "net", "unknown", "smash"]}).to_csv(labels, index=False)
    kept = club_classifier.club_rows(None, labels)
    assert kept["frame_num"].tolist() == [10, 40] and kept["type_detail"].tolist() == ["lift", "smash"]
