import numpy as np
import pytest

from vision.hit_training import inner_training_sources, main, match_weights


def test_inner_sources_leave_one_out_unless_there_is_only_one():
    assert inner_training_sources(["club:a", "club:b"], "club:a") == ["club:b"]
    assert inner_training_sources(["club:a"], "club:a") == ["club:a"]
    assert inner_training_sources(["m1", "m2", "club:a"], "m2") == ["m1", "club:a"]


def test_two_clubs_only_get_half_the_weight_each_whatever_their_size():
    rows = {"club:a": (np.zeros((300, 2)), np.zeros(300)), "club:b": (np.zeros((100, 2)), np.zeros(100))}
    weights = match_weights(["club:a", "club:b"], rows, ["club:a", "club:b"], 1.0)
    assert len(weights) == 400 and weights.mean() == pytest.approx(1.0)
    assert weights[:300].sum() == pytest.approx(weights[300:].sum())
    single = match_weights(["club:a"], rows, ["club:a"], 1.0)
    assert np.allclose(single, 1.0)


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--clubs-only", "--club", "v.mp4", "l.csv"], "--clubs-only needs --save"),
        (["--clubs-only", "--save", "m.joblib"], "--clubs-only needs --save"),
        (["--clubs-only", "--save", "m.joblib", "--club", "v.mp4", "l.csv", "--held-out"], "--save (and --tune) only"),
        (["--clubs-only", "--save", "m.joblib", "--club", "v.mp4", "l.csv", "--by-player"], "--save (and --tune) only"),
        (["--save", "m.joblib"], "give a match list, or --clubs-only"),
    ],
)
def test_clubs_only_refusals_and_a_missing_match_list(arguments, message, capsys, tmp_path):
    assert main(arguments) != 0
    assert message in capsys.readouterr().err
    assert not (tmp_path / "m.joblib").exists()


def test_clubs_only_never_reads_the_match_list(monkeypatch, tmp_path, capsys):
    import vision.hit_training as hit_training
    monkeypatch.setattr(hit_training, "loaded_matches", lambda path: pytest.fail("match list read"))
    monkeypatch.setattr(hit_training, "load_club", lambda video, labels: (_ for _ in ()).throw(FileNotFoundError("club video not found")))
    assert main(["--clubs-only", "--save", str(tmp_path / "m.joblib"), "--club", "v.mp4", "l.csv"]) != 0
    assert "club video not found" in capsys.readouterr().err
