import importlib
from pathlib import Path

import pytest

MODULES = [
    "shots.predict",
    "shots.train",
    "shots.compare_features",
    "shots.compare_hit_points",
    "shots.results_tables",
    "analytics.build_shots",
    "analytics.spot_check",
    "analytics.tendencies",
    "report.preview",
    "ingest.load_shuttleset",
]


@pytest.mark.parametrize("module_name", MODULES)
@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_prints_usage_exits_zero_and_creates_no_file(module_name, flag, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        importlib.import_module(module_name).main([flag])
    assert stopped.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: python -m {module_name}")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module_name", MODULES)
def test_an_unknown_flag_is_a_usage_error_not_a_path(module_name, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        importlib.import_module(module_name).main(["a", "b", "--foo"])
    assert stopped.value.code == 2 and "--foo" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_positional_paths_and_defaults_are_unchanged():
    predict = importlib.import_module("shots.predict")
    assert predict.parse_arguments([]).database_path == Path("data/scout.duckdb")
    assert predict.parse_arguments(["x.duckdb", "m.joblib"]).model_path == Path("m.joblib")
    results = importlib.import_module("shots.results_tables").parse_arguments(["m", "c", "d", "h"])
    assert (results.model_path, results.comparison_path, results.database_path, results.hit_point_comparison_path) == (Path("m"), Path("c"), Path("d"), Path("h"))
    spot = importlib.import_module("analytics.spot_check").parse_arguments(["d.duckdb", "out.png"])
    assert (spot.database_path, spot.output_path) == (Path("d.duckdb"), Path("out.png"))


def test_tendencies_arguments_keep_their_order_and_empty_sets_mean_the_whole_match():
    tendencies = importlib.import_module("analytics.tendencies")
    parsed = tendencies.parse_arguments(["club_1", "me", "", "x.duckdb"])
    assert (parsed.match_id, parsed.player, parsed.database_path) == ("club_1", "me", Path("x.duckdb"))
    assert tendencies.parse_sets(parsed.sets) is None and tendencies.parse_sets(tendencies.parse_arguments(["m", "A", "1,3"]).sets) == (1, 3)
    with pytest.raises(SystemExit) as stopped:
        tendencies.main(["club_1"])
    assert stopped.value.code == 2


def test_preview_default_png_is_named_after_the_report_folder():
    preview = importlib.import_module("report.preview")
    assert preview.default_preview_path(Path("data/reports/stand_in_set1/index.html")) == preview.PREVIEW_DIR / f"stand_in_set1_{preview.PHONE_WIDTH_PX}px.png"
    with pytest.raises(SystemExit) as stopped:
        preview.main([])
    assert stopped.value.code == 2
