import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from analytics.build_shots import SCHEMA_PATH
from analytics.tendencies import (
    KNOWN_SWAPPED_RALLIES,
    MIN_SAMPLE_SIZE,
    UnknownMatchPlayerError,
    all_observations,
    family_observations,
    load_tendency_shots,
    reliable_shots,
    tendencies_for,
    tendencies_from_observations,
    wilson_lower_bound,
)
from analytics.validate_tendencies import validation_rows

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DATABASE_PATH = PROJECT_ROOT / "data" / "scout.duckdb"
TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b"
TENDENCY_KEYS = {"id", "family", "match_id", "player", "sets", "condition", "outcome", "n", "k", "rate", "reference_rate", "score", "label_source", "text"}


def observation_rows(match_id: str, player: str, set_no: int, condition: str, outcome_counts: dict, source: str = "shuttleset", family: str = "zone_mix", group_source: str = "label") -> list[dict]:
    return [
        {"family": family, "source": source, "match_id": match_id, "set_no": set_no, "player": player, "condition": condition, "outcome": outcome, "group_source": group_source}
        for outcome, count in outcome_counts.items()
        for _ in range(count)
    ]


def pro_rows(condition: str, outcome_counts: dict, match_count: int = 4) -> list[dict]:
    return [row for match_index in range(match_count) for row in observation_rows(f"pro_{match_index}", "A", 1, condition, outcome_counts)]


def test_three_of_three_never_appears_and_five_of_five_versus_fourteen_of_eighteen_follow_the_lower_bound():
    observations = pd.DataFrame(
        pro_rows("rear_backhand", {"drop": 5, "clear": 5})
        + pro_rows("front_forehand", {"drop": 5, "clear": 5})
        + pro_rows("mid_middle", {"drop": 5, "clear": 5})
        + observation_rows("my_match", "me", 1, "rear_backhand", {"drop": 3})
        + observation_rows("my_match", "me", 1, "front_forehand", {"drop": 14, "clear": 4})
        + observation_rows("my_match", "me", 1, "mid_middle", {"drop": 5})
    )
    tendencies = tendencies_from_observations(observations, "my_match", "me")
    ids = [tendency["id"] for tendency in tendencies]
    assert "zone_mix:rear_backhand:drop" not in ids
    assert all(tendency["n"] >= MIN_SAMPLE_SIZE for tendency in tendencies)
    assert wilson_lower_bound(3, 3) < wilson_lower_bound(14, 18)
    assert ids.index("zone_mix:mid_middle:drop") < ids.index("zone_mix:front_forehand:drop")
    assert wilson_lower_bound(5, 5) > wilson_lower_bound(14, 18)


def test_more_evidence_at_the_same_rate_ranks_higher():
    observations = pd.DataFrame(
        pro_rows("rear_backhand", {"drop": 5, "clear": 5})
        + pro_rows("front_forehand", {"drop": 5, "clear": 5})
        + observation_rows("my_match", "me", 1, "rear_backhand", {"drop": 8, "clear": 2})
        + observation_rows("my_match", "me", 1, "front_forehand", {"drop": 40, "clear": 10})
    )
    ids = [tendency["id"] for tendency in tendencies_from_observations(observations, "my_match", "me")]
    assert ids.index("zone_mix:front_forehand:drop") < ids.index("zone_mix:rear_backhand:drop")


def test_every_tendency_has_the_documented_fields_and_its_evidence_in_the_text():
    observations = pd.DataFrame(pro_rows("front_forehand", {"drop": 2, "clear": 8}) + observation_rows("my_match", "me", 1, "front_forehand", {"drop": 14, "clear": 4}))
    tendencies = tendencies_from_observations(observations, "my_match", "me")
    assert tendencies
    for tendency in tendencies:
        assert set(tendency) == TENDENCY_KEYS
        assert tendency["rate"] == tendency["k"] / tendency["n"]
        assert f"{tendency['k']} of {tendency['n']}" in tendency["text"]
        assert tendency["label_source"] in {"labels", "predictions", "mixed"}
    json.dumps(tendencies)


def test_label_source_records_predictions_and_mixtures():
    observations = pd.DataFrame(
        pro_rows("front_forehand", {"drop": 2, "clear": 8})
        + pro_rows("rear_backhand", {"drop": 2, "clear": 8})
        + observation_rows("my_match", "me", 1, "front_forehand", {"drop": 14, "clear": 4}, source="own", group_source="prediction")
        + observation_rows("my_match", "me", 1, "rear_backhand", {"drop": 10}, source="own", group_source="prediction")
        + observation_rows("my_match", "me", 1, "rear_backhand", {"drop": 4}, source="own", group_source="label")
    )
    sources = {tendency["id"]: tendency["label_source"] for tendency in tendencies_from_observations(observations, "my_match", "me")}
    assert sources == {"zone_mix:front_forehand:drop": "predictions", "zone_mix:rear_backhand:drop": "mixed"}


def test_reference_excludes_the_analysed_match_and_own_rows():
    base = pro_rows("front_forehand", {"drop": 2, "clear": 8}) + observation_rows("my_match", "me", 1, "front_forehand", {"drop": 14, "clear": 4})
    with_own_match_noise = base + observation_rows("my_match", "opp", 2, "front_forehand", {"drop": 500}) + observation_rows("other_own", "me", 1, "front_forehand", {"drop": 500}, source="own")
    plain = tendencies_from_observations(pd.DataFrame(base), "my_match", "me")
    noisy = tendencies_from_observations(pd.DataFrame(with_own_match_noise), "my_match", "me")
    assert [(tendency["id"], tendency["reference_rate"]) for tendency in plain] == [(tendency["id"], tendency["reference_rate"]) for tendency in noisy]
    assert plain[0]["reference_rate"] == pytest.approx(0.2)


def test_set_one_tendencies_do_not_depend_on_set_two():
    set_one = pro_rows("front_forehand", {"drop": 2, "clear": 8}) + observation_rows("my_match", "me", 1, "front_forehand", {"drop": 14, "clear": 4})
    with_set_two = set_one + observation_rows("my_match", "me", 2, "front_forehand", {"clear": 40})
    assert tendencies_from_observations(pd.DataFrame(set_one), "my_match", "me", sets=(1,)) == tendencies_from_observations(
        pd.DataFrame(with_set_two), "my_match", "me", sets=(1,)
    )


def test_situations_without_any_pro_data_are_skipped():
    observations = pd.DataFrame(pro_rows("front_forehand", {"drop": 2, "clear": 8}) + observation_rows("my_match", "me", 1, "rear_middle", {"smash": 20}))
    assert tendencies_from_observations(observations, "my_match", "me") == []


def shot_row(source: str, match_id: str, set_no: int, rally: int, ball_round: int, player: str, **columns) -> dict:
    return {"source": source, "match_id": match_id, "set_no": set_no, "rally": rally, "ball_round": ball_round, "frame_num": 100 * rally + ball_round, "player": player, **columns}


def write_shots_database(database_path: Path, rows: list[dict]) -> Path:
    connection = duckdb.connect(str(database_path))
    connection.execute(SCHEMA_PATH.read_text())
    frame = pd.DataFrame(rows)
    connection.register("synthetic_rows", frame)
    columns = ", ".join(frame.columns)
    connection.execute(f"INSERT INTO shots ({columns}) SELECT {columns} FROM synthetic_rows")
    connection.close()
    return database_path


def test_one_shot_unknown_rallies_and_known_swapped_rallies_are_left_out(tmp_path):
    swapped_match, swapped_set, swapped_rally = sorted(KNOWN_SWAPPED_RALLIES)[0]
    rows = [
        shot_row("shuttleset", "m", 1, 1, 1, "A", type_en="unknown", type_group="unknown", type_pred="serve", hit_zone="mid_middle"),
        shot_row("shuttleset", "m", 1, 2, 1, "A", type_en="short service", type_group="serve", hit_zone="mid_middle"),
        shot_row("shuttleset", swapped_match, swapped_set, swapped_rally, 1, "A", type_en="short service", type_group="serve"),
        shot_row("shuttleset", swapped_match, swapped_set, swapped_rally, 2, "B", type_en="net shot", type_group="net", hit_zone="front_middle"),
        shot_row("shuttleset", "m", 1, 3, 1, "A", type_en="unknown", type_group="unknown", type_pred="serve"),
        shot_row("shuttleset", "m", 1, 3, 2, "B", type_en="lob", type_group="lift", hit_zone="front_middle"),
    ]
    connection = duckdb.connect(str(write_shots_database(tmp_path / "scout.duckdb", rows)), read_only=True)
    kept = reliable_shots(load_tendency_shots(connection))
    connection.close()
    kept_keys = set(zip(kept["match_id"], kept["rally"], kept["ball_round"]))
    assert kept_keys == {("m", 2, 1), ("m", 3, 1), ("m", 3, 2)}


def test_own_rows_with_me_and_opp_use_predictions_end_to_end(tmp_path):
    rows = []
    for match_index in range(3):
        for rally in range(1, 21):
            rows.append(shot_row("shuttleset", f"pro_{match_index}", 1, rally, 1, "A", type_en="short service", type_group="serve", type_pred="serve", landing_zone="front_middle"))
            rows.append(shot_row("shuttleset", f"pro_{match_index}", 1, rally, 2, "B", type_en="net shot" if rally % 2 else "lob", type_group="net" if rally % 2 else "lift", hit_zone="front_forehand", getpoint_player="A"))
    for rally in range(1, 21):
        rows.append(shot_row("own", "my_game", 1, rally, 1, "opp", type_pred="serve", landing_zone="front_middle"))
        rows.append(shot_row("own", "my_game", 1, rally, 2, "me", type_pred="lift", hit_zone="front_forehand", getpoint_player="opp"))
    database_path = write_shots_database(tmp_path / "scout.duckdb", rows)
    tendencies = tendencies_for("my_game", "me", database_path)
    assert tendencies and {tendency["player"] for tendency in tendencies} == {"me"}
    assert {tendency["label_source"] for tendency in tendencies} == {"predictions"}
    assert "response:serve:lift" in [tendency["id"] for tendency in tendencies]
    with pytest.raises(UnknownMatchPlayerError):
        tendencies_for("my_game", "nobody", database_path)


def test_cli_unknown_player_exits_one_and_missing_arguments_exit_two(tmp_path):
    database_path = write_shots_database(tmp_path / "scout.duckdb", [shot_row("own", "my_game", 1, 1, 1, "me", type_pred="serve", landing_zone="front_middle")])
    unknown = subprocess.run([sys.executable, "-m", "analytics.tendencies", "my_game", "nobody", "", str(database_path)], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert unknown.returncode == 1 and "no shots for player 'nobody'" in unknown.stderr
    missing = subprocess.run([sys.executable, "-m", "analytics.tendencies"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert missing.returncode == 2


@pytest.fixture(scope="module")
def real_observations():
    if not REAL_DATABASE_PATH.exists():
        pytest.skip("data/scout.duckdb not built")
    return all_observations(REAL_DATABASE_PATH)


def test_runs_on_every_shuttleset_player_with_n_at_least_five(real_observations):
    players = real_observations[real_observations["source"] == "shuttleset"][["match_id", "player"]].drop_duplicates()
    assert len(players) == 88
    for match_id, player in players.itertuples(index=False):
        tendencies = tendencies_from_observations(real_observations, match_id, player)
        assert tendencies, (match_id, player)
        assert min(tendency["n"] for tendency in tendencies) >= MIN_SAMPLE_SIZE
        assert [tendency["score"] for tendency in tendencies] == sorted((tendency["score"] for tendency in tendencies), reverse=True)


def test_set_one_selection_does_not_change_when_set_two_changes(real_observations):
    match_id = "Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals"
    is_axelsen_set_two = (real_observations["match_id"] == match_id) & (real_observations["player"] == "A") & (real_observations["set_no"] == 2)
    changed = real_observations.copy()
    changed.loc[is_axelsen_set_two, "outcome"] = changed.loc[is_axelsen_set_two, "outcome"].sample(frac=1.0, random_state=0).to_numpy()
    changed.loc[is_axelsen_set_two & (changed["family"] == "zone_mix"), "outcome"] = "clear"
    columns = ["id", "set_one_n", "set_one_rate", "reference_rate", "score", "is_tendency", "is_top_bullet"]
    original_rows = validation_rows(real_observations)
    changed_rows = validation_rows(changed)
    original = original_rows[(original_rows["match_id"] == match_id) & (original_rows["player"] == "A")][columns].reset_index(drop=True)
    after_change = changed_rows[(changed_rows["match_id"] == match_id) & (changed_rows["player"] == "A")][columns].reset_index(drop=True)
    assert original["is_top_bullet"].sum() == 3
    pd.testing.assert_frame_equal(original, after_change)


@pytest.mark.parametrize("source_name", ["tendencies.py", "validate_tendencies.py"])
def test_tendency_sources_have_no_comments_or_docstrings(source_name):
    source_text = (PROJECT_ROOT / "analytics" / source_name).read_text()
    module = ast.parse(source_text)
    assert ast.get_docstring(module) is None
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node) is None
    assert [line for line in source_text.splitlines() if line.lstrip().startswith("#")] == []


@pytest.mark.parametrize("doc_name", ["analytics/tendencies.md", "analytics/validate_tendencies.md", "results/tendencies.md"])
def test_tendency_docs_have_no_task_ids(doc_name):
    assert re.findall(TASK_ID_PATTERN, (PROJECT_ROOT / "docs" / doc_name).read_text()) == []


from analytics.validate_tendencies import matched_chance, set_two_rates_by_situation, with_matched_chance

REAL_VALIDATION_PATH = PROJECT_ROOT / "data" / "checks" / "tendencies" / "set_one_to_set_two.json"
RESULTS_PATH = PROJECT_ROOT / "docs" / "results" / "tendencies.md"


def test_matched_chance_counts_other_players_in_the_same_situation_including_zero_outcomes():
    observations = pd.DataFrame(
        observation_rows("my_match", "A", 2, "front_backhand", {"lift": 4, "drop": 1})
        + observation_rows("other_1", "A", 2, "front_backhand", {"lift": 5})
        + observation_rows("other_1", "B", 2, "front_backhand", {"drop": 5})
        + observation_rows("other_2", "A", 2, "front_backhand", {"lift": 2, "drop": 3})
        + observation_rows("other_2", "B", 2, "front_backhand", {"lift": 1, "drop": 3})
        + observation_rows("other_3", "A", 2, "rear_forehand", {"lift": 10})
        + observation_rows("other_3", "A", 2, "front_backhand", {"lift": 10}, family="response")
    )
    set_two_rates = set_two_rates_by_situation(observations)
    assert matched_chance("zone_mix", "front_backhand", "lift", 0.373, "my_match", set_two_rates) == pytest.approx(2 / 3)
    assert matched_chance("zone_mix", "front_backhand", "lift", 0.0, "my_match", set_two_rates) == pytest.approx(2 / 3)
    assert matched_chance("zone_mix", "front_backhand", "drop", 0.5, "my_match", set_two_rates) == pytest.approx(2 / 3)
    assert matched_chance("zone_mix", "front_backhand", "drop", 0.9, "my_match", set_two_rates) == pytest.approx(1 / 3)
    assert matched_chance("zone_mix", "rear_forehand", "lift", 0.5, "my_match", set_two_rates) == 1.0
    assert matched_chance("zone_mix", "mid_middle", "lift", 0.5, "my_match", set_two_rates) is None
    assert matched_chance("zone_mix", "front_backhand", "lift", 0.5, "other_1", set_two_rates) == pytest.approx(1 / 2)


def test_with_matched_chance_leaves_non_evaluable_rows_empty():
    observations = pd.DataFrame(observation_rows("other_1", "A", 2, "front_backhand", {"lift": 5}))
    rows = pd.DataFrame(
        [
            {"id": "zone_mix:front_backhand:lift", "reference_rate": 0.3, "match_id": "my_match", "is_evaluable": True},
            {"id": "zone_mix:front_backhand:lift", "reference_rate": 0.3, "match_id": "my_match", "is_evaluable": False},
        ]
    )
    matched = with_matched_chance(rows, observations)["matched_chance"].tolist()
    assert matched[0] == 1.0 and pd.isna(matched[1])


def documented_rows(section_heading: str) -> list[list[str]]:
    section = RESULTS_PATH.read_text().split(section_heading)[1].split("\n## ")[0]
    table_lines = [line for line in section.splitlines() if line.startswith("|")]
    return [[cell.strip().strip("*`") for cell in line.strip("|").split("|")] for line in table_lines[2:]]


@pytest.mark.skipif(not REAL_VALIDATION_PATH.exists(), reason="run analytics.validate_tendencies first")
def test_results_doc_tables_equal_the_validation_json():
    summary = json.loads(REAL_VALIDATION_PATH.read_text())
    for name, evaluable, follows, follow_rate, matched, lift in documented_rows("## Set 1 -> set 2"):
        selection = summary["selections"][name]
        assert (int(evaluable.replace(",", "")), int(follows.replace(",", ""))) == (selection["evaluable"], selection["follows"]), name
        assert follow_rate == f"{selection['follow_rate']:.1%}" and matched == f"{selection['matched_chance']:.1%}", name
        assert lift == f"{selection['lift_over_matched_chance'] * 100:+.1f}", name
    by_family_sections = RESULTS_PATH.read_text().split("## By family")[1].split("\n## ")[0].split("Top 3 per player only:")
    for section_text, summary_key in zip(by_family_sections, ("tendencies_by_family", "top_bullets_by_family")):
        table_lines = [line for line in section_text.splitlines() if line.startswith("|")][2:]
        assert len(table_lines) == 6
        for line in table_lines:
            family, evaluable, follow_rate, matched, lift = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
            scores = summary[summary_key][family]
            assert int(evaluable) == scores["evaluable"] and follow_rate == f"{scores['follow_rate']:.1%}", (summary_key, family)
            assert matched == f"{scores['matched_chance']:.1%}" and lift == f"{scores['lift_over_matched_chance'] * 100:+.1f}", (summary_key, family)
    counts = summary["per_player_set_counts"]
    results_text = RESULTS_PATH.read_text()
    assert f"{counts['shots']['min']} to {counts['shots']['max']} shots in a set (median {counts['shots']['median']:g})" in results_text
    assert f"{counts['rally_ending_shots']['min']} to {counts['rally_ending_shots']['max']} rally-ending shots in a set" in results_text
    assert f"{counts['shots']['player_sets']} player-sets" in results_text


@pytest.mark.skipif(not REAL_VALIDATION_PATH.exists() or not REAL_DATABASE_PATH.exists(), reason="needs data/scout.duckdb and the validation JSON")
def test_builder_brute_forced_row_reproduces(real_observations):
    set_two_rates = set_two_rates_by_situation(real_observations)
    match_id = "An_Se_Young_Pornpawee_Chochuwong_TOYOTA_THAILAND_OPEN_2021_QuarterFinals"
    pool = [entry for entry in set_two_rates["players_by_situation"][("zone_mix", "front_backhand")] if entry[0] != match_id]
    assert len(pool) == 85
    assert matched_chance("zone_mix", "front_backhand", "lift", 0.373, match_id, set_two_rates) == pytest.approx(38 / 85)
