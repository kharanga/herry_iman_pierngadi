import ast
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd
import pytest

import analytics.tendencies as tendencies_module
from analytics.build_shots import SCHEMA_PATH
from analytics.court import ZONE_NAMES
from analytics.tendencies import (
    FAMILIES,
    GROUP_ORDER,
    KNOWN_SWAPPED_RALLIES,
    UnknownMatchPlayerError,
    all_observations,
    family_observations,
    load_tendency_shots,
    reliable_shots,
    tendencies_for,
    tendencies_from_observations,
    tendency_text,
)
from analytics.court import NOT_OVER_NET_ZONE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DATABASE_PATH = PROJECT_ROOT / "data" / "scout.duckdb"
RESULTS_PATH = PROJECT_ROOT / "docs" / "results" / "tendencies.md"
AXELSEN_GINTING = "Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals"
TASK_ID_PATTERN = r"\b(ANA|VIS|INT)-\d+\b"
PROS_PATTERN = r"(?i)\bpros?\b|professional"
TENDENCY_KEYS = {"id", "family", "match_id", "player", "sets", "condition", "outcome", "n", "k", "rate", "label_source", "text"}
REMOVED_NAMES = ("MIN_SAMPLE_SIZE", "MIN_SCORE", "CONFIDENCE_Z", "reference_rates", "wilson_lower_bound", "tendency_score", "candidate_tendencies")


def observation_rows(match_id: str, player: str, set_no: int, condition: str, outcome_counts: dict, source: str = "shuttleset", family: str = "zone_mix", group_source: str = "label") -> list[dict]:
    return [
        {"family": family, "source": source, "match_id": match_id, "set_no": set_no, "player": player, "condition": condition, "outcome": outcome, "group_source": group_source}
        for outcome, count in outcome_counts.items()
        for _ in range(count)
    ]


def situation_key(tendency: dict) -> tuple[str, str]:
    return tendency["family"], tendency["condition"]


def assert_choices_add_up_to_n(tendencies: list[dict]) -> int:
    k_sums, ns = Counter(), {}
    for tendency in tendencies:
        k_sums[situation_key(tendency)] += tendency["k"]
        assert ns.setdefault(situation_key(tendency), tendency["n"]) == tendency["n"]
    assert dict(k_sums) == ns
    return len(ns)


def assert_plain_order(tendencies: list[dict]) -> None:
    family_positions = [FAMILIES.index(tendency["family"]) for tendency in tendencies]
    assert family_positions == sorted(family_positions)
    for family in FAMILIES:
        family_entries = [tendency for tendency in tendencies if tendency["family"] == family]
        conditions = list(dict.fromkeys(tendency["condition"] for tendency in family_entries))
        assert len(conditions) == len({tendency["condition"] for tendency in family_entries})
        if family == "zone_mix":
            assert conditions == [zone for zone in ZONE_NAMES if zone in conditions]
        if family == "response":
            parts = [condition.split("|") for condition in conditions]
            assert all(len(part) == 3 for part in parts)
            keys = [(GROUP_ORDER.index(group), ZONE_NAMES.index(hit_zone), ZONE_NAMES.index(landing_zone)) for group, hit_zone, landing_zone in parts]
            assert keys == sorted(keys) and len(set(keys)) == len(keys)
        for condition in conditions:
            choices = [(-tendency["k"], tendency["outcome"]) for tendency in family_entries if tendency["condition"] == condition]
            assert choices == sorted(choices)


def test_every_choice_is_kept_and_the_ks_of_a_situation_add_up_to_its_n():
    observations = pd.DataFrame(
        observation_rows("my_match", "me", 1, "rear_backhand", {"drop": 1})
        + observation_rows("my_match", "me", 1, "front_forehand", {"drop": 14, "clear": 3, "lift": 1})
        + observation_rows("my_match", "me", 1, "net|front_forehand|front_backhand", {"lift cross": 4, "lift straight": 3, "net straight": 1, "drive": 1}, family="response")
    )
    tendencies = tendencies_from_observations(observations, "my_match", "me")
    assert [(tendency["id"], tendency["k"], tendency["n"]) for tendency in tendencies] == [
        ("zone_mix:front_forehand:drop", 14, 18),
        ("zone_mix:front_forehand:clear", 3, 18),
        ("zone_mix:front_forehand:lift", 1, 18),
        ("zone_mix:rear_backhand:drop", 1, 1),
        ("response:net|front_forehand|front_backhand:lift cross", 4, 9),
        ("response:net|front_forehand|front_backhand:lift straight", 3, 9),
        ("response:net|front_forehand|front_backhand:drive", 1, 9),
        ("response:net|front_forehand|front_backhand:net straight", 1, 9),
    ]
    assert assert_choices_add_up_to_n(tendencies) == 3
    assert tendencies[4]["text"] == "After a net shot from the other player's front forehand to the front backhand: lifts cross 4 of 9 times."
    assert tendencies[6]["text"] == "After a net shot from the other player's front forehand to the front backhand: drives 1 of 9 times."
    assert tendencies[3]["text"] == "From the back court, backhand side: drops 1 of 1 times."


def test_order_is_family_then_situation_then_most_times_first_not_a_ranking():
    rows = []
    for family, condition, counts in (
        ("error_reason", "error", {"out": 2, "net": 2, "foul": 5}),
        ("winning_shot", "winner", {"smash": 1, "drop": 4}),
        ("serve_placement", "serve", {"front_middle": 3, "rear_backhand": 9}),
        ("response", "drive|front_forehand|mid_middle", {"drive cross": 30}),
        ("response", "serve|mid_middle|rear_backhand", {"net straight": 1, "lift cross": 1}),
        ("response", "serve|mid_middle|front_forehand", {"clear straight": 1}),
        ("response", "clear|rear_backhand|front_forehand", {"drop cross": 2, "clear straight": 6}),
        ("response", "clear|mid_forehand|front_forehand", {"smash straight": 1}),
        ("zone_mix", "rear_backhand", {"clear": 50}),
        ("zone_mix", "front_forehand", {"net": 1}),
        ("zone_mix", "mid_middle", {"drive": 2, "smash": 2}),
        ("error_shot", "error", {"net": 3}),
    ):
        rows += observation_rows("m", "A", 1, condition, counts, family=family)
    tendencies = tendencies_from_observations(pd.DataFrame(rows), "m", "A")
    assert [tendency["id"] for tendency in tendencies] == [
        "zone_mix:front_forehand:net",
        "zone_mix:mid_middle:drive",
        "zone_mix:mid_middle:smash",
        "zone_mix:rear_backhand:clear",
        "response:serve|mid_middle|front_forehand:clear straight",
        "response:serve|mid_middle|rear_backhand:lift cross",
        "response:serve|mid_middle|rear_backhand:net straight",
        "response:clear|mid_forehand|front_forehand:smash straight",
        "response:clear|rear_backhand|front_forehand:clear straight",
        "response:clear|rear_backhand|front_forehand:drop cross",
        "response:drive|front_forehand|mid_middle:drive cross",
        "serve_placement:serve:rear_backhand",
        "serve_placement:serve:front_middle",
        "winning_shot:winner:drop",
        "winning_shot:winner:smash",
        "error_shot:error:net",
        "error_reason:error:foul",
        "error_reason:error:net",
        "error_reason:error:out",
    ]
    assert_plain_order(tendencies)
    shuffled = tendencies_from_observations(pd.DataFrame(rows).sample(frac=1.0, random_state=3), "m", "A")
    assert shuffled == tendencies


def test_every_entry_has_exactly_the_documented_fields_and_no_pros_wording():
    observations = pd.DataFrame(
        observation_rows("my_match", "me", 1, "front_forehand", {"drop": 14, "clear": 4})
        + observation_rows("my_match", "me", 1, "serve", {"rear_backhand": 2, "not_over_net": 1}, family="serve_placement")
        + observation_rows("my_match", "me", 1, "error", {"out": 1, "misjudged landing": 1, "something new": 1}, family="error_reason")
    )
    tendencies = tendencies_from_observations(observations, "my_match", "me")
    assert len(tendencies) == 7
    for tendency in tendencies:
        assert set(tendency) == TENDENCY_KEYS
        assert tendency["rate"] == tendency["k"] / tendency["n"]
        assert f"{tendency['k']} of {tendency['n']}" in tendency["text"] and not re.search(PROS_PATTERN, tendency["text"])
        assert tendency["id"] == f"{tendency['family']}:{tendency['condition']}:{tendency['outcome']}"
        assert (tendency["match_id"], tendency["player"], tendency["sets"]) == ("my_match", "me", [1])
    json.dumps(tendencies)
    assert "Serves into the net 1 of 3 times." in [tendency["text"] for tendency in tendencies]
    assert "Errors something new: 1 of 3 errors." in [tendency["text"] for tendency in tendencies]
    with pytest.raises(ValueError):
        tendency_text("made_up_family", "x", "y", 1, 2)


def test_removed_names_are_gone_from_the_module():
    for name in REMOVED_NAMES:
        assert not hasattr(tendencies_module, name), name
    source = (PROJECT_ROOT / "analytics" / "tendencies.py").read_text()
    assert not re.search(r"(?i)reference|wilson|score|pros?\b|min_sample", source)


def test_only_the_players_own_observations_count():
    mine = observation_rows("my_match", "me", 1, "front_forehand", {"drop": 14, "clear": 4})
    noise = (
        observation_rows("my_match", "opp", 1, "front_forehand", {"drop": 500})
        + observation_rows("other_own", "me", 1, "front_forehand", {"lift": 500}, source="own")
        + observation_rows("pro_1", "A", 1, "front_forehand", {"smash": 500})
    )
    plain = tendencies_from_observations(pd.DataFrame(mine), "my_match", "me")
    noisy = tendencies_from_observations(pd.DataFrame(mine + noise), "my_match", "me")
    assert plain == noisy and [(tendency["k"], tendency["n"]) for tendency in plain] == [(14, 18), (4, 18)]
    assert tendencies_from_observations(pd.DataFrame(noise), "my_match", "me") == []


def test_sets_selection_counts_only_those_sets():
    set_one = observation_rows("my_match", "me", 1, "front_forehand", {"drop": 14, "clear": 4})
    both = set_one + observation_rows("my_match", "me", 2, "front_forehand", {"clear": 40}) + observation_rows("my_match", "me", 2, "rear_middle", {"smash": 3})
    assert tendencies_from_observations(pd.DataFrame(set_one), "my_match", "me", sets=(1,)) == tendencies_from_observations(pd.DataFrame(both), "my_match", "me", sets=(1,))
    whole = tendencies_from_observations(pd.DataFrame(both), "my_match", "me")
    assert [(tendency["id"], tendency["k"], tendency["n"]) for tendency in whole] == [
        ("zone_mix:front_forehand:clear", 44, 58), ("zone_mix:front_forehand:drop", 14, 58), ("zone_mix:rear_middle:smash", 3, 3)]
    assert {tuple(tendency["sets"]) for tendency in whole} == {(1, 2)}
    assert tendencies_from_observations(pd.DataFrame(both), "my_match", "me", sets=(3,)) == []


def test_label_source_is_per_situation_over_its_n_shots():
    observations = pd.DataFrame(
        observation_rows("my_match", "me", 1, "front_forehand", {"drop": 14, "clear": 4}, source="own", group_source="prediction")
        + observation_rows("my_match", "me", 1, "rear_backhand", {"drop": 10}, source="own", group_source="prediction")
        + observation_rows("my_match", "me", 1, "rear_backhand", {"clear": 1}, source="own", group_source="label")
        + observation_rows("my_match", "me", 1, "mid_middle", {"drive": 2}, source="own", group_source="label")
    )
    sources = {tendency["id"]: tendency["label_source"] for tendency in tendencies_from_observations(observations, "my_match", "me")}
    assert sources == {
        "zone_mix:front_forehand:drop": "predictions",
        "zone_mix:front_forehand:clear": "predictions",
        "zone_mix:mid_middle:drive": "labels",
        "zone_mix:rear_backhand:drop": "mixed",
        "zone_mix:rear_backhand:clear": "mixed",
    }


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


def test_own_rows_with_me_and_opp_end_to_end_with_no_shuttleset_rows(tmp_path):
    rows = []
    for rally in range(1, 11):
        rows.append(shot_row("own", "my_game", 1, rally, 1, "opp", type_pred="serve", hit_zone="mid_middle", landing_zone="front_middle" if rally <= 9 else "rear_backhand"))
        rows.append(shot_row("own", "my_game", 1, rally, 2, "me", type_pred="lift" if rally <= 7 else "net", hit_zone="front_forehand", is_cross=rally % 2 == 0 if rally <= 8 else None,
                             getpoint_player="opp", lose_reason="out"))
    database_path = write_shots_database(tmp_path / "scout.duckdb", rows)
    mine = tendencies_for("my_game", "me", database_path)
    assert [(tendency["id"], tendency["k"], tendency["n"]) for tendency in mine] == [
        ("zone_mix:front_forehand:lift", 7, 10), ("zone_mix:front_forehand:net", 3, 10),
        ("response:serve|mid_middle|front_middle:lift straight", 4, 9), ("response:serve|mid_middle|front_middle:lift cross", 3, 9),
        ("response:serve|mid_middle|front_middle:net", 1, 9), ("response:serve|mid_middle|front_middle:net cross", 1, 9),
        ("response:serve|mid_middle|rear_backhand:net", 1, 1),
        ("error_shot:error:lift", 7, 10), ("error_shot:error:net", 3, 10),
        ("error_reason:error:out", 10, 10),
    ]
    assert {tendency["label_source"] for tendency in mine if tendency["family"] != "error_reason"} == {"predictions"}
    theirs = tendencies_for("my_game", "opp", database_path)
    assert [tendency["text"] for tendency in theirs] == ["Serves short down the middle 9 of 10 times.", "Serves long to the receiver's backhand 1 of 10 times."]
    with pytest.raises(UnknownMatchPlayerError):
        tendencies_for("my_game", "nobody", database_path)


def reply_shot_rows(previous: dict, reply: dict, rally: int) -> list[dict]:
    return [
        shot_row("shuttleset", "m", 1, rally, 1, "A", type_en="clear", type_group="clear", **previous),
        shot_row("shuttleset", "m", 1, rally, 2, "B", type_en="drop", type_group="drop", hit_zone="rear_forehand", **reply),
    ]


def test_replies_count_only_previous_shots_by_the_other_player_with_a_type_and_both_zones_over_the_net(tmp_path):
    counted = {"hit_zone": "rear_backhand", "landing_zone": "rear_forehand"}
    rows = (
        reply_shot_rows(counted, {"is_cross": True}, 1)
        + reply_shot_rows(counted, {"is_cross": False}, 2)
        + reply_shot_rows(counted, {}, 3)
        + reply_shot_rows({"hit_zone": "rear_backhand", "landing_zone": NOT_OVER_NET_ZONE}, {"is_cross": True}, 4)
        + reply_shot_rows({"hit_zone": None, "landing_zone": "rear_forehand"}, {"is_cross": True}, 5)
        + reply_shot_rows({"hit_zone": "rear_backhand", "landing_zone": None}, {"is_cross": True}, 6)
        + [shot_row("shuttleset", "m", 1, 7, 1, "A", type_en="unknown", type_group="unknown", hit_zone="rear_backhand", landing_zone="rear_forehand"),
           shot_row("shuttleset", "m", 1, 7, 2, "B", type_en="drop", type_group="drop", hit_zone="rear_forehand", is_cross=True)]
        + [shot_row("shuttleset", "m", 1, 8, 1, "B", type_en="clear", type_group="clear", hit_zone="rear_backhand", landing_zone="rear_forehand"),
           shot_row("shuttleset", "m", 1, 8, 2, "B", type_en="drop", type_group="drop", hit_zone="rear_forehand", is_cross=True)]
    )
    connection = duckdb.connect(str(write_shots_database(tmp_path / "scout.duckdb", rows)), read_only=True)
    observations = family_observations(reliable_shots(load_tendency_shots(connection)))
    connection.close()
    replies = observations[observations["family"] == "response"]
    assert sorted(zip(replies["player"], replies["condition"], replies["outcome"])) == [
        ("B", "clear|rear_backhand|rear_forehand", "drop"),
        ("B", "clear|rear_backhand|rear_forehand", "drop cross"),
        ("B", "clear|rear_backhand|rear_forehand", "drop straight"),
    ]
    tendencies = tendencies_from_observations(observations, "m", "B")
    assert [(tendency["outcome"], tendency["k"], tendency["n"]) for tendency in tendencies if tendency["family"] == "response"] == [
        ("drop", 1, 3), ("drop cross", 1, 3), ("drop straight", 1, 3)]


def test_cli_prints_every_entry_and_exits_one_or_two_on_bad_input(tmp_path):
    rows = [shot_row("own", "my_game", 1, rally, 1, "me", type_pred="serve", landing_zone="front_middle" if rally < 3 else "not_over_net") for rally in range(1, 4)]
    database_path = write_shots_database(tmp_path / "scout.duckdb", rows)
    printed = subprocess.run([sys.executable, "-m", "analytics.tendencies", "my_game", "me", "", str(database_path)], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert printed.returncode == 0 and json.loads(printed.stdout) == tendencies_for("my_game", "me", database_path)
    assert [entry["text"] for entry in json.loads(printed.stdout)] == ["Serves short down the middle 2 of 3 times.", "Serves into the net 1 of 3 times."]
    unknown = subprocess.run([sys.executable, "-m", "analytics.tendencies", "my_game", "nobody", "", str(database_path)], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert unknown.returncode == 1 and "no shots for player 'nobody'" in unknown.stderr
    missing = subprocess.run([sys.executable, "-m", "analytics.tendencies"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert missing.returncode == 2


@pytest.fixture(scope="module")
def real_entries():
    if not REAL_DATABASE_PATH.exists():
        pytest.skip("data/scout.duckdb not built")
    observations = all_observations(REAL_DATABASE_PATH)
    players = observations[["source", "match_id", "player"]].drop_duplicates()
    return observations, {(source, match_id, player): tendencies_from_observations(observations, match_id, player) for source, match_id, player in players.itertuples(index=False)}


def test_every_real_player_keeps_every_choice_in_plain_order(real_entries):
    observations, entries = real_entries
    assert Counter(source for source, _, _ in entries) == {"shuttleset": 88, "own": 2}
    for (source, match_id, player), tendencies in entries.items():
        assert tendencies, (match_id, player)
        assert_choices_add_up_to_n(tendencies)
        assert_plain_order(tendencies)
        player_observations = observations[(observations["match_id"] == match_id) & (observations["player"] == player)]
        assert len(tendencies) == len(player_observations[["family", "condition", "outcome"]].drop_duplicates())
        assert sum(tendency["k"] for tendency in tendencies) == len(player_observations)
        for tendency in tendencies:
            assert set(tendency) == TENDENCY_KEYS and not re.search(PROS_PATTERN, tendency["text"])
            if tendency["family"] == "response":
                group, hit_zone, landing_zone = tendency["condition"].split("|")
                assert group in GROUP_ORDER and hit_zone in ZONE_NAMES and landing_zone in ZONE_NAMES
                reply_group, _, direction = tendency["outcome"].partition(" ")
                assert reply_group in GROUP_ORDER and direction in ("straight", "cross", "")
                assert "from the other player's" in tendency["text"] and f"to the {landing_zone.replace('_', ' ')}:" in tendency["text"]


def test_results_doc_numbers_equal_the_entries(real_entries):
    _, entries = real_entries
    doc = RESULTS_PATH.read_text()
    flat = [tendency for tendencies in entries.values() for tendency in tendencies]
    situations = {(key, situation_key(tendency)) for key, tendencies in entries.items() for tendency in tendencies}
    shuttleset_sizes = sorted(len(tendencies) for (source, _, _), tendencies in entries.items() if source == "shuttleset")
    median = (shuttleset_sizes[43] + shuttleset_sizes[44]) / 2
    assert f"| ShuttleSet, whole match | {shuttleset_sizes[0]} to {shuttleset_sizes[-1]} (median {median:g}) |" in doc
    assert f"| `match_tracked`, `me` | {len(entries[('own', 'match_tracked', 'me')])} |" in doc
    assert f"| `match_tracked`, `opp` | {len(entries[('own', 'match_tracked', 'opp')])} |" in doc
    assert f"{len(flat):,} in total over {len(situations):,} situations" in doc
    assert f"checked for all {len(situations):,} situations ({len(situations):,} hold)" in doc
    assert f"{sum(tendency['k'] == 1 for tendency in flat):,} entries are a single shot" in doc
    families = Counter(tendency["family"] for tendency in flat)
    assert "Entries per family: " + ", ".join(f"{family} {families[family]:,}" for family in FAMILIES) + "." in doc
    shuttleset_sources = Counter(tendency["label_source"] for (source, _, _), tendencies in entries.items() if source == "shuttleset" for tendency in tendencies)
    own_sources = Counter(tendency["label_source"] for (source, _, _), tendencies in entries.items() if source == "own" for tendency in tendencies)
    assert set(own_sources) == {"predictions"}
    label_source_line = re.sub(r"\s+", " ", doc.split("- Label source: ", 1)[1].split("\n\n", 1)[0])
    for source_name, count in shuttleset_sources.items():
        assert f"{count:,} `{source_name}`" in label_source_line.split("my tracked match")[0], (source_name, count, label_source_line)
    assert f"my tracked match {own_sources['predictions']} `predictions` and 0 `mixed`" in label_source_line
    ginting_replies = [tendency for tendency in entries[("shuttleset", AXELSEN_GINTING, "B")] if tendency["family"] == "response"]
    most_frequent = max(ginting_replies, key=lambda tendency: tendency["n"])["condition"]
    in_situation = [tendency for tendency in ginting_replies if tendency["condition"] == most_frequent]
    for tendency in in_situation:
        assert f"- {tendency['text']}" in doc
    assert " + ".join(str(tendency["k"]) for tendency in in_situation) + f" = {in_situation[0]['n']}." in doc


def test_results_doc_reply_counts_equal_the_observations(real_entries):
    observations, _ = real_entries
    doc = re.sub(r"\s+", " ", RESULTS_PATH.read_text())
    replies = observations[observations["family"] == "response"]
    directions = Counter(outcome.partition(" ")[2] for outcome in replies["outcome"])
    assert f"{len(replies):,} counted replies, {directions['straight']:,} straight, {directions['cross']:,} cross and {directions['']} with no direction" in doc
    assert all(condition.count("|") == 2 and not condition.endswith(NOT_OVER_NET_ZONE) for condition in replies["condition"])


def test_validate_tendencies_is_gone_with_no_stale_references():
    assert importlib.util.find_spec("analytics.validate_tendencies") is None
    assert not (PROJECT_ROOT / "analytics" / "validate_tendencies.py").exists()
    assert not (PROJECT_ROOT / "docs" / "analytics" / "validate_tendencies.md").exists()
    assert not (PROJECT_ROOT / "data" / "checks" / "tendencies" / "set_one_to_set_two.json").exists()
    tracked = subprocess.run(["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.split()
    stale = []
    for name in tracked:
        path = PROJECT_ROOT / name
        if name.startswith("tests/") or not path.is_file() or path.suffix not in (".py", ".md", ".sh", ".sql", ".txt"):
            continue
        if re.search(r"validate_tendencies|set_one_to_set_two|matched.chance", path.read_text(encoding="utf-8", errors="ignore")):
            stale.append(name)
    assert stale == []
    assert not re.search(r"(?i)set one|set 1 -> set 2|pros", (PROJECT_ROOT / "docs" / "analytics" / "README.md").read_text())


def test_tendency_source_has_no_comments_or_docstrings():
    source_text = (PROJECT_ROOT / "analytics" / "tendencies.py").read_text()
    module = ast.parse(source_text)
    assert ast.get_docstring(module) is None
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node) is None
    assert [line for line in source_text.splitlines() if line.lstrip().startswith("#")] == []


@pytest.mark.parametrize("doc_name", ["analytics/tendencies.md", "results/tendencies.md"])
def test_tendency_docs_have_no_task_ids_and_no_pros_comparison(doc_name):
    text = (PROJECT_ROOT / "docs" / doc_name).read_text()
    assert re.findall(TASK_ID_PATTERN, text) == []
    assert not re.search(r"reference_rate|wilson|`score`|MIN_SAMPLE_SIZE|MIN_SCORE", text)
