import numpy as np
import pandas as pd
import pytest

import analytics.court as court
from analytics.court import NET_Y_M
from vision.zone_split import OLD_DEPTH_EDGES_M, changed_counts, depth_edges, habit_change_lines, text_or_none, zones_of


def test_depth_edges_restores_the_constant_also_after_an_error():
    current = court.DEPTH_EDGES_M
    with depth_edges((1.0, 2.0)):
        assert court.DEPTH_EDGES_M == (1.0, 2.0)
    assert court.DEPTH_EDGES_M == current
    with pytest.raises(RuntimeError):
        with depth_edges((1.0, 2.0)):
            raise RuntimeError("stop")
    assert court.DEPTH_EDGES_M == current


def test_zones_under_both_edges_for_hits_at_2_3_and_3_8_m_from_the_net():
    rows = pd.DataFrame({"hit_x": [4.0, 4.0, 4.0, np.nan], "hit_y": [NET_Y_M - 2.0, NET_Y_M - 3.0, NET_Y_M - 3.8, 1.0], "landing_x": [2.0, 2.0, 2.0, 2.0], "landing_y": [NET_Y_M + 2.0, NET_Y_M + 3.0, NET_Y_M + 3.8, 5.0]})
    old = zones_of(rows, OLD_DEPTH_EDGES_M)
    new = zones_of(rows, (2.75, 4.00))
    assert old["hit_zone"].tolist()[:3] == ["mid_forehand", "mid_forehand", "rear_forehand"] and pd.isna(old["hit_zone"].iloc[3])
    assert new["hit_zone"].tolist()[:3] == ["front_forehand", "mid_forehand", "mid_forehand"]
    assert old["landing_zone"].tolist()[3] == "not_over_net"
    assert [zone.split("_")[0] for zone in new["landing_zone"][:3]] == ["front", "mid", "mid"]
    assert court.DEPTH_EDGES_M == (2.75, 4.00)


def test_text_or_none_turns_nan_into_none():
    assert text_or_none(np.nan) is None and text_or_none(None) is None and text_or_none("front_forehand") == "front_forehand"


def test_changed_counts_judge_only_rows_with_a_zone_in_both():
    old = pd.DataFrame({"hit_zone": ["mid_forehand", "rear_backhand", None, "front_forehand"], "landing_zone": [None, None, None, "mid_backhand"]})
    new = pd.DataFrame({"hit_zone": ["front_forehand", "rear_backhand", "mid_backhand", None], "landing_zone": [None, "front_backhand", None, "front_backhand"]})
    assert changed_counts(old, new) == {"hit_zone": (1, 2), "landing_zone": (1, 1)}


def habit(habit_id, k, n):
    return {"id": habit_id, "k": k, "n": n}


def test_habit_lines_list_zone_and_serve_habits_and_count_the_other_families():
    old = {key: habit(key, k, n) for key, k, n in [("zone_mix:mid_forehand:lift", 3, 5), ("serve_placement:serve_short:mid_backhand", 2, 2), ("response:a:lift", 1, 3), ("response:b:net", 2, 2)]}
    new = {key: habit(key, k, n) for key, k, n in [("zone_mix:front_forehand:lift", 3, 5), ("serve_placement:serve_short:front_backhand", 2, 2), ("response:a:lift", 1, 3), ("response:c:net", 2, 2)]}
    lines = habit_change_lines(old, new)
    assert "| zone_mix: mid_forehand:lift | 3 of 5 | - |" in lines and "| zone_mix: front_forehand:lift | - | 3 of 5 |" in lines
    assert "| serve_placement: serve_short:front_backhand | - | 2 of 2 |" in lines
    assert "| response (not listed) | 2 of 3 habits change | |" in lines
    assert not any(line.startswith("| response:") for line in lines)
