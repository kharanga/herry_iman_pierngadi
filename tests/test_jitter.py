from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shots.features import load_shot_feature_rows, shot_features
from shots.jitter import JITTER_OFFSETS_FRAMES, JITTER_SEED, jitter_hit_frames, jitter_label, rally_order_index, uniform_frame_offsets
from shots.train import jittered_feature_frames, labelled_rows_of
from tests.test_shot_classifier import synthetic_shot_rows, write_shots_database

TIMING_FEATURES = ["flight_seconds", "seconds_since_previous_hit"]


def feature_rows(tmp_path: Path, match_count: int = 2, shots_per_group: int = 3) -> pd.DataFrame:
    import duckdb

    database_path = write_shots_database(tmp_path / "scout.duckdb", synthetic_shot_rows(match_count, shots_per_group))
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        return load_shot_feature_rows(connection)
    finally:
        connection.close()


def test_offsets_are_integers_within_the_bound_seeded_and_cover_both_ends():
    offsets = uniform_frame_offsets(20000, 5, random_seed=0)
    assert np.issubdtype(offsets.dtype, np.integer)
    assert offsets.min() == -5 and offsets.max() == 5
    assert set(np.unique(offsets)) == set(range(-5, 6))
    assert (uniform_frame_offsets(20000, 5, random_seed=0) == offsets).all()
    assert not (uniform_frame_offsets(20000, 5, random_seed=1) == offsets).all()
    assert (uniform_frame_offsets(50, 0) == 0).all()
    assert JITTER_OFFSETS_FRAMES == (5, 10) and JITTER_SEED == 0
    assert jitter_label(5) == "hit frames +/- 5"


def test_zero_offset_reproduces_the_sql_neighbour_frames_on_the_whole_table(tmp_path):
    shot_rows = feature_rows(tmp_path)
    jittered = jitter_hit_frames(shot_rows, 0)
    for column in ("frame_num", "previous_frame_num", "next_frame_num"):
        np.testing.assert_array_equal(jittered[column].to_numpy(float), shot_rows[column].to_numpy(float))
    pd.testing.assert_frame_equal(shot_features(jittered), shot_features(shot_rows))


def test_jitter_shifts_every_frame_and_recomputes_neighbours_within_the_rally_only(tmp_path):
    shot_rows = feature_rows(tmp_path)
    before = shot_rows.copy()
    jittered = jitter_hit_frames(shot_rows, 10, random_seed=3)
    pd.testing.assert_frame_equal(shot_rows, before)
    assert jittered.index.equals(shot_rows.index)
    shifts = jittered["frame_num"] - shot_rows["frame_num"]
    assert shifts.abs().le(10).all() and (shifts != 0).any()
    np.testing.assert_array_equal(shifts.to_numpy(), uniform_frame_offsets(len(shot_rows), 10, random_seed=3))
    ordered = jittered.loc[rally_order_index(shot_rows)]
    for _, rally_rows in ordered.groupby(["source", "match_id", "set_no", "rally"], sort=False):
        frames = rally_rows["frame_num"].tolist()
        assert rally_rows["previous_frame_num"].tolist()[1:] == frames[:-1] and np.isnan(rally_rows["previous_frame_num"].iloc[0])
        assert rally_rows["next_frame_num"].tolist()[:-1] == frames[1:] and np.isnan(rally_rows["next_frame_num"].iloc[-1])
    untouched = [column for column in shot_rows.columns if column not in ("frame_num", "previous_frame_num", "next_frame_num")]
    pd.testing.assert_frame_equal(jittered[untouched], shot_rows[untouched])


def test_jitter_changes_only_the_timing_features(tmp_path):
    shot_rows = feature_rows(tmp_path)
    exact = shot_features(shot_rows)
    jittered = shot_features(jitter_hit_frames(shot_rows, 10, random_seed=4))
    position_features = [column for column in exact.columns if column not in TIMING_FEATURES]
    pd.testing.assert_frame_equal(exact[position_features], jittered[position_features])
    assert not np.allclose(exact["flight_seconds"].fillna(-1), jittered["flight_seconds"].fillna(-1))


def test_jittered_feature_frames_line_up_with_the_labelled_rows(tmp_path):
    shot_rows = feature_rows(tmp_path)
    labelled_rows = labelled_rows_of(shot_rows, "feet")
    frames = jittered_feature_frames(shot_rows, labelled_rows, "feet")
    assert list(frames) == [jitter_label(offset) for offset in JITTER_OFFSETS_FRAMES]
    exact = shot_features(labelled_rows, "feet")
    for jittered in frames.values():
        assert jittered.index.equals(exact.index) and list(jittered.columns) == list(exact.columns)
        pd.testing.assert_frame_equal(jittered.drop(columns=TIMING_FEATURES), exact.drop(columns=TIMING_FEATURES))
        assert jittered["flight_seconds"].notna().all()


def test_jitter_needs_the_rally_columns():
    with pytest.raises(KeyError):
        jitter_hit_frames(pd.DataFrame({"frame_num": [1, 2]}), 5)
