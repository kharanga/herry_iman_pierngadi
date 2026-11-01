import duckdb
import numpy as np
import pandas as pd
import pytest

from vision.label_scores import (
    detail_lines,
    ScoredRun,
    agreement_counts,
    counts_by,
    headline_table,
    label_outcomes,
    labelled_rally_spans,
    main,
    missed_table,
    nearest_offset,
    precision_in_labelled_rallies,
    rallies_file_for,
    read_labels,
    read_predictions,
    serve_table,
    share,
    source_lines,
)


def labels_table():
    return pd.DataFrame(
        {
            "rally": [1, 1, 1, 1, 2, 2],
            "frame": [100, 130, 160, 190, 500, 540],
            "hitter": ["me", "opp", "me", np.nan, "me", "opp"],
            "type": ["serve", "net", "lift", "unknown", "serve", "clear"],
            "source": "hand",
        }
    )


def hits_table():
    return pd.DataFrame(
        {
            "rally": [1, 1, 1, 1, 1, 2, 2],
            "frame": [97, 133, 158, 175, 300, 470, 545],
            "hitter": ["me", "me", "me", "opp", "opp", "me", "opp"],
            "score": [np.nan, 0.9, 0.8, 0.7, 0.6, np.nan, 0.9],
            "reach": np.nan,
        }
    )


def predictions_table():
    return pd.DataFrame({"frame_num": [97, 133, 158, 175, 470, 545], "type_pred": ["serve", "net", "drive", "drop", "serve", None]})


def test_read_labels_defaults_the_source_and_drops_unknown_players(tmp_path):
    labels_file = tmp_path / "game.csv"
    pd.DataFrame({"rally": [2, 1], "frame_num": [50, 10], "player": ["me", "ref"], "type_group": ["net", "serve"]}).to_csv(labels_file, index=False)
    labels = read_labels(labels_file)
    assert labels["frame"].tolist() == [10, 50] and labels["source"].tolist() == ["hand", "hand"]
    assert pd.isna(labels["hitter"].iloc[0]) and labels["hitter"].iloc[1] == "me"
    assert rallies_file_for(labels_file) == tmp_path / "game_rallies.csv"


def test_only_rallies_that_hold_labels_are_spans():
    rallies = pd.DataFrame({"rally": [3, 1, 2], "start_frame": [900, 90, 450], "end_frame": [990, 200, 560]})
    assert labelled_rally_spans(rallies, labels_table()) == [(90, 200), (450, 560)]


def test_outcomes_match_one_to_one_and_carry_hitter_serve_flag_and_prediction():
    outcomes = label_outcomes(hits_table(), labels_table(), predictions_table(), 5)
    assert outcomes["detected_frame"].tolist() == [pd.NA, 133, 158, pd.NA, pd.NA, 545]
    with_serves = label_outcomes(hits_table(), labels_table(), predictions_table(), 5, with_serves=True)
    assert with_serves["detected_frame"].tolist()[0] == 97 and bool(with_serves["inferred_serve"].iloc[0])
    assert with_serves["detected_hitter"].tolist()[:3] == ["me", "me", "me"] and with_serves["type_pred"].tolist()[:3] == ["serve", "net", "drive"]
    wide = label_outcomes(hits_table(), labels_table(), predictions_table(), 20)
    assert wide["detected_frame"].tolist() == [pd.NA, 133, 158, 175, pd.NA, 545]


def test_agreement_leaves_out_unknown_types_missing_predictions_and_missing_hitters():
    counts = agreement_counts(label_outcomes(hits_table(), labels_table(), predictions_table(), 20, with_serves=True))
    assert counts == {"labelled": 6, "matched": 5, "hitter_right": 3, "hitter_judged": 4, "type_right": 2, "type_judged": 3, "detail_right": 0, "detail_judged": 0}
    by_type = counts_by(label_outcomes(hits_table(), labels_table(), predictions_table(), 20, with_serves=True), "type")
    assert by_type["serve"]["matched"] == 1 and by_type["serve"]["labelled"] == 2 and by_type["unknown"]["type_judged"] == 0


def test_precision_counts_only_detections_inside_the_padded_labelled_rallies():
    spans = [(100, 190), (500, 540)]
    outcomes = label_outcomes(hits_table(), labels_table(), predictions_table(), 5)
    assert precision_in_labelled_rallies(hits_table(), outcomes, spans, 5) == (3, 4)
    with_serves = label_outcomes(hits_table(), labels_table(), predictions_table(), 5, with_serves=True)
    assert precision_in_labelled_rallies(hits_table(), with_serves, spans, 5, with_serves=True) == (4, 5)
    wide = label_outcomes(hits_table(), labels_table(), predictions_table(), 30, with_serves=True)
    assert precision_in_labelled_rallies(hits_table(), wide, spans, 30, with_serves=True) == (6, 6)


def test_nearest_offset_is_signed_and_ignores_inferred_serves():
    assert nearest_offset(100, hits_table()) == 33 and nearest_offset(500, hits_table()) == 45
    assert nearest_offset(10, hits_table().iloc[:0]) is None
    assert share(1, 3) == "1/3 = 33.3%" and share(0, 0) == "0/0"


def test_serve_table_splits_serves_by_inferred_row_and_detected_hit():
    run = ScoredRun("model", hits_table(), predictions_table())
    lines = serve_table([run], labels_table(), 5)
    assert lines[2] == "| model | 2 | 1 | 0 | 1 | 1/1 = 100.0% |"
    assert serve_table([run], labels_table(), 30)[2] == "| model | 2 | 2 | 0 | 0 | 2/2 = 100.0% |"


def test_headline_and_missed_tables():
    run = ScoredRun("model", hits_table(), predictions_table())
    headline = headline_table([run], labels_table(), [(100, 190), (500, 540)], [(0.25, 5)], 20.0)
    assert headline[2] == "| model | ±0.25 s (±5 frames at 20 fps) | detected hits | 3/6 = 50.0% | 2/3 = 66.7% | 1/2 = 50.0% | 3/4 = 75.0% |"
    assert headline[3] == "| model | ±0.25 s (±5 frames at 20 fps) | detected + inferred serves | 4/6 = 66.7% | 3/4 = 75.0% | 2/3 = 66.7% | 4/5 = 80.0% |"
    missed = missed_table([run], labels_table(), 5, True)
    assert missed[2:] == ["| 190 | 1 | nan | unknown | missed (nearest -15) |", "| 500 | 2 | me | serve | missed (nearest +45) |"]


def test_source_split_only_when_there_are_two_sources():
    run = ScoredRun("model", hits_table(), predictions_table())
    assert source_lines([run], labels_table(), [(100, 190)], [(0.25, 5)], 20.0)[-1] == "every label is hand; no per-source split"
    mixed = labels_table().assign(source=["hand", "accepted", "hand", "hand", "accepted", "hand"])
    lines = source_lines([run], mixed, [(100, 190)], [(0.25, 5)], 20.0)
    assert "## Recall of accepted labels" in lines and "## Recall of hand labels" in lines
    assert lines[lines.index("## Recall of accepted labels") + 3] == "| model | ±0.25 s (±5 frames at 20 fps) | detected hits | 1/2 = 50.0% | 0/1 = 0.0% | 1/1 = 100.0% | 1/3 = 33.3% |"


def write_inputs(tmp_path):
    labels_file = tmp_path / "game.csv"
    labels = labels_table()
    pd.DataFrame({"rally": labels["rally"], "frame_num": labels["frame"], "player": labels["hitter"].fillna("?"), "type_group": labels["type"], "source": labels["source"]}).to_csv(labels_file, index=False)
    pd.DataFrame({"rally": [1, 2], "start_frame": [100, 500], "end_frame": [190, 540]}).to_csv(tmp_path / "game_rallies.csv", index=False)
    hits_file = tmp_path / "hits.csv"
    hits_table().to_csv(hits_file, index=False)
    database = tmp_path / "scout.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute("CREATE TABLE shots (source VARCHAR, match_id VARCHAR, frame_num INTEGER, type_pred VARCHAR)")
    for frame_num, type_pred in predictions_table().itertuples(index=False):
        connection.execute("INSERT INTO shots VALUES ('own', 'club_1', ?, ?)", [int(frame_num), None if pd.isna(type_pred) else type_pred])
    connection.execute("INSERT INTO shots VALUES ('shuttleset', 'club_1', 133, 'smash'), ('own', 'other', 133, 'smash')")
    connection.close()
    return labels_file, hits_file, database


def test_predictions_are_the_own_rows_of_the_match_only(tmp_path):
    _, _, database = write_inputs(tmp_path)
    predictions = read_predictions(database, "club_1")
    assert sorted(predictions["frame_num"].tolist()) == [97, 133, 158, 175, 470, 545]
    assert "smash" not in predictions["type_pred"].tolist()


def test_main_prints_the_report_even_with_a_label_of_no_known_player(tmp_path, capsys):
    labels_file, hits_file, database = write_inputs(tmp_path)
    assert main(["--labels", str(labels_file), "--match-id", "club_1", "--run", "model", str(hits_file), str(database), "--tolerance", "0.25", "--breakdown-tolerance", "0.25", "--fps", "20"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "labels: 6 in 2 rallies; sources: hand 6"
    assert "| model | ±0.25 s (±5 frames at 20 fps) | detected hits | 3/6 = 50.0% | 2/3 = 66.7% | 1/2 = 50.0% | 3/4 = 75.0% |" in out
    assert "## Labelled serves, ±0.25 s (±5 frames at 20 fps)" in out and "| 500 | 2 | me | serve | missed (nearest +45) |" in out


def test_a_missing_database_is_an_error_and_is_not_created(tmp_path, capsys):
    labels_file, hits_file, _ = write_inputs(tmp_path)
    missing = tmp_path / "nowhere.duckdb"
    assert main(["--labels", str(labels_file), "--match-id", "club_1", "--run", "model", str(hits_file), str(missing), "--fps", "20"]) == 1
    assert "database not found" in capsys.readouterr().err and not missing.exists()
    assert main(["--labels", str(tmp_path / "none.csv"), "--match-id", "club_1", "--run", "model", str(hits_file), str(missing), "--fps", "20"]) == 1


def test_main_needs_a_run():
    with pytest.raises(SystemExit):
        main(["--labels", "x.csv", "--match-id", "club_1", "--fps", "20"])


def test_labels_with_no_type_get_their_own_breakdown_row(tmp_path, capsys):
    labels_file, hits_file, database = write_inputs(tmp_path)
    labels = pd.read_csv(labels_file)
    labels.loc[labels["frame_num"] == 160, "type_group"] = np.nan
    labels.to_csv(labels_file, index=False)
    assert main(["--labels", str(labels_file), "--match-id", "club_1", "--run", "model", str(hits_file), str(database), "--breakdown-tolerance", "0.25", "--fps", "20"]) == 0
    out = capsys.readouterr().out
    assert "| no type | 1 | 1/1 = 100.0% | 1/1 = 100.0% | 0/0 |" in out and "| unknown hitter | 1 |" in out


def detailed_labels():
    return labels_table().assign(detail=["serve_long", "return_net", np.nan, "unknown", "serve_short", "clear"])


def detailed_predictions():
    return predictions_table().assign(type_detail_pred=["serve_long", "net", "lift", "drop", "serve_short", None])


def test_detail_agreement_counts_only_labels_and_predictions_that_both_have_a_detail():
    outcomes = label_outcomes(hits_table(), detailed_labels(), detailed_predictions(), 20, with_serves=True)
    assert outcomes["detail_pred"].tolist()[:4] == ["serve_long", "net", "lift", "drop"]
    counts = agreement_counts(outcomes)
    assert (counts["detail_right"], counts["detail_judged"]) == (1, 2)
    assert (counts["type_right"], counts["type_judged"]) == (2, 3)


def test_detail_lines_say_why_detail_is_not_scored():
    run = ScoredRun("model", hits_table(), predictions_table())
    assert detail_lines([run], labels_table(), 20) == ["no label has a type_detail; detail agreement not scored"]
    assert detail_lines([run], detailed_labels(), 20)[2] == "| model | 5 | no detail predictions | |"
    detailed = ScoredRun("model", hits_table(), detailed_predictions())
    assert detail_lines([detailed], detailed_labels(), 20)[2] == "| model | 5 | 2 | 1/2 = 50.0% |"


def test_label_files_and_databases_with_details_are_read(tmp_path, capsys):
    labels_file, hits_file, database = write_inputs(tmp_path)
    labels = pd.read_csv(labels_file).assign(type_detail=["serve_long", "return_net", "", "unknown", "serve_short", "clear"])
    labels.to_csv(labels_file, index=False)
    read = read_labels(labels_file)
    assert read["detail"].isna().tolist() == [False, False, True, False, False, False]
    connection = duckdb.connect(str(database))
    connection.execute("ALTER TABLE shots ADD COLUMN type_detail_pred VARCHAR")
    connection.execute("UPDATE shots SET type_detail_pred = 'serve_long' WHERE shots.frame_num = 97")
    connection.execute("UPDATE shots SET type_detail_pred = 'net' WHERE shots.frame_num = 133")
    connection.close()
    predictions = read_predictions(database, "club_1")
    assert "type_detail_pred" in predictions.columns
    assert main(["--labels", str(labels_file), "--match-id", "club_1", "--run", "model", str(hits_file), str(database), "--breakdown-tolerance", "0.25", "--fps", "20"]) == 0
    assert "| model | 5 | 2 | 1/2 = 50.0% |" in capsys.readouterr().out


def test_rally_bounds_compare_each_runs_rallies_file_with_the_labelled_rallies(tmp_path, capsys):
    labels_file, hits_file, database = write_inputs(tmp_path)
    named_hits = tmp_path / "hits_0_600.csv"
    hits_file.rename(named_hits)
    pd.DataFrame({"rally": [1, 2, 3], "start_frame": [110, 480, 900], "end_frame": [200, 560, 950], "hits": [3, 2, 1]}).to_csv(tmp_path / "rallies_0_600.csv", index=False)
    assert main(["--labels", str(labels_file), "--match-id", "club_1", "--run", "model", str(named_hits), str(database), "--tolerance", "0.5", "--breakdown-tolerance", "0.5", "--fps", "10"]) == 0
    out = capsys.readouterr().out
    assert "## Rally bounds against the labelled rallies" in out
    assert "| model | 2 | 3 | 2 | 0 | 0 | 0 | 1 | 1.50 | 1.50 | -0.50 | +1.50 |" in out


def test_a_run_without_a_rallies_file_says_so_and_a_hits_file_is_never_read_as_its_own_rallies(tmp_path, capsys):
    labels_file, hits_file, database = write_inputs(tmp_path)
    assert main(["--labels", str(labels_file), "--match-id", "club_1", "--run", "model", str(hits_file), str(database), "--tolerance", "0.5", "--breakdown-tolerance", "0.5", "--fps", "10"]) == 0
    assert "| model | 2 | no rallies file | | | | | | | | | |" in capsys.readouterr().out


def test_fps_is_required():
    with pytest.raises(SystemExit):
        main(["--labels", "x.csv", "--match-id", "club_1", "--run", "model", "hits.csv", "db.duckdb"])
