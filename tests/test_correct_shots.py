import pandas as pd

from vision.label_scores import ScoredRun, correct_shot_table, correct_shots, report_lines

NAN = float("nan")


def labels():
    return pd.DataFrame({
        "rally": [1, 1, 1, 1],
        "frame": [100, 140, 180, 220],
        "hitter": ["me", "opp", "me", "opp"],
        "type": ["serve", "clear", "smash", "net"],
        "detail": ["serve_short", "clear", "smash", "net"],
        "source": ["hand"] * 4,
    })


def run(rows, predicted):
    hits = pd.DataFrame(rows, columns=["rally", "frame", "hitter", "score", "reach"])
    return ScoredRun("m", hits, pd.DataFrame(predicted, columns=["frame_num", "type_pred"]))


SPANS = [(100, 220)]


def test_a_missed_hit_a_wrong_type_and_a_false_hit_inside_each_cost_one_and_serves_count():
    perfect = run([(1, 100, "me", NAN, NAN), (1, 140, "opp", 0.9, NAN), (1, 180, "me", 0.9, NAN), (1, 220, "opp", 0.9, NAN)], [(100, "serve"), (140, "clear"), (180, "smash"), (220, "net")])
    assert correct_shots(perfect, labels(), SPANS, 5) == (4, 4)
    missed = run([(1, 100, "me", NAN, NAN), (1, 140, "opp", 0.9, NAN), (1, 180, "me", 0.9, NAN)], [(100, "serve"), (140, "clear"), (180, "smash")])
    assert correct_shots(missed, labels(), SPANS, 5) == (3, 4)
    wrong_type = run([(1, 100, "me", NAN, NAN), (1, 140, "opp", 0.9, NAN), (1, 180, "me", 0.9, NAN), (1, 220, "opp", 0.9, NAN)], [(100, "serve"), (140, "drop"), (180, "smash"), (220, "net")])
    assert correct_shots(wrong_type, labels(), SPANS, 5) == (3, 4)
    false_inside = run([(1, 100, "me", NAN, NAN), (1, 140, "opp", 0.9, NAN), (1, 160, "me", 0.9, NAN), (1, 180, "me", 0.9, NAN), (1, 220, "opp", 0.9, NAN)], [(100, "serve"), (140, "clear"), (160, "lift"), (180, "smash"), (220, "net")])
    assert correct_shots(false_inside, labels(), SPANS, 5) == (4, 5)
    false_outside = run([(1, 100, "me", NAN, NAN), (1, 140, "opp", 0.9, NAN), (1, 180, "me", 0.9, NAN), (1, 220, "opp", 0.9, NAN), (2, 900, "me", 0.9, NAN)], [(100, "serve"), (140, "clear"), (180, "smash"), (220, "net"), (900, "lift")])
    assert correct_shots(false_outside, labels(), SPANS, 5) == (4, 4)
    no_serve_row = run([(1, 140, "opp", 0.9, NAN), (1, 180, "me", 0.9, NAN), (1, 220, "opp", 0.9, NAN)], [(140, "clear"), (180, "smash"), (220, "net")])
    assert correct_shots(no_serve_row, labels(), SPANS, 5) == (3, 4)


def test_the_table_line_and_its_place_after_the_headline():
    perfect = run([(1, 100, "me", NAN, NAN), (1, 140, "opp", 0.9, NAN), (1, 160, "me", 0.9, NAN), (1, 180, "me", 0.9, NAN), (1, 220, "opp", 0.9, NAN)], [(100, "serve"), (140, "clear"), (160, "lift"), (180, "smash"), (220, "net")])
    assert correct_shot_table([perfect], labels(), SPANS, [(0.25, 5)], 20.0)[2] == "| m | ±0.25 s (±5 frames at 20 fps) | 4/5 = 80.0% | 4 | 1 |"
    lines = report_lines([perfect], labels(), SPANS, 20.0, (0.25,), 0.25)
    headings = [line for line in lines if line.startswith("## ")]
    assert headings[0] == "## Headline" and headings[1].startswith("## Correct shots:")
