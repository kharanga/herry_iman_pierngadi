import re

from vision.label_ui import PAGE_HTML, apply_action
from tests.test_label_segments import blank_state


def page_script():
    return re.search(r"<script>(.*)</script>", PAGE_HTML, re.S).group(1)


def test_the_header_has_the_frame_box_its_button_and_a_message():
    header = re.search(r"<header>(.*?)</header>", PAGE_HTML, re.S).group(1)
    assert 'id="framebox"' in header and 'id="gobutton"' in header and 'id="framemsg"' in header
    assert 'inputmode="numeric"' in header


def test_typing_in_the_box_never_reaches_the_shortcuts():
    script = page_script()
    box_handler = re.search(r'\$\("framebox"\)\.addEventListener\("keydown",e=>\{(.*?)\}\);', script, re.S).group(1)
    assert box_handler.startswith("e.stopPropagation();")
    page_handler = re.search(r'document\.addEventListener\("keydown",e=>\{(.*?)let k=', script, re.S).group(1)
    assert 'e.target.tagName==="INPUT"' in page_handler and 'e.target.tagName==="TEXTAREA"' in page_handler


def test_the_box_checks_the_text_before_sending_and_reuses_the_jump_action():
    function = re.search(r"function goToTypedFrame\(\)\{(.*?)\n\$\(\"gobutton\"\)", page_script(), re.S).group(1)
    assert "/^[0-9]+$/" in function and "state.last_frame" in function
    assert 'post({type:"jump",frame:frame})' in function
    assert function.index("type a whole frame number") < function.index("post(") and function.index("frames run 0-") < function.index("post(")


def test_the_jump_action_moves_the_segment_in_blank_mode():
    state = blank_state()
    moved = apply_action(state, {"type": "jump", "frame": 3000})
    assert moved.frame == 3000 and moved.view.start_frame <= 3000 <= moved.view.end_frame and moved.segment_index == 2
    clamped = apply_action(state, {"type": "jump", "frame": 10 ** 6})
    assert clamped.frame == state.last_frame
