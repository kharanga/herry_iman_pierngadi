SHOT_TYPE_ENGLISH_BY_CHINESE = {
    "放小球": "net shot",
    "擋小球": "return net",
    "殺球": "smash",
    "點扣": "wrist smash",
    "挑球": "lob",
    "防守回挑": "defensive return lob",
    "長球": "clear",
    "平球": "drive",
    "小平球": "driven flight",
    "後場抽平球": "back-court drive",
    "切球": "drop",
    "過度切球": "passive drop",
    "過渡切球": "passive drop",
    "推球": "push",
    "撲球": "rush",
    "防守回抽": "defensive return drive",
    "勾球": "cross-court net shot",
    "發短球": "short service",
    "發長球": "long service",
    "未知球種": "unknown",
}

LOSE_REASON_ENGLISH_BY_CHINESE = {
    "對手落地致勝": "opponent winner",
    "出界": "out",
    "掛網": "net",
    "未過網": "not over net",
    "落點判斷失誤": "misjudged landing",
    "對手落地判斷失誤": "misjudged opponent landing",
    "犯規": "foul",
}

WIN_REASON_ENGLISH_BY_CHINESE = {
    "落地致勝": "winner",
    "對手出界": "opponent out",
    "對手掛網": "opponent net",
    "對手未過網": "opponent not over net",
    "對手落點判斷失誤": "opponent misjudged landing",
    "落地判斷失誤": "landed, opponent misjudged",
    "對手犯規": "opponent foul",
}

ENGLISH_BY_CHINESE_BY_RAW_SHOTS_COLUMN = {
    "type": SHOT_TYPE_ENGLISH_BY_CHINESE,
    "lose_reason": LOSE_REASON_ENGLISH_BY_CHINESE,
    "win_reason": WIN_REASON_ENGLISH_BY_CHINESE,
}


class UnmappedLabelError(ValueError):
    def __init__(self, column_name: str, label: str):
        super().__init__(f"raw_shots.{column_name} has a label with no English translation: {label!r}")
        self.column_name = column_name
        self.label = label


def english_label_by_source_label(column_name: str) -> dict[str, str]:
    english_by_chinese = ENGLISH_BY_CHINESE_BY_RAW_SHOTS_COLUMN[column_name]
    already_english = {english: english for english in english_by_chinese.values()}
    return {**already_english, **english_by_chinese}


def translate_label(column_name: str, label: str | None) -> str | None:
    if label is None:
        return None
    english_by_source = english_label_by_source_label(column_name)
    if label not in english_by_source:
        raise UnmappedLabelError(column_name, label)
    return english_by_source[label]
