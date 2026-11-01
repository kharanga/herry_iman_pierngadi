SHOT_TYPE_GROUPS = ("serve", "clear", "drop", "smash", "net", "lift", "drive", "unknown")

TYPE_GROUP_BY_TYPE_EN = {
    "short service": "serve",
    "long service": "serve",
    "clear": "clear",
    "drop": "drop",
    "passive drop": "drop",
    "smash": "smash",
    "wrist smash": "smash",
    "net shot": "net",
    "return net": "net",
    "cross-court net shot": "net",
    "rush": "net",
    "lob": "lift",
    "defensive return lob": "lift",
    "drive": "drive",
    "driven flight": "drive",
    "back-court drive": "drive",
    "push": "drive",
    "defensive return drive": "drive",
    "unknown": "unknown",
}

SHOT_TYPE_DETAILS = ("serve_short", "serve_long", "net", "return_net", "drive", "push", "clear", "drop", "smash", "lift", "unknown")

TYPE_GROUP_BY_TYPE_DETAIL = {
    "serve_short": "serve",
    "serve_long": "serve",
    "net": "net",
    "return_net": "net",
    "drive": "drive",
    "push": "drive",
    "clear": "clear",
    "drop": "drop",
    "smash": "smash",
    "lift": "lift",
    "unknown": "unknown",
}

TYPE_DETAIL_BY_SPLIT_OFF_TYPE_EN = {
    "short service": "serve_short",
    "long service": "serve_long",
    "return net": "return_net",
    "push": "push",
}


class UnmappedShotTypeError(ValueError):
    def __init__(self, type_en: str):
        super().__init__(f"shot type has no group: {type_en!r}")
        self.type_en = type_en


def type_group_for(type_en: str | None) -> str | None:
    if type_en is None:
        return None
    if type_en not in TYPE_GROUP_BY_TYPE_EN:
        raise UnmappedShotTypeError(type_en)
    return TYPE_GROUP_BY_TYPE_EN[type_en]


def type_detail_for(type_en: str | None) -> str | None:
    group = type_group_for(type_en)
    if group is None:
        return None
    return TYPE_DETAIL_BY_SPLIT_OFF_TYPE_EN.get(type_en, group)


def group_of_detail(type_detail: str | None) -> str | None:
    if type_detail is None or type_detail != type_detail:
        return None
    return TYPE_GROUP_BY_TYPE_DETAIL[type_detail]
