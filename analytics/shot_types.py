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
