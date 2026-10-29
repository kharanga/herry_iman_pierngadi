from pathlib import Path

import duckdb

from analytics.build_shots import LEFT_HANDED_PLAYER_NAMES
from analytics.tendencies import DATABASE_PATH, FAMILIES, all_observations, tendencies_from_observations
from report.court_svg import bullet_zone_rect

OWN_PLAYER_NAMES = {"me": "Me", "opp": "Opponent"}
FAMILY_HEADINGS = {
    "zone_mix": "Shots by court zone",
    "response": "Replies to the other player's shot",
    "serve_placement": "Serves",
    "winning_shot": "Winning shots",
    "error_shot": "Shots that ended in an error",
    "error_reason": "How errors happened",
}

MATCH_SQL = """
SELECT matches.tournament, matches.round, matches.year, matches.winner, matches.loser
FROM matches
WHERE matches.match_id = ?
"""

HAS_MATCHES_TABLE_SQL = """
SELECT COUNT(*)
FROM information_schema.tables
WHERE information_schema.tables.table_name = 'matches'
"""

MATCH_PLAYERS_SQL = """
SELECT shots.player, COUNT(*) AS shot_count
FROM shots
WHERE shots.match_id = ?
GROUP BY shots.player
ORDER BY shots.player
"""


class UnknownMatchError(ValueError):
    def __init__(self, match_id: str):
        super().__init__(f"no shots for match {match_id!r}")
        self.match_id = match_id


class UnknownPlayerError(ValueError):
    def __init__(self, match_id: str, player: str, players: list[str]):
        super().__init__(f"player {player!r} is not in match {match_id!r}; players are {players}")
        self.player = player
        self.players = players


def match_players(connection: duckdb.DuckDBPyConnection, match_id: str) -> list[str]:
    players = [player for player, _ in connection.execute(MATCH_PLAYERS_SQL, [match_id]).fetchall()]
    if not players:
        raise UnknownMatchError(match_id)
    return players


def has_matches_table(connection: duckdb.DuckDBPyConnection) -> bool:
    return connection.execute(HAS_MATCHES_TABLE_SQL).fetchone()[0] > 0


def match_title(connection: duckdb.DuckDBPyConnection, match_id: str) -> tuple[str, dict[str, str]]:
    match_row = connection.execute(MATCH_SQL, [match_id]).fetchone() if has_matches_table(connection) else None
    if match_row is None:
        return match_id, dict(OWN_PLAYER_NAMES)
    tournament, round_name, year, winner, loser = match_row
    return f"{tournament} {round_name}, {year}", {"A": winner, "B": loser}


def default_me(players: list[str]) -> str:
    for preferred in ("me", "A"):
        if preferred in players:
            return preferred
    return players[0]


def opponent_of(me: str, players: list[str]) -> str:
    others = [player for player in players if player != me]
    if len(others) != 1:
        raise ValueError(f"expected exactly one opponent, found {others}")
    return others[0]


def evidence_text(tendency: dict) -> str:
    return f"{tendency['k']} of {tendency['n']}"


def bullet_from_tendency(tendency: dict, hitter_is_me: bool, me_left_handed: bool, opponent_left_handed: bool) -> dict:
    zone = tendency["condition"] if tendency["family"] == "zone_mix" else tendency["outcome"]
    return {
        "id": tendency["id"],
        "family": tendency["family"],
        "text": tendency["text"],
        "evidence": evidence_text(tendency),
        "k": tendency["k"],
        "n": tendency["n"],
        "rate": tendency["rate"],
        "label_source": tendency["label_source"],
        "zone_rect": bullet_zone_rect(hitter_is_me, tendency["family"], zone, me_left_handed, opponent_left_handed),
        "sets": tendency["sets"],
    }


def rewrite_bullets(bullets: list[dict]) -> list[dict]:
    return bullets


def family_groups(bullets: list[dict]) -> list[dict]:
    return [
        {"family": family, "heading": FAMILY_HEADINGS[family], "bullets": [bullet for bullet in bullets if bullet["family"] == family]}
        for family in FAMILIES
        if any(bullet["family"] == family for bullet in bullets)
    ]


def player_section(observations, match_id: str, player: str, name: str, hitter_is_me: bool, left_handed: dict[str, bool], sets) -> dict:
    tendencies = tendencies_from_observations(observations, match_id, player, sets)
    bullets = rewrite_bullets(
        [bullet_from_tendency(tendency, hitter_is_me, left_handed["me"], left_handed["opponent"]) for tendency in tendencies]
    )
    return {
        "player": player,
        "name": name,
        "bullets": bullets,
        "groups": family_groups(bullets),
        "sets": sorted({set_no for tendency in tendencies for set_no in tendency["sets"]}),
    }


def build_report(
    match_id: str,
    me: str | None = None,
    database_path: Path = DATABASE_PATH,
    sets: tuple[int, ...] | None = None,
    left_handed_names: frozenset[str] = LEFT_HANDED_PLAYER_NAMES,
) -> dict:
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        players = match_players(connection, match_id)
        title, names = match_title(connection, match_id)
    finally:
        connection.close()
    me = default_me(players) if me is None else me
    if me not in players:
        raise UnknownPlayerError(match_id, me, players)
    opponent = opponent_of(me, players)
    names = {player: names.get(player, player) for player in players}
    left_handed = {"me": names[me] in left_handed_names, "opponent": names[opponent] in left_handed_names}
    observations = all_observations(database_path)
    return {
        "match_id": match_id,
        "title": title,
        "sets": list(sets) if sets else None,
        "opponent": player_section(observations, match_id, opponent, names[opponent], False, left_handed, sets),
        "me": player_section(observations, match_id, me, names[me], True, left_handed, sets),
    }
