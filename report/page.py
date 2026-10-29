from html import escape

from report.court_svg import court_svg

PAGE_CSS = """
:root { --ink: #1b1f23; --muted: #5c6670; --line: #d9dee3; --accent: #c8102e; --floor: #2f7d4f; --paper: #fafaf8; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--paper); color: var(--ink); }
body { font: 17px/1.45 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; padding: 16px; max-width: 640px; margin: 0 auto; }
h1 { font-size: 24px; line-height: 1.2; margin: 0 0 4px; }
h2 { font-size: 20px; margin: 28px 0 4px; }
h3 { font-size: 16px; margin: 18px 0 4px; color: var(--muted); }
p { margin: 0; }
.subtitle { color: var(--muted); font-size: 15px; overflow-wrap: anywhere; }
.lead { color: var(--muted); font-size: 15px; margin-bottom: 10px; }
ol.bullets { list-style: none; padding: 0; margin: 0; }
ol.bullets li { display: flex; gap: 12px; align-items: flex-start; padding: 12px 0; border-top: 1px solid var(--line); }
.bullet-body { flex: 1 1 auto; min-width: 0; }
.claim { font-weight: 600; }
.meta { margin-top: 6px; font-size: 14px; color: var(--muted); }
.evidence { font-weight: 700; color: var(--ink); }
.tag { display: inline-block; margin-left: 6px; padding: 0 6px; border-radius: 4px; font-size: 13px; line-height: 19px; border: 1px solid var(--line); vertical-align: 1px; }
.tag.predicted { color: var(--accent); border-color: var(--accent); }
svg.court { flex: 0 0 auto; }
svg.court .floor { fill: var(--floor); }
svg.court .zone { fill: var(--accent); fill-opacity: 0.7; }
svg.court .zone.from { fill: #f2c200; fill-opacity: 0.8; }
svg.court line.arrow { stroke: #111111; stroke-width: 1.6; }
svg.court .arrowhead { fill: #111111; }
svg.court line { stroke: #ffffff; stroke-width: 1.2; }
svg.court line.net { stroke: #111111; stroke-width: 2; }
.empty { color: var(--muted); padding: 12px 0; border-top: 1px solid var(--line); }
footer { margin-top: 28px; padding-top: 12px; border-top: 1px solid var(--line); color: var(--muted); font-size: 13px; }
footer p { margin-bottom: 6px; }
"""


def sets_text(sets: list[int] | None) -> str:
    if not sets:
        return "whole match"
    return "set " + ", ".join(str(set_no) for set_no in sets) if len(sets) > 1 else f"set {sets[0]}"


def tags_html(bullet: dict) -> str:
    tags = []
    if bullet["label_source"] != "labels":
        tags.append('<span class="tag predicted">predicted shot types</span>')
    return "".join(tags)


def bullet_html(bullet: dict, hitter_name: str) -> str:
    diagram = ""
    if bullet.get("reply_rects") is not None:
        diagram = court_svg(
            None, f"court diagram, you at the bottom, arrow from where the shot was hit to where it landed: {escape(bullet['text'])}", bullet["reply_rects"]
        )
    elif bullet["zone_rect"] is not None:
        diagram = court_svg(bullet["zone_rect"], f"court diagram, zone shaded, you at the bottom: {escape(bullet['text'])}")
    return (
        "<li>"
        '<div class="bullet-body">'
        f'<p class="claim">{escape(bullet["text"])}</p>'
        f'<p class="meta"><span class="evidence">{escape(bullet["evidence"])}</span> for {escape(hitter_name)}'
        f'{tags_html(bullet)}</p>'
        "</div>"
        f"{diagram}"
        "</li>"
    )


def group_html(group: dict, hitter_name: str) -> str:
    bullets = "".join(bullet_html(bullet, hitter_name) for bullet in group["bullets"])
    return f'<h3>{escape(group["heading"])}</h3><ol class="bullets">{bullets}</ol>'


def section_html(section: dict, heading: str, lead: str) -> str:
    if section["groups"]:
        body = "".join(group_html(group, section["name"]) for group in section["groups"])
    else:
        body = '<p class="empty">No shots counted for this player yet.</p>'
    return f'<section><h2>{escape(heading)}</h2><p class="lead">{escape(lead)}</p>{body}</section>'


def footer_html(report: dict) -> str:
    bullets = report["opponent"]["bullets"] + report["me"]["bullets"]
    label_sources = {bullet["label_source"] for bullet in bullets}
    if label_sources <= {"labels"}:
        types_note = "Shot types are the labelled types from the match data."
    elif label_sources == {"predictions"}:
        types_note = "Shot types are predicted by the classifier from court positions, not labelled by a person."
    else:
        types_note = "Bullets marked “predicted shot types” use the classifier's predicted types; the others use labelled types."
    return (
        "<footer>"
        "<p>\"k of n\" counts the player's own shots: in n shots from that situation, they made this choice k times. "
        "Every choice is listed, however rare, with no comparison and no ranking.</p>"
        f"<p>{types_note}</p>"
        "<p>Court diagrams: you are at the bottom, the opponent at the top; the shaded zone is the one the bullet is about. "
        "Reply diagrams show the other player's shot: hit from the yellow zone, landing in the red zone, with an arrow. "
        "Winners and errors have no zone, so no diagram.</p>"
        "</footer>"
    )


def render_page(report: dict) -> str:
    opponent, me = report["opponent"], report["me"]
    title = f"{opponent['name']}: shot counts"
    subtitle = f"{escape(report['title'])} · {escape(me['name'])} vs {escape(opponent['name'])} · {escape(sets_text(report['sets']))}"
    return (
        "<!doctype html>"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title>"
        f"<style>{PAGE_CSS}</style></head><body>"
        f"<header><h1>{escape(title)}</h1><p class=\"subtitle\">{subtitle}</p></header>"
        + section_html(opponent, f"Opponent: {opponent['name']}", "Every choice they made, counted from their own shots.")
        + section_html(me, f"Me: {me['name']}", "Every choice I made, counted from my own shots.")
        + footer_html(report)
        + "</body></html>"
    )
