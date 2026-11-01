# `analytics/shot_types.py`

Maps each English shot type (`raw_shots.type`, which ingest has already translated
to English) to one of 11 shot labels (`type_detail`), and each label to one of 8 shot-type groups
(`type_group`). The classifier predicts the labels; the tendencies report on the groups, and split
serves into short and long.

## Constants

`SHOT_TYPE_GROUPS` (`tuple[str, ...]`): `serve, clear, drop, smash, net, lift, drive, unknown`.

`TYPE_GROUP_BY_TYPE_EN` (`dict[str, str]`), 19 types:

| Group | Types |
|---|---|
| serve | short service, long service |
| clear | clear |
| drop | drop, passive drop |
| smash | smash, wrist smash |
| net | net shot, return net, cross-court net shot, rush |
| lift | lob, defensive return lob |
| drive | drive, driven flight, back-court drive, push, defensive return drive |
| unknown | unknown |

```python
>>> from analytics.shot_types import SHOT_TYPE_GROUPS, TYPE_GROUP_BY_TYPE_EN, UnmappedShotTypeError, type_group_for
>>> len(TYPE_GROUP_BY_TYPE_EN), sorted(set(TYPE_GROUP_BY_TYPE_EN.values())) == sorted(SHOT_TYPE_GROUPS)
(19, True)

```

`SHOT_TYPE_DETAILS` (`tuple[str, ...]`), 11 labels: `serve_short, serve_long, net, return_net, drive, push,
clear, drop, smash, lift, unknown`. Each belongs to exactly one group (`TYPE_GROUP_BY_TYPE_DETAIL`):

| Label | Group | ShuttleSet types |
|---|---|---|
| serve_short | serve | short service |
| serve_long | serve | long service |
| net | net | net shot, cross-court net shot, rush |
| return_net | net | return net |
| drive | drive | drive, driven flight, back-court drive, defensive return drive |
| push | drive | push |
| clear | clear | clear |
| drop | drop | drop, passive drop |
| smash | smash | smash, wrist smash |
| lift | lift | lob, defensive return lob |
| unknown | unknown | unknown |

`TYPE_DETAIL_BY_SPLIT_OFF_TYPE_EN` holds the four types with a label of their own; every other type's label
is its group's name. Why these four: `docs/results/classifier_fine_types.md`.

```python
>>> from analytics.shot_types import SHOT_TYPE_DETAILS, TYPE_GROUP_BY_TYPE_DETAIL
>>> len(SHOT_TYPE_DETAILS), set(TYPE_GROUP_BY_TYPE_DETAIL) == set(SHOT_TYPE_DETAILS)
(11, True)
>>> sorted(set(TYPE_GROUP_BY_TYPE_DETAIL.values())) == sorted(SHOT_TYPE_GROUPS)
True

```

## `type_detail_for(type_en)` and `group_of_detail(type_detail)`

- `type_detail_for` returns the label of an English type, `None` for `None`; `UnmappedShotTypeError` as `type_group_for`.
- `group_of_detail` returns the label's group, `None` for `None` or NaN; `KeyError` for a string that is not a label.
- `group_of_detail(type_detail_for(type_en)) == type_group_for(type_en)` for every type.

```python
>>> from analytics.shot_types import type_detail_for, group_of_detail
>>> [type_detail_for(type_en) for type_en in ("short service", "long service", "return net", "push", "rush", "wrist smash", None)]
['serve_short', 'serve_long', 'return_net', 'push', 'net', 'smash', None]
>>> all(group_of_detail(type_detail_for(type_en)) == type_group_for(type_en) for type_en in TYPE_GROUP_BY_TYPE_EN)
True
>>> group_of_detail("push"), group_of_detail(None)
('drive', None)

```

## `type_group_for(type_en)`

- `type_en` (`str | None`): an English shot type.
- Returns the group (`str`), or `None` if `type_en` is `None`.
- Errors: `UnmappedShotTypeError` (a `ValueError`) if `type_en` is not in the map.
  Matching is exact: case and spaces matter.

```python
>>> type_group_for("passive drop"), type_group_for("rush"), type_group_for("unknown"), type_group_for(None)
('drop', 'net', 'unknown', None)
>>> type_group_for("Smash")
Traceback (most recent call last):
    ...
analytics.shot_types.UnmappedShotTypeError: shot type has no group: 'Smash'

```

## `UnmappedShotTypeError(type_en)`

Raised by `type_group_for`. The `type_en` attribute holds the label that has no group.

```python
>>> error = UnmappedShotTypeError("bicycle kick")
>>> error.type_en, isinstance(error, ValueError)
('bicycle kick', True)

```
