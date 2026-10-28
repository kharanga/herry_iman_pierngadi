# `analytics/shot_types.py`

Maps each English shot type (`raw_shots.type`, which ingest has already translated
to English) to one of 8 shot-type groups. The groups are what the classifier predicts
and what the tendencies report on.

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
