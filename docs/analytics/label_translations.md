# `ingest/label_translations.py`

The Chinese-to-English maps for ShuttleSet's text labels, in one place.
`ingest/load_shuttleset.py` uses them to translate `raw_shots.type`,
`raw_shots.lose_reason` and `raw_shots.win_reason`, so the database holds no Chinese.
Other code (for example image captions in the vision track) can import the same maps:

```
from ingest.label_translations import SHOT_TYPE_ENGLISH_BY_CHINESE
```

## Constants

| Name | Type | Value |
|---|---|---|
| `SHOT_TYPE_ENGLISH_BY_CHINESE` | `dict[str, str]` | 20 Chinese shot types -> 19 English labels (both passive-drop spellings, `過度切球` and `過渡切球`, become `passive drop`; `未知球種` becomes `unknown`) |
| `LOSE_REASON_ENGLISH_BY_CHINESE` | `dict[str, str]` | 7 rally-end reasons, from the loser's side |
| `WIN_REASON_ENGLISH_BY_CHINESE` | `dict[str, str]` | 7 rally-end reasons, from the winner's side |
| `ENGLISH_BY_CHINESE_BY_RAW_SHOTS_COLUMN` | `dict[str, dict[str, str]]` | `type`, `lose_reason`, `win_reason` -> the map for that column |

Shot types:

| Chinese | English |
|---|---|
| 放小球 | net shot |
| 擋小球 | return net |
| 殺球 | smash |
| 點扣 | wrist smash |
| 挑球 | lob |
| 防守回挑 | defensive return lob |
| 長球 | clear |
| 平球 | drive |
| 小平球 | driven flight |
| 後場抽平球 | back-court drive |
| 切球 | drop |
| 過度切球, 過渡切球 | passive drop |
| 推球 | push |
| 撲球 | rush |
| 防守回抽 | defensive return drive |
| 勾球 | cross-court net shot |
| 發短球 | short service |
| 發長球 | long service |
| 未知球種 | unknown |

Rally-end reasons (each row is the same event seen from each side):

| `lose_reason` Chinese | `lose_reason` English | `win_reason` Chinese | `win_reason` English |
|---|---|---|---|
| 對手落地致勝 | opponent winner | 落地致勝 | winner |
| 出界 | out | 對手出界 | opponent out |
| 掛網 | net | 對手掛網 | opponent net |
| 未過網 | not over net | 對手未過網 | opponent not over net |
| 落點判斷失誤 | misjudged landing | 對手落點判斷失誤 | opponent misjudged landing |
| 對手落地判斷失誤 | misjudged opponent landing | 落地判斷失誤 | landed, opponent misjudged |
| 犯規 | foul | 對手犯規 | opponent foul |

The two "misjudged" pairs are kept as separate labels, exactly as the source distinguishes them.

```python
>>> from ingest.label_translations import (
...     ENGLISH_BY_CHINESE_BY_RAW_SHOTS_COLUMN,
...     LOSE_REASON_ENGLISH_BY_CHINESE,
...     SHOT_TYPE_ENGLISH_BY_CHINESE,
...     WIN_REASON_ENGLISH_BY_CHINESE,
... )
>>> SHOT_TYPE_ENGLISH_BY_CHINESE["殺球"]
'smash'
>>> SHOT_TYPE_ENGLISH_BY_CHINESE["過度切球"], SHOT_TYPE_ENGLISH_BY_CHINESE["過渡切球"]
('passive drop', 'passive drop')
>>> len(SHOT_TYPE_ENGLISH_BY_CHINESE), len(set(SHOT_TYPE_ENGLISH_BY_CHINESE.values()))
(20, 19)
>>> LOSE_REASON_ENGLISH_BY_CHINESE["出界"], WIN_REASON_ENGLISH_BY_CHINESE["對手出界"]
('out', 'opponent out')
>>> sorted(ENGLISH_BY_CHINESE_BY_RAW_SHOTS_COLUMN)
['lose_reason', 'type', 'win_reason']

```

## `english_label_by_source_label(column_name)`

Every label accepted in a `raw_shots` column, mapped to its English label: the
Chinese labels from the column's map, plus each English label mapped to itself
(so rows that are already English, such as test fixtures or my own footage, pass through unchanged).

Inputs:
- `column_name` (`str`): `type`, `lose_reason` or `win_reason`.

Output: `dict[str, str]`, accepted label -> English label.

Errors:
- `KeyError` if `column_name` is not one of the three translated columns.

```python
>>> from ingest.label_translations import english_label_by_source_label
>>> win_reason_labels = english_label_by_source_label("win_reason")
>>> win_reason_labels["落地判斷失誤"], win_reason_labels["winner"]
('landed, opponent misjudged', 'winner')
>>> len(win_reason_labels)
14
>>> english_label_by_source_label("player")
Traceback (most recent call last):
...
KeyError: 'player'

```

## `translate_label(column_name, label)`

Translates one label from a translated column.

Inputs:
- `column_name` (`str`): `type`, `lose_reason` or `win_reason`.
- `label` (`str | None`): a Chinese label, an English label, or `None`.

Output: `str | None`, the English label; `None` stays `None`.

Errors:
- `UnmappedLabelError` if `label` is in neither the Chinese nor the English labels of the column.
- `KeyError` if `column_name` is not a translated column.

```python
>>> from ingest.label_translations import UnmappedLabelError, translate_label
>>> translate_label("type", "未知球種")
'unknown'
>>> translate_label("type", "lob")
'lob'
>>> translate_label("lose_reason", None) is None
True
>>> translate_label("type", "飛球")
Traceback (most recent call last):
...
ingest.label_translations.UnmappedLabelError: raw_shots.type has a label with no English translation: '飛球'

```

## `UnmappedLabelError`

Subclass of `ValueError`, raised when a label has no English translation.
Its message names the column and the label; `column_name` and `label` are attributes.

```python
>>> error = UnmappedLabelError("win_reason", "新原因")
>>> error.column_name, error.label, isinstance(error, ValueError)
('win_reason', '新原因', True)

```
