# Section 3.4 — Property Size Correction Log

Context: `src/data_preprocessing.py` Section 3.4 flags `Property Size` values that are
either below a physically possible lower bound (< 100 sq.ft.) or a digit-shift
artefact (~10x/100x/1000x the description's independently stated size, only checked
above 8,000 sq.ft.). Four records were corrected; this is the console output printed
under "Property Size correction log", kept here instead of a CSV so it doesn't clutter
`data/processed/`.

| Ad List | Original Property Size | Description Property Size | Final Property Size | Reason |
|---|---|---|---|---|
| 101812262 | 122,774 | 1,227.74 | 1,228 | Digit-shift error |
| 102897216 | 14,500 | 1,450.00 | 1,450 | Digit-shift error |
| 103423738 | 9 | 991.00 | 991 | Below screening threshold |
| 103788197 | 1 | 850.00 | 850 | Below screening threshold |

Three other large values (9,376 / 9,800 / 17,611 sq.ft.) matched neither rule and were
left untouched, deferred to Section 3.6 as genuine extreme values.
