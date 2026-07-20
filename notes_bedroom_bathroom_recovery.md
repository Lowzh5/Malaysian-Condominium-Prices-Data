# Section 3.5 — Bedroom / Bathroom Recovery Log

Context: `src/data_preprocessing.py` Section 3.5 recovers missing `Bedroom`/`Bathroom`
values from the listing description (same evidence-based approach as Section 3.4)
instead of falling back to a blanket median. Only one row was missing both fields;
this is the console output printed under "Bedroom / Bathroom recovery from
description", kept here instead of a CSV so it doesn't clutter `data/processed/`.

| Ad List | Variable | Original | Description evidence | Final |
|---|---|---|---|---|
| 103803053 | Bedroom | NaN | 2 | 2 |
| 103803053 | Bathroom | NaN | 2 | 2 |

Description excerpt: "✅ 2 bedrooms + 2 bathroom".

The recovery uses a generic regex (`\d+\s*bed\s*rooms?` / `\d+\s*bath\s*rooms?`)
applied to any row with a missing value, not a hardcoded row — reproducible if a
future data pull has more such cases.
