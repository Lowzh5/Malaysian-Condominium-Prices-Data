# Section 3.11 — Train-Test Split Log

Context: `src/data_preprocessing.py` Section 3.11 runs after 3.7 (Feature
Selection) and before 3.10 (Feature Scaling) - the split has to exist before
any statistic-fitting step (imputation medians, scaler mean/std, State/
Property Type's rare-category thresholds), since those must be computed from
`X_train` only. Everything in this file after the split itself — including
the new "Part 3" encoding section below — runs on `X_train`/`X_test`, never
on the pre-split `df`.

## Split strategy: group-aware, not a plain `train_test_split`

3.7's post-drop duplicate re-check found 67 groups of row-identical listings
(136 rows) on the final 36-column dataset (still carrying raw `State`/
`Property Type` text at this point - see `notes_feature_selection_log.md`)
— different real listings (different `Ad List`/`Address`/`description`) that
only look identical once those identifying columns are dropped. A plain
random 80:20 split would tear some of these groups apart, landing test rows
that were exact duplicates of a training row, silently inflating any
evaluation metric computed on that split.

**First attempt (reverted): delete the 136 duplicate-looking rows.** Wrong
call — these are genuinely different, valid listings, not the same scrape
captured twice (which is what 3.2's exact-duplicate removal actually was).
Deleting them throws away real data to fix a problem that isn't about the
rows' validity.

**Fix kept**: `GroupShuffleSplit` (not `train_test_split`), grouped by a row-
content group ID computed right after 3.7's drop
(`df.groupby(list(df.columns)).ngroup()`). Every row in a group now lands on
the same side of the split — no group can ever straddle train/test — while
all 3793 rows are retained. This is the same principle already recorded for
the Stage-3 near-duplicate relistings in `notes_near_duplicate_relistings.md`:
*"use a group-aware train/test split... this avoids leakage independent of
the dedup decision."*

Result: `X_train (3028, 35)`, `X_test (765, 35)` — 35 columns at this point,
*before* Part 3's encoding/imputation below adds any new column. Actual test
proportion 0.202 (target 0.2) — the group constraint only ever moves 2-3 rows
together at a time, so the ratio barely shifts. Confirmed zero row-identical
groups split across train/test after the fix.

X and y are split in a single `gss.split(X, y, groups=row_group_ids)` call,
not two separate calls, so `X_train`/`y_train` stay row-aligned. No
`stratify=` — this is a regression target (`price`, continuous), not
classification.

## Part 3 — statistic-fitting steps (fit on `X_train` only, applied to `X_test`)

Everything below runs after the split, directly on `X_train`/`X_test` copies
— nothing here ever touches `df` or computes a statistic from the full,
pre-split data.

### State — rare-category merge + one-hot (fit on `X_train` only)

- NaN (genuine missing `Address`) filled with the explicit string
  `"Unknown"` in both `X_train` and `X_test` independently — this is a
  constant fill, not a fitted statistic, so no leakage risk either way (same
  treatment as Floor Range in 3.5).
- The <10-listings rare-category threshold **is** a fitted statistic (unlike
  the fill above): computed from `X_train['State'].value_counts()` only.
  This run: `['Kedah', 'Kelantan', 'Terengganu']` merged into `'Other'` in
  `X_train` — the *same* list is then applied to `X_test['State']` too, not
  recomputed from `X_test`.
- Both `X_train['State']` and `X_test['State']` are then cast to a
  `pd.Categorical` using `X_train`'s post-merge category list (not each
  encoded independently) — this guarantees `get_dummies(..., drop_first=True)`
  drops the *same* reference category on both sides. Any category `X_train`
  never saw at all (e.g. a singleton state that happened to land only in the
  test split) becomes NaN in `X_test` and gets an all-zero dummy row — the
  same behaviour as a proper unseen-category fallback in a deployed model.
  `X_test`'s resulting dummy frame is additionally `reindex`ed onto
  `X_train`'s dummy columns as a safety net.
- Result this run: 12 categories in `X_train` (11 real/merged states +
  `Unknown`) → 12 one-hot columns (`Johor` dropped as the alphabetically-first
  reference level).

### Property Type — rare-category merge + one-hot (fit on `X_train` only)

- Same fit-on-`X_train`-only reasoning as State above, `<20`-listings
  threshold. No missing values, so no `"Unknown"` fill needed.
- This run: `X_train` counts before merge — Condominium (1290), Apartment
  (1137), Service Residence (389), Flat (181), Studio (12), Others (10),
  Duplex (5), Townhouse Condo (4). `['Studio', 'Others', 'Duplex',
  'Townhouse Condo']` merged into `'Other'`.
- Same `pd.Categorical`-on-`X_train`'s-categories + `reindex` approach as
  State, for the same drop_first-consistency reason.
- Result this run: 4 one-hot columns (`Apartment` dropped as reference):
  `PropertyType_Condominium`, `PropertyType_Flat`, `PropertyType_Other`,
  `PropertyType_Service_Residence`.
- **Data quality observation for the report**: Duplex and Townhouse Condo
  appearing under "Property Type" in what the source describes as an
  apartment/condominium dataset suggests the source's own categorisation
  isn't fully clean — worth a mention, not something corrected here.

### Dropping the raw `State` / `Property Type` text columns

Only now, once their one-hot columns exist in `X_train`/`X_test`, are the raw
text columns dropped — from `X_train`/`X_test` individually, not from `df`
(which keeps them, as documented in `notes_feature_selection_log.md`).

## Missing-value imputation (median + indicator, fit on `X_train` only)

Five columns still carried real NaN going into this step — deliberately
left unfilled since 3.5 specifically to avoid computing a statistic from the
full dataset before a train/test split existed:

| Column | Train median | Train missing | Test missing |
|---|---|---|---|
| `Property Age` | 9.00 | 1550 | 379 |
| `# of Floors` | 20.00 | 1315 | 344 |
| `Total Units` | 462.00 | 1439 | 366 |
| `Parking Lot` | 1.00 | 928 | 234 |
| `Property Size` | 904.00 | 1 | 0 |
| `Floor_Range_Ordinal` | 2.00 | (see note) | (see note) |

Each of the first 5 gets its own missing-indicator flag
(`Property_Age_Missing`, `Num_Floors_Missing`, `Total_Units_Missing`,
`Parking_Lot_Missing`, `Property_Size_Missing`) before the median fill — an
imputed median and a genuinely-observed value aren't the same information,
so collapsing that distinction silently would hide it from the model (same
reasoning as `Facilities_Recorded`/`Floor_Range_Known` elsewhere in this
pipeline). `Floor_Range_Ordinal` is excluded from getting a *new* indicator —
it already has one (`Floor_Range_Known`, built in 3.9), so a second would be
redundant.

The median comes from `X_train[col].median()` only (pandas' `skipna=True`
default means it's computed correctly from the genuinely-observed values,
ignoring the NaN); the exact same value then fills `X_test` — `X_test`'s own
median is never computed, which is what avoids leaking test-set information
into the imputation.

Result: 0 remaining NaN in both `X_train` and `X_test`, confirmed by
assertion (not assumed). `df` itself (pre-split) intentionally still carries
the real NaN for these columns (plus `State`, still raw text) — see
`notes_final_summary_log.md`.

`X_train`/`X_test` are now `(3028, 54)`/`(765, 54)` — the 35 columns coming
out of the split, plus 12 State one-hot + 4 Property Type one-hot columns,
minus the 2 raw `State`/`Property Type` columns dropped above, plus 5
missing-indicator flags (35 + 12 + 4 − 2 + 5 = 54).

Saved: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`,
`train_test_split.pkl` (all still unscaled at this point, but fully encoded
and imputed — scaling happens next, in 3.10, saved under separate `_scaled`
filenames).
