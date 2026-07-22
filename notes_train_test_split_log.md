# Section 3.11 — Train-Test Split Log

Context: `src/data_preprocessing.py` Section 3.11 runs after 3.7 (Feature
Selection) and before 3.10 (Feature Scaling) - the split has to exist before
any statistic-fitting step (imputation medians, scaler mean/std), since those
must be computed from `X_train` only.

## Split strategy: group-aware, not a plain `train_test_split`

3.7's post-drop duplicate re-check found 67 groups of row-identical listings
(136 rows) on the final 50-column dataset — different real listings
(different `Ad List`/`Address`/`description`) that only look identical once
those identifying columns are dropped. A plain random 80:20 split would tear
some of these groups apart: verified 20 of the 67 groups landed split across
train/test, meaning 26 test rows were exact duplicates of a training row,
silently inflating any evaluation metric computed on that split.

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

Result: `X_train (3030, 49)`, `X_test (763, 49)`. Actual test proportion
0.201 (target 0.2) — the group constraint only ever moves 2-3 rows together
at a time, so the ratio barely shifts. Confirmed zero row-identical groups
split across train/test after the fix.

X and y are split in a single `gss.split(X, y, groups=row_group_ids)` call,
not two separate calls, so `X_train`/`y_train` stay row-aligned. No
`stratify=` — this is a regression target (`price`, continuous), not
classification.

## Missing-value imputation (median + indicator, fit on `X_train` only)

Five columns still carried real NaN going into this section — deliberately
left unfilled since 3.5 specifically to avoid computing a statistic from the
full dataset before a train/test split existed:

| Column | Train median | Train missing | Test missing |
|---|---|---|---|
| `Property Age` | 9.00 | 1555 | 374 |
| `# of Floors` | 20.00 | 1317 | 342 |
| `Total Units` | 462.00 | 1441 | 364 |
| `Parking Lot` | 1.00 | 941 | 221 |
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
the real NaN for these columns — see `notes_final_summary_log.md`.

Saved: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`,
`train_test_split.pkl` (all still unscaled at this point — scaling happens
next, in 3.10, saved under separate `_scaled` filenames).
