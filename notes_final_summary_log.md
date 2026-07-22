# Section 3.12 — Final Dataset Structure Summary Log

Context: `src/data_preprocessing.py` Section 3.12 produces no new data
processing — it's a pure accounting step. Every number in it is read from a
variable already computed earlier in the script (`RAW_COLUMNS`,
`cols_no_engineering`, `cols_replaced_by_engineering`, `X_train`, `X_test`,
`df`, `row_group_ids`), not re-typed by hand, so the summary can't silently
drift out of sync with what the pipeline actually did further up the script.

## Feature accounting

`RAW_COLUMNS` is snapshotted immediately after the raw CSV is loaded (before
any processing), so at the end of the script:

- **Raw features retained as-is** = `[c for c in RAW_COLUMNS if c in df.columns]`
- **Raw features dropped** = `[c for c in RAW_COLUMNS if c not in df.columns]`
  (should match `cols_no_engineering + cols_replaced_by_engineering`'s length)
- **Engineered/encoded features created** = `[c for c in df.columns if c not in RAW_COLUMNS]`

This avoids the failure mode of a hand-built count (e.g. manually writing
`len(state_dummies.columns) + len(property_type_dummies.columns) + ... + 4`
for the remaining binary features) — any hardcoded "+4" for features like
`Is_Non_Bumi_Lot`/`Freehold Indicator` is exactly the kind of number that
goes stale the next time a feature is added or dropped upstream. Diffing
`RAW_COLUMNS` against `df.columns` needs no maintenance.

Result: 7 raw retained, 25 dropped, 43 engineered (7 + 25 = 32 original
columns; 43 + 7 = 50 final columns — both check out against the earlier
section logs).

## Missing values — verified, not assumed

`X_train.isna().sum().sum() + X_test.isna().sum().sum()` = 0, checked after
3.11's imputation, not assumed because a `fillna()` line was written earlier
in the script. `df` itself (pre-split) still legitimately carries real NaN
in 6 columns (`Property Size`, `# of Floors`, `Total Units`, `Parking Lot`,
`Property Age`, `Floor_Range_Ordinal`) — this is intentional, not a bug: 3.11
imputes these *after* the split (median fit on `X_train` only), so `df`
itself is left honest for EDA purposes rather than silently filled with a
full-dataset statistic that would amount to a leakage shortcut.

## Duplicate rows — re-verified on the final df, not assumed clean from 3.2

3.2 de-duplicated the *original* 32-column dataset; that result doesn't
automatically carry forward once 26 columns are later dropped in 3.7. Re-
checked on the final df and found 67 groups of row-identical listings (136
rows) — see `notes_feature_selection_log.md` and `notes_train_test_split_log.md`
for the full investigation and the group-aware-split fix.

The summary table reports this distinction explicitly rather than a single
`df.duplicated().sum()` number, because that number alone (69) would read as
"an unfinished cleanup step" when it's the opposite — deliberately retained,
real data:

| Item | Result | What it means |
|---|---|---|
| Row-identical groups retained (not deleted) | 67 | Evidence the rows were kept on purpose, per the 3.7 note - not a leftover bug |
| Row-identical groups split across train/test | 0 | Evidence the actual leakage risk was resolved (via `GroupShuffleSplit` in 3.11), computed by tagging `X_train`/`X_test` rows with their source and checking whether any row-identical group has both sources present |

## Reconciling 49 vs. 54 — a gap that looked like an error but wasn't

An external check of this pipeline caught that `df`'s pre-split feature
count (49) doesn't match `X_train.shape[1]` (54), and flagged it correctly:
not a bug, but a real summary-table gap. 3.11's imputation step adds a
`*_Missing` indicator flag per imputed column (`Property_Age_Missing`,
`Num_Floors_Missing`, `Total_Units_Missing`, `Parking_Lot_Missing`,
`Property_Size_Missing` — 5 columns) that don't exist in `df`, since `df` is
a pre-split snapshot and those flags are only created after the split. The
original table reported 49 without ever surfacing that 54 is the real,
larger number a reader would see in `X_train.shape` - a legitimate
inconsistency, since nothing in the printed output explained the gap.

Fixed by adding two explicit rows, both computed as a set difference
(`[c for c in X_train.columns if c not in df.drop(columns=['price']).columns]`)
rather than hardcoded as "+5" - the same principle as everywhere else in this
section, so this number can't go stale if a future edit changes how many
indicator columns get added.

## Final summary table (as printed by the script)

| Item | Result |
|---|---|
| Final number of rows (pre-split df) | 3793 |
| Final number of features (X, pre-split) | 49 |
| Numerical features | 50 |
| Non-numeric features remaining | 0 |
| Raw features retained as-is | 7 |
| Raw features dropped (Section 3.7) | 25 |
| Engineered/encoded features created (3.8+3.9) | 43 |
| Missing-value indicator flags added (Section 3.11) | 5 |
| Final number of features in X_train/X_test | 54 |
| Training set shape | (3030, 54) |
| Testing set shape | (763, 54) |
| Remaining missing values (X_train + X_test) | 0 |
| Row-identical groups retained (not deleted - see 3.7 note) | 67 |
| Row-identical groups split across train/test (should be 0) | 0 |

(49 + 5 = 54, reconciling `df`'s pre-split feature count against
`X_train`/`X_test`'s actual column count.)
